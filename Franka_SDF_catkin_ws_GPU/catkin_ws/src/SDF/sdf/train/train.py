#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.



import torch
import numpy as np
import json
import os
from datetime import datetime
import argparse
import cv2

from sdf import visualisation
from sdf.modules import trainer


def train(
    device,
    config_file,
    chkpt_load_file=None,
    incremental=True,
    # vis
    show_obj=False,
    update_im_freq=50,
    update_mesh_freq=200,
    grid_dim = 200, 
    # opt
    extra_opt_steps = 400,
    # save
    save_path=None,
):
    # init trainer-------------------------------------------------------------
    sdf_trainer = trainer.Trainer(
        device,
        config_file,
        chkpt_load_file=chkpt_load_file,
        incremental=incremental,
        grid_dim = grid_dim
    )

    # saving init--------------------------------------------------------------
    save = save_path is not None
    if save:
        with open(save_path + "/config.json", "w") as outfile:
            json.dump(sdf_trainer.config, outfile, indent=4)

        if sdf_trainer.save_checkpoints:
            checkpoint_path = os.path.join(save_path, "checkpoints")
            os.makedirs(checkpoint_path)
        if sdf_trainer.save_slices:
            slice_path = os.path.join(save_path, 'slices')
            os.makedirs(slice_path)
            sdf_trainer.write_slices(
                slice_path, prefix="0.000_", include_gt=True)
        if sdf_trainer.save_meshes:
            mesh_path = os.path.join(save_path, 'meshes')
            os.makedirs(mesh_path)

    # eval init--------------------------------------------------------------
    if sdf_trainer.do_eval:
        res = {}
        if sdf_trainer.sdf_eval:
            res['sdf_eval'] = {}
        if sdf_trainer.mesh_eval:
            res['mesh_eval'] = {}
    if sdf_trainer.do_vox_comparison:
        vox_res = {}

    last_eval = 0

    # live vis init--------------------------------------------------------------
    # if sdf_trainer.live:
    kf_vis = None
    cv2.namedWindow('sdf keyframes', cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow("sdf keyframes", 100, 700)

    # main  loop---------------------------------------------------------------
    print("Starting training for max", sdf_trainer.n_steps, "steps...")
    size_dataset = len(sdf_trainer.scene_dataset)

    break_at = -1

    t = 0
    for _ in range(sdf_trainer.n_steps):
        # break at end -------------------------------------------------------
        if t == break_at and len(sdf_trainer.eval_times) == 0:
            if save:
                if sdf_trainer.save_slices:
                    sdf_trainer.write_slices(slice_path)

                if sdf_trainer.do_eval:
                    kf_list = sdf_trainer.frames.frame_id[:-1].tolist()
                    res['kf_indices'] = kf_list
                    with open(os.path.join(save_path, 'res.json'), 'w') as f:
                        json.dump(res, f, indent=4)

            break

        # get/add data---------------------------------------------------------
        finish_optim = \
            sdf_trainer.steps_since_frame == sdf_trainer.optim_frames
        if incremental and (finish_optim or t == 0):
            # After n steps with new frame, check whether to add it to kf set.
            if t == 0:
                add_new_frame = True
            else:
                add_new_frame = sdf_trainer.check_keyframe_latest()

            if add_new_frame:
                new_frame_id = sdf_trainer.get_latest_frame_id()
                if new_frame_id >= size_dataset:
                    break_at = t + extra_opt_steps
                    print(f"**************************************",
                          "End of sequence, runnining {extra_opt_steps} steps",
                          "**************************************")
                else:
                    print("Total step time", sdf_trainer.tot_step_time)
                    print("frame______________________", new_frame_id)

                    frame_data = sdf_trainer.get_data([new_frame_id])
                    sdf_trainer.add_frame(frame_data)

                    if t == 0:
                        sdf_trainer.last_is_keyframe = True
                        sdf_trainer.optim_frames = 200

            if t == 0 or (sdf_trainer.last_is_keyframe and not add_new_frame):
                kf_vis = visualisation.draw.add_im_to_vis(
                    kf_vis, sdf_trainer.frames.im_batch_np[-1], reduce_factor=6)
                cv2.imshow('sdf keyframes', kf_vis)
                cv2.waitKey(1)

        # optimisation step---------------------------------------------
        losses, step_time = sdf_trainer.step()
        if not sdf_trainer.live:
            status = [k + ': {:.6f}  '.format(losses[k]) for k in losses.keys()]
            status = "".join(status) + '-- Step time: {:.2f}  '.format(step_time)
            print(t, status)

        # visualisation----------------------------------------------------------
        if (
            not sdf_trainer.live and update_im_freq is not None and
            (t % update_im_freq == 0)
        ):
            display = {}
            sdf_trainer.update_vis_vars()
            display["keyframes"] = sdf_trainer.frames_vis()
            # display["slices"] = sdf_trainer.slices_vis()
            if show_obj:
                obj_slices_viz = sdf_trainer.obj_slices_vis()

            if update_mesh_freq is not None and (t % update_mesh_freq == 0):
                scene = sdf_trainer.draw_3D(
                    show_pc=False, show_mesh=t > 200, draw_cameras=True,
                    camera_view=False, show_gt_mesh=False)
                if show_obj:
                    try:
                        obj_scene = sdf_trainer.draw_obj_3D()
                    except:
                        print('Failed to draw mesh')

            display["scene"] = scene
            if show_obj and obj_scene is not None:
                display["obj_scene"] = obj_scene
            if show_obj and obj_slices_viz is not None:
                display["obj_slices"] = obj_slices_viz
            yield display

        t += 1

        # render live view ----------------------------------------------------
        #view_freq = 10
        view_freq = 10
        if t % view_freq == 0 and sdf_trainer.live:
            latest_vis = sdf_trainer.latest_frame_vis()
            print(latest_vis[0].shape, latest_vis[1].shape,latest_vis[2].shape)
            # #cv2.imshow('sdf (frame rgb, depth), (rendered normals, depth)', latest_vis[0])
            # cv2.imshow('sdf (frame rgb, depth), (rendered normals, depth)', latest_vis[1])
            # #cv2.imshow('sdf (frame rgb, depth), (rendered normals, depth)', latest_vis[2])

            vis_rgb, vis_normals, _ = latest_vis  # Ignore vis_depth

            # Concatenate the images horizontally
            horizontal_concatenation = np.hstack((vis_rgb, vis_normals))

            # Show the concatenated image
            cv2.imshow('sdf (frame rgb, depth), (rendered normals, depth)', horizontal_concatenation)

            key = cv2.waitKey(5)

            # active keyframes vis
            # kf_active_vis = sdf_trainer.keyframe_vis(reduce_factor=4)
            # cv2.imshow('sdf keyframes v2', kf_active_vis)
            # cv2.waitKey(1)

            if key == 115:
                # s key to show SDF slices
                sdf_trainer.view_sdf()

            if key == 99:
                # c key clears keyframes
                print('Clearing keyframes...')
                sdf_trainer.clear_keyframes()
                kf_vis = None
                t = 0

        # save ----------------------------------------------------------------
        if save and len(sdf_trainer.save_times) > 0:
            if sdf_trainer.tot_step_time > sdf_trainer.save_times[0]:
                save_t = f"{sdf_trainer.save_times.pop(0):.3f}"
                print(
                    f"Saving at {save_t}s",
                    f" --  model {sdf_trainer.save_checkpoints} ",
                    f"slices {sdf_trainer.save_slices} ",
                    f"mesh {sdf_trainer.save_meshes} "
                )

                if sdf_trainer.save_checkpoints:
                    torch.save(
                        {
                            "step": t,
                            "model_state_dict":
                                sdf_trainer.sdf_map.state_dict(),
                            "optimizer_state_dict":
                                sdf_trainer.optimiser.state_dict(),
                            "loss": losses['total_loss'].item(),
                        },
                        os.path.join(
                            checkpoint_path, "step_" + save_t + ".pth")
                    )

                if sdf_trainer.save_slices:
                    sdf_trainer.write_slices(
                        slice_path, prefix=save_t + "_",
                        include_gt=False, include_diff=False,
                        include_chomp=False, draw_cams=True)

                if sdf_trainer.save_meshes and sdf_trainer.tot_step_time > 0.4:
                    sdf_trainer.write_mesh(mesh_path + f"/{save_t}.ply")

        # evaluation -----------------------------------------------------

        if len(sdf_trainer.eval_times) > 0:
            if sdf_trainer.tot_step_time > sdf_trainer.eval_times[0]:
                eval_t = sdf_trainer.eval_times[0]
                print("voxblox eval at ----------------------------->", eval_t)
                vox_res[sdf_trainer.tot_step_time] = sdf_trainer.eval_fixed()
                if save:
                    with open(os.path.join(save_path, 'vox_res.json'), 'w') as f:
                        json.dump(vox_res, f, indent=4)

        elapsed_eval = sdf_trainer.tot_step_time - last_eval
        if sdf_trainer.do_eval and elapsed_eval > sdf_trainer.eval_freq_s:
            last_eval = sdf_trainer.tot_step_time - \
                sdf_trainer.tot_step_time % sdf_trainer.eval_freq_s

            if sdf_trainer.sdf_eval and sdf_trainer.gt_sdf_file is not None:
                visible_res = sdf_trainer.eval_sdf(visible_region=True)
                obj_errors = sdf_trainer.eval_object_sdf()

                print("Time ---------->", sdf_trainer.tot_step_time)
                print("Visible region SDF error: {:.4f}".format(
                    visible_res["av_l1"]))
                print("Objects SDF error: ", obj_errors)

                if not incremental:
                    full_vol_res = sdf_trainer.eval_sdf(visible_region=False)
                    print("Full region SDF error: {:.4f}".format(
                        full_vol_res["av_l1"]))
                if save:
                    res['sdf_eval'][t] = {
                        'time': sdf_trainer.tot_step_time,
                        'rays': visible_res,
                    }
                    if obj_errors is not None:
                        res['sdf_eval'][t]['objects_l1'] = obj_errors

            if sdf_trainer.mesh_eval:
                acc, comp = sdf_trainer.eval_mesh()
                print("Mesh accuracy and completion:", acc, comp)
                if save:
                    res['mesh_eval'][t] = {
                        'time': sdf_trainer.tot_step_time,
                        'acc': acc,
                        'comp': comp,
                    }

            if save:
                with open(os.path.join(save_path, 'res.json'), 'w') as f:
                    json.dump(res, f, indent=4)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)

    parser = argparse.ArgumentParser(description="sdf.")
    parser.add_argument("--config", type=str, required = True, help="input json config")
    parser.add_argument(
        "-ni",
        "--no_incremental",
        action="store_false",
        help="disable incremental SLAM option",
    )
    parser.add_argument(
        "-hd", "--headless",
        action="store_true",
        help="run headless (i.e. no visualisations)"
    )
    args, _ = parser.parse_known_args()  # ROS adds extra unrecongised args

    config_file = args.config
    headless = args.headless
    incremental = args.no_incremental
    chkpt_load_file = None

    # vis
    show_obj = False
    update_im_freq = 40
    update_mesh_freq = 200
    if headless:
        update_im_freq = None
        update_mesh_freq = None

    # save
    save = False
    if save:
        now = datetime.now()
        time_str = now.strftime("%m-%d-%y_%H-%M-%S")
        save_path = "../../results/sdf/" + time_str
        os.mkdir(save_path)
    else:
        save_path = None

    scenes = train(
        device,
        config_file,
        chkpt_load_file=chkpt_load_file,
        incremental=incremental,
        # vis
        show_obj=show_obj,
        update_im_freq=update_im_freq,
        update_mesh_freq=update_mesh_freq,
        # save
        save_path=save_path,
    )

    if headless:
        on = True
        while on:
            try:
                out = next(scenes)
            except StopIteration:
                on = False

    else:
        # window size based on screen resolution
        import tkinter as tk
        w, h = tk.Tk().winfo_screenwidth(), tk.Tk().winfo_screenheight()
        n_cols = 2
        if show_obj:
            n_cols = 3
        tiling = (1, n_cols)
        visualisation.display.display_scenes(
            scenes, height=int(h * 0.5), width=int(w * 0.5), tile=tiling
        )

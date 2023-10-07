# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import json
import os
from datetime import datetime

from sdf.modules import trainer


def batch_train(
    device,
    config_file,
    save_path=None,
):
    # init trainer-------------------------------------------------------------
    incremental = True
    sdf_trainer = trainer.Trainer(
        device,
        config_file,
        incremental=incremental,
    )

    # saving init--------------------------------------------------------------
    save = save_path is not None
    print("Save path:", save_path)
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

    # main  loop---------------------------------------------------------------
    print("Starting training for max", sdf_trainer.n_steps, "steps...")
    size_dataset = len(sdf_trainer.scene_dataset)

    break_at = -1

    for t in range(sdf_trainer.n_steps):
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
                    break_at = t + 400
                    print("**************************************",
                          "End of sequence",
                          "**************************************")
                else:
                    print("Total step time", sdf_trainer.tot_step_time)
                    print("frame______________________", new_frame_id)

                    frame_data = sdf_trainer.get_data([new_frame_id])
                    sdf_trainer.add_frame(frame_data)

                    if t == 0:
                        sdf_trainer.last_is_keyframe = True
                        sdf_trainer.optim_frames = 200

        # optimisation step---------------------------------------------
        losses, step_time = sdf_trainer.step()
        status = [k + ': {:.6f}  '.format(losses[k]) for k in losses.keys()]
        status = "".join(status) + '-- Step time: {:.2f}  '.format(step_time)
        loss = losses['total_loss']
        print(t, status)

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
                            "loss": loss.item(),
                        },
                        os.path.join(
                            checkpoint_path, "step_" + save_t + ".pth")
                    )

                if sdf_trainer.save_slices:
                    sdf_trainer.write_slices(
                        slice_path, prefix=save_t + "_",
                        include_gt=False, include_diff=False,
                        include_chomp=False, draw_cams=True)

                if sdf_trainer.save_meshes:
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

    sdf_errors = [vox_res[k]['rays']['vis']['av_l1'] for k in vox_res.keys()]

    return sdf_errors


def run(config_file, save_path):
    assert torch.cuda.is_available()

    sdf_errors = batch_train(
        device="cuda",
        config_file=config_file,
        save_path=save_path,
    )

    return sdf_errors


def load_params(base_config_file):

    with open(base_config_file) as json_file:
        base_config = json.load(json_file)

    seqs = [
        # (dataset_format, seq_name, gt_sdf_dir)

        # ReplicaCAD sequences
        ("replicaCAD", "apt_2_mnp", "apt_2_v1"),
        ("replicaCAD", "apt_2_obj", "apt_2"),
        ("replicaCAD", "apt_2_nav", "apt_2"),
        ("replicaCAD", "apt_3_mnp", "apt_3_v1"),
        ("replicaCAD", "apt_3_obj", "apt_3"),
        ("replicaCAD", "apt_3_nav", "apt_3"),

        # ScanNet longer sequences
        ("ScanNet", "scene0010_00", "scene0010_00"),
        ("ScanNet", "scene0030_00", "scene0030_00"),
        ("ScanNet", "scene0031_00", "scene0031_00"),

        # ScanNet shorter sequences
        ("ScanNet", "scene0004_00", "scene0004_00"),
        ("ScanNet", "scene0005_00", "scene0005_00"),
        ("ScanNet", "scene0009_00", "scene0009_00"),
    ]

    return base_config, seqs


def create_config(
    base_config, format, seq, gt_sdf, data_dir, scannet_root,
):
    config = base_config.copy()

    config['dataset']['format'] = format
    config['dataset']['scene_mesh_file'] = \
        data_dir + "gt_sdfs/" + gt_sdf + "/mesh.obj"
    config['dataset']['gt_sdf_dir'] = \
        data_dir + "gt_sdfs/" + gt_sdf + "/"

    config['eval']['eval_pts_root'] = data_dir + "eval_pts/"
    config['eval']['do_vox_comparison'] = True
    config['eval']['do_eval'] = True

    config['dataset']['seq_dir'] = \
        data_dir + "seqs/" + seq + "/"

    if format == "ScanNet":
        config['dataset']['scannet_dir'] = \
            scannet_root + "/scans/" + seq + "/"

    return config


def create_configs_setallparams(
    exp_settings, base_config_file, data_dir, scannet_root, save_root,
    runs_per_seq=10,
):
    """
        Only one paramter at a time is varied from the default config.
        Creates runs_per_seq x n_seqs configs.
        Creates config files for experiments.
        Input is the setting for the experiments.
        Settings should be in a dict where the keys match the keys
        in the base config file.
    """
    base_config, seqs = load_params(base_config_file)

    now = datetime.now()
    time_str = now.strftime("%m-%d-%y_%H-%M-%S-%f")
    save_root = save_root + time_str + "/"
    os.makedirs(save_root, exist_ok=True)

    config_files = []
    save_paths = []

    for sequence in seqs:

        (dataset_format, seq, gt_sdf) = sequence

        config = create_config(
            base_config, dataset_format, seq, gt_sdf, data_dir, scannet_root)

        # set parameters that are being updated from the default config values
        for category in exp_settings.keys():
            for setting in exp_settings[category].keys():
                config[category][setting] = exp_settings[category][setting]

        # Create multiple experiments per sequence
        for r in range(runs_per_seq):
            save_path = save_root + \
                config['dataset']['seq_dir'].split('/')[-2] + "_" + str(r)
            os.makedirs(save_path, exist_ok=True)

            exp_config = config.copy()

            config_file = save_path + "/config.json"
            with open(config_file, "w") as outfile:
                json.dump(exp_config, outfile, indent=4)

            save_paths.append(save_path)
            config_files.append(config_file)

    return config_files, save_paths


def create_configs_vary1param(
    exp_settings, base_config_file, data_dir, scannet_root, save_root,
    runs_per_seq=10, save_slices=False,
):
    """
        Only one paramter at a time is varied from the default config.
        Creates config files for experiments.
        Input is the setting for the experiments.
        Settings should be in a dict where the keys match the keys in the
        base config file and the values to try are stored in a list.
        Note if there are two parameter lists each with 5 values, then there
        will be 10 experiments in total, not 25. i.e. each parameter is
        updated independently.
    """
    base_config, seqs = load_params(base_config_file)

    now = datetime.now()
    time_str = now.strftime("%m-%d-%y_%H-%M-%S")
    save_root = save_root + time_str + "/"
    os.makedirs(save_root, exist_ok=True)

    config_files = []
    save_paths = []

    for category in exp_settings.keys():
        for setting in exp_settings[category].keys():
            save_path_setting = save_root + "/" + setting + "/"
            os.makedirs(save_path_setting, exist_ok=True)

            for val in exp_settings[category][setting]:
                save_path_val = save_path_setting + str(val) + "/"
                os.makedirs(save_path_val, exist_ok=True)
                for sequence in seqs:

                    (dataset_format, seq, gt_sdf) = sequence

                    config = create_config(
                        base_config, dataset_format, seq, gt_sdf,
                        data_dir, scannet_root
                    )

                    config[category][setting] = val

                    for r in range(runs_per_seq):
                        if r == 0 and save_slices:
                            config['save']['save_slices'] = 1
                        else:
                            config['save']['save_slices'] = 0

                        save_path = save_path_val + \
                            config['dataset']['seq_dir'].split('/')[-2] + \
                            "_" + str(r)
                        os.makedirs(save_path, exist_ok=True)

                        config_file = save_path + "/config.json"
                        with open(config_file, "w") as outfile:
                            json.dump(config, outfile, indent=4)

                        save_paths.append(save_path)
                        config_files.append(config_file)

    return config_files, save_paths


def create_configs_nruns(
    base_config_file, data_dir, scannet_root, save_root,
    runs_per_seq=10, save_slices=False,
):
    """
        Only one paramter at a time is varied from the default config.
        Creates config files for experiments.
        Input is the setting for the experiments.
        Settings should be in a dict where the keys match the keys in the
        base config file and the values to try are stored in a list.
        Note if there are two parameter lists each with 5 values, then there
        will be 10 experiments in total, not 25. i.e. each parameter is
        updated independently.
    """
    base_config, seqs = load_params(base_config_file)

    now = datetime.now()
    time_str = now.strftime("%m-%d-%y_%H-%M-%S")
    save_root = save_root + time_str + "/"
    os.makedirs(save_root, exist_ok=True)

    config_files = []
    save_paths = []

    for sequence in seqs:

        (dataset_format, seq, gt_sdf) = sequence

        config = create_config(
            base_config, dataset_format, seq, gt_sdf, data_dir, scannet_root)

        for r in range(runs_per_seq):
            if r == 0 and save_slices:
                config['save']['save_slices'] = 1
            else:
                config['save']['save_slices'] = 0

            save_path = save_root + \
                config['dataset']['seq_dir'].split('/')[-2] + \
                "_" + str(r)
            os.makedirs(save_path, exist_ok=True)

            config_file = save_path + "/config.json"
            with open(config_file, "w") as outfile:
                json.dump(config, outfile, indent=4)

            save_paths.append(save_path)
            config_files.append(config_file)

    return config_files, save_paths

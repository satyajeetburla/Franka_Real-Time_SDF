# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import queue
import numpy as np
from scipy.spatial.transform import Rotation
import rospy
import trimesh
import cv2
#from orb_slam3_ros_wrapper.msg import frame
from sensor_msgs.msg import Image  # ROS message type
from geometry_msgs.msg import Pose, PoseStamped  # ROS message type
from matplotlib import pyplot as plt


class sdfNode:
    def __init__(self, queue, crop=False) -> None:
        print("sdf Node: starting", os.getpid())
        print("Waiting for first frame...")

        self.queue = queue
        self.crop = crop

        rospy.init_node("sdf", anonymous=True)
        rospy.Subscriber("/camera/color/image_raw", Image, self.rgb_callback, queue_size=1)
        rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.depth_callback, queue_size=1)
        #rospy.Subscriber("/natnet_ros/Realsense_Rigid_Body1/pose", PoseStamped, self.pose_callback, queue_size=1)
        rospy.Subscriber("/natnet_ros/Realsense_camera/pose", PoseStamped, self.pose_callback, queue_size=1)

        rospy.spin()

    def rgb_callback(self, msg):
        rgb_np = np.frombuffer(msg.data, dtype=np.uint8)
        rgb_np = rgb_np.reshape(msg.height, msg.width, 3)
        rgb_np = rgb_np[..., ::-1]

        if self.depth is None or self.pose is None:
            return

        try:
            self.queue.put(
                (rgb_np.copy(), self.depth.copy(), self.pose.copy()),
                block=False,
            )
            #print("In Queue",rgb_np.copy(), self.depth.copy(), self.pose.copy())
        except queue.Full:
            pass

        print("Received RGB image:", rgb_np.shape)
    def depth_callback(self, msg):
        depth_np = np.frombuffer(msg.data, dtype=np.uint16)
        depth_np = depth_np.reshape(msg.height, msg.width)

        if self.crop:
            mw = 40
            mh = 20
            depth_np = depth_np[mh:(msg.height - mh), mw:(msg.width - mw)]

        # self.depth = depth_np.copy()
        # try:
        #     self.queue.put(
        #         (self.rgb_np.copy(), depth_np.copy(), self.pose.copy()),
        #         block=False,
        #     )
        # except queue.Full:
        #     pass
        self.depth = depth_np.copy()
        #print("Received depth image:", depth_np.shape)
        del depth_np

    def pose_callback(self, msg):
        position = msg.pose.position
        quat = msg.pose.orientation
        trans = np.asarray([position.x, position.y, position.z])
        rot = Rotation.from_quat([quat.x, quat.y, quat.z, quat.w]).as_matrix()

        #trans = np.asarray([position.z, position.x, position.y])
        #rot = Rotation.from_quat([quat.z, quat.x, quat.y,  quat.w]).as_matrix()

        camera_transform = np.concatenate((rot, trans.reshape((3, 1))), axis=1)
        camera_transform = np.vstack((camera_transform, [0.0, 0.0, 0.0, 1.0]))

        camera_transform = np.linalg.inv(camera_transform)
        # try:
        #     self.queue.put(
        #         (self.rgb.copy(), self.depth.copy(), self.camera_transform.copy()),
        #         block=False,
        #     )
        # except queue.Full:
        #     pass
        #print("Received camera pose:", camera_transform)

        # rotation_matrix = np.array([[0, 0, 1, 0],
        #                             [1, 0, 0, 0],
        #                             [0, 1, 0, 0],camera_transform
        #                             [0, 0, 0, 1]])
        # rotation_matrix = np.array([[-0.20023241, -0.97929215, 0.02989748, -2.15888469],
        #                             [0.23832766, -0.0782835, -0.9680246, -0.24582459],
        #                             # [0.95031937, -0.18670451, 0.2490673, 3.1134075],
        #                             [0, 0, 0, 1]])

        # Define the given rotation matrix
        # rotation_matrix = np.array([[-0.20023241, -0.97929215, 0.02989748, -2.15888469],
        #                             [0.23832766, -0.0782835, -0.9680246, -0.24582459],
        #                             [0.95031937, -0.18670451, 0.2490673, 3.1134075],
        #                             [0, 0, 0, 1]])

        # Additional rotation by 180 degrees around z-axis
        # additional_rotation_matrix = np.array([[-1, 0, 0, 0],
        #                                        [0, -1, 0, 0],
        #                                        [0, 0, 1, 0],
        #                                        [0, 0, 0, 1]])
        #
        # # Apply the rotation to the camera_transform matrix
        # rotated_camera_transform = np.dot(rotation_matrix, camera_transform)
        # rotated_camera_transform = np.dot(additional_rotation_matrix, rotated_camera_transform)
        #
        # # Update the camera_transform with the rotated version
        # camera_transform = rotated_camera_transform
        # trail -1 with HoJIn
        # transformation_matrix = np.array([[-0.20023241, -0.97929215, 0.02989748, -2.15888469],
        #                                   [0.23832766, -0.0782835, -0.9680246, -0.24582459],
        #                                   [0.95031937, -0.18670451, 0.2490673, 3.1134075],
        #                                   [0, 0, 0, 1]])
        #trail-2 - 15th July
        transformation_matrix = np.array([[-0.78175188, 0.1889632, -0.59427006, -3.01492643],
                                          [0.61911688, 0.34912602, -0.70342399, -0.18474955],
                                          [-0.07455389, 0.91782565, 0.38992023, -0.91009114],
                                          [0, 0, 0, 1]])

        #transformation_matrix = np.linalg.inv(transformation_matrix)
        # rotation_matrix = transformation_matrix[:3, :3]
        # translation_vector = transformation_matrix[:3, 3]
        #
        # # Correct the rotation matrix by transposing it
        # rotation_matrix_corrected = rotation_matrix.T
        #
        # # Compute the corrected translation vector
        # translation_vector_corrected = -rotation_matrix_corrected @ translation_vector
        #
        # # Create the corrected transformation matrix
        # transformation_matrix_corrected = np.eye(4)
        # transformation_matrix_corrected[:3, :3] = rotation_matrix_corrected
        # transformation_matrix_corrected[:3, 3] = translation_vector_corrected
        #camera_transform = np.linalg.inv(camera_transform)
        camera_transform = np.dot(transformation_matrix,camera_transform)
        #camera_transform = np.dot( camera_transform, transformation_matrix)
        self.pose = np.linalg.inv(camera_transform)
        #self.pose = camera_transform
        del camera_transform


class sdfFrankaNode:
    def __init__(self, queue, crop=False, ext_calib=None) -> None:
        print("sdf Franka Node: starting", os.getpid())
        print("Waiting for first frame...")

        self.queue = queue
        self.crop = crop
        self.camera_transform = None

        self.cal = ext_calib

        self.rgb, self.depth, self.pose = None, None, None

        self.first_pose_inv = None

        rospy.init_node("sdf_franka")
        rospy.Subscriber("/camera/color/image_raw", Image, self.rgb_callback, queue_size=1)

        #rospy.Subscriber("/camera/depth/image_rect_raw", Image, self.depth_callback, queue_size=1)
        rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.depth_callback, queue_size=1)
        rospy.Subscriber("/franka_state_controller/ee_pose", Pose, self.pose_callback, queue_size=1)
        rospy.spin()

    def rgb_callback(self, msg):
        rgb_np = np.frombuffer(msg.data, dtype=np.uint8)
        rgb_np = rgb_np.reshape(msg.height, msg.width, 3)
        rgb_np = rgb_np[..., ::-1]

        if self.depth is None or self.pose is None:
            return

        try:
            self.queue.put(
                (rgb_np.copy(), self.depth.copy(), self.pose.copy()),
                block=False,
            )
        except queue.Full:
            pass

    def depth_callback(self, msg):
        depth_np = np.frombuffer(msg.data, dtype=np.uint16)
        depth_np = depth_np.reshape(msg.height, msg.width)

        if self.crop:
            mw = 40
            mh = 20
            depth_np = depth_np[mh:(msg.height - mh), mw:(msg.width - mw)]

        self.depth = depth_np.copy()

    def pose_callback(self, msg):
        position = msg.position
        quat = msg.orientation
        trans = np.asarray([position.x, position.y, position.z])
        rot = Rotation.from_quat([quat.x, quat.y, quat.z, quat.w]).as_matrix()
        camera_transform = np.concatenate((rot, trans.reshape((3, 1))), axis=1)
        camera_transform = np.vstack((camera_transform, [0.0, 0.0, 0.0, 1.0]))
        self.pose = np.linalg.inv(camera_transform)

        del camera_transform


def show_rgbd(rgb, depth, timestamp):
    plt.clf()
    plt.subplot(2, 1, 1)
    plt.imshow(rgb)
    plt.title('RGB ' + str(timestamp))
    plt.subplot(2, 1, 2)
    plt.imshow(depth)
    plt.title('Depth ' + str(timestamp))
    plt.draw()
    plt.pause(1e-6)


def get_latest_frame(q):
    # Empties the queue to get the latest frame
    message = None
    while True:
        try:
            message_latest = q.get(block=False)
            if message is not None:
                del message
            message = message_latest

        except queue.Empty:
            break

    return message

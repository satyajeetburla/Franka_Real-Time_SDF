# Real-Time Signed Distance Function for Franka robot for Downstream Reactive Control Tasks
This project aims to create a real-time system that reconstructs a Signed Distance Field (SDF) by training a neural network model using a continual learning approach for the Franka Emika robot. The model, represented by a Multi-Layer Perceptron (MLP), is trained in real-time using a stream of posed depth images and a self-supervised loss function. Our primary source of inspiration for this project is the paper titled "iSDF: Real-time Neural Signed Distance Fields for Robot Perception." However, adapting GPU-based algorithms for real robotics applications presents significant challenges. This challenge is amplified when dealing with robots like Franka, which operate in a real-time kernel environment where GPU usage is limited or prohibited.

Achieving real-time performance on Franka, which operates at a frequency of 100 Hz, poses a substantial engineering hurdle. This is because our GPU-based algorithm can only achieve a maximum processing speed of 15 Hz. Consequently, it is crucial to meticulously coordinate and optimize the various components of the system to ensure that it meets the real-time requirements of the Franka robot while maintaining the accuracy and effectiveness of the neural SDF reconstruction. Below we provide detailed instructions for running it in your own lab/system.
<table>
  <tr>
    <td>
      <img src="image/image1.jpg" width="200" alt="Image 1">
    </td>
    <td>
      <img src="image/image2.webp" width="200" alt="Image 2">
    </td>
    <td>
      <img src="image/image3.png" width="200" alt="Image 3">
    </td>
  </tr>
</table>

## Major References:
- Real-Time Signed Distance Function Generator: https://github.com/facebookresearch/iSDF
- Franka Interactive Controllers : https://github.com/penn-figueroa-lab/franka_interactive_controllers

## Table of Contents

- [Prerequisites](#prerequisites)
- [Setup and Installation](#setup-and-installation)
- [How to Run](#how-to-run)
- [Results](#results)
  - [Videos](#videos)
- [Acknowledgments](#acknowledgments)

## Prerequisites

Before you begin, ensure you have met the following requirements:
- **Hardware**: Master PC with real-time Linux kernel for Franka Emika robot, Slave PC with GPU for running real-time SDF generator, Franka Emika robot, Realsense Camera, Motion Capture Systems (OptiTrack).
- **Software**: ROS, NVIDIA-CUDA installation, Franka Software Setup, Motion Capture System Software Setup.
## Setup and Installation

1. **Clone the repository**:
git clone [Repository URL]

2. **Setup Environment**:
Both PC
```
conda env create -f environment.yml
conda activate fsdf
```

3. **Install required dependencies in GPU PC**:
```
pip install -r requirements.txt
```
```
pip install -e .
```
## How to Run

We have to run two PC - A master PC running the real-time Linux kernel for the Franka Emika robot & Slave PC with GPU for running real-time SDF generator, both of them communicate with each other using ROS.

**Master PC -- In RT kernel**:

1. Use the "catkin_ws" inside "Franka_SDF_catkin_ws_Franks_CPU" for the Master PC:
``` 
cd ~/catkin_ws
source devel/setup.bash
catkin_make
```
It will install the Franka interactive controller, optitrack setup and other requirements.

2. Install driver for Realsense camera:
```
sudo apt-get install ros-$ROS_DISTRO-realsense2-camera
```
3. Launch the camera:
```
roslaunch realsense2_camera rs_camera.launch color_width:=1280 color_height:=720 color_fps:=30 depth_width:=1280 depth_height:=720 depth_fps:=30 enable_sync:=true align_depth:=true
```
4. Setup RoS communication:
```
export ROS_MASTER_URI=http://"master pc ip":11311
export ROS_IP="master pc IP"
```
5. Main Robot Launch:
To bring up the standalone robot with franka_ros without any specific controller (useful for testing -- can be included in your custom launch file):
```
roslaunch franka_interactive_controllers franka_interactive_bringup.launch
```
6. Luanch the Cartesian Impedance Controller:
```
roslaunch franka_interactive_controllers franka_interactive_bringup.launch controller:=cartesian_pose_impedance
```

7. To showcase the effectiveness of the project, we want to do a simple experiment of moving the Franka to different joint configurations and see how we calculate SDFs in real-time. To move the robot to the different joint configurations using * = 1,2,3 ... 
```
rosrun franka_interactive_controllers libfranka_joint_goal_motion_generator *
```

Please refer to the famous Franka interactive controllers library by Prof. Nadia Figueroa : [franka_interactive_controllers](https://github.com/nbfigueroa/franka_interactive_controllers/tree/main#cartesian-impedance-controller-with-pose-command) for detailed reference.



3. **Slave PC with GPU**:
1. Use the "catkin_ws" inside "Franka_SDF_catkin_ws_GPU" for the Slave PC:
``` 
cd ~/catkin_ws
source devel/setup.bash
catkin_make
```
2. Setup ROS communication:
```
export ROS_MASTER_URI=http://"master pc ip":11311
export ROS_IP="slave pc IP"
```

3. Run a real-time SDF generator inspired by [isdf](https://github.com/facebookresearch/iSDF/tree/main):
```
roslaunch sdf train_franka.launch
```
## Sample Results

### Videos

Sample Demo
[!(https://github.com/satyajeetburla/Franka_Real-Time_SDF/blob/main/%231.mkv)]



## Acknowledgments

- Name or organization
- Another name or organization


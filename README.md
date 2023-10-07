# Real-Time Signed Distance Function for Franka robot for Downstream Reactive Control Tasks
This project aims to create a real-time system that reconstructs a Signed Distance Field (SDF) by training a neural network model using a continual learning approach for the Franka Emika robot. The model, represented by a Multi-Layer Perceptron (MLP), is trained in real-time using a stream of posed depth images and a self-supervised loss function. Our primary source of inspiration for this project is the paper titled "iSDF: Real-time Neural Signed Distance Fields for Robot Perception." However, adapting GPU-based algorithms for real robotics applications presents significant challenges. This challenge is amplified when dealing with robots like Franka, which operate in a real-time kernel environment where GPU usage is limited or prohibited.

Achieving real-time performance on Franka, which operates at a frequency of 100 Hz, poses a substantial engineering hurdle. This is because our GPU-based algorithm can only achieve a maximum processing speed of 15 Hz. Consequently, it is crucial to meticulously coordinate and optimize the various components of the system to ensure that it meets the real-time requirements of the Franka robot while maintaining the accuracy and effectiveness of the neural SDF reconstruction. Below we provide detailed instructions for running it in your own lab/system.

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

2. **Setup Enviornment**:
```
conda env create -f environment.yml
conda activate fsdf
```
```
pip install -e .
```
3. **Install required dependencies**:
pip install -r requirements.txt

## How to Run

1. **Start the Master PC**:
python run_master.py

markdown
Copy code

2. **Start the Slave PC**:
python run_slave.py

bash
Copy code

## Results

### Videos

Embed or link to your result videos here:

<iframe width="560" height="315" src="[YouTube/Vimeo Video Link]" frameborder="0" allowfullscreen></iframe>

### Franka Image

![Franka Robot](path/to/franka/image.jpg)

## License

This project is licensed under the [License Name] - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

- Name or organization
- Another name or organization


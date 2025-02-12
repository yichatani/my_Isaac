# my_Isaac



## Logs 

**[2025/2/13]**

**Progress:** Finish the Grasping System. 

**Next Step:** Recording Data. There are two things I can do now. First, initialize the stage randomly and automatically. Second, write the recording codes to record data. 



**[2025/2/7]**

**Issues:** The Grasping faces physical problems. The grasp penetrates the objects. 

**Solution:** Change the way of controlling the gripper from position control to effort control. 



**[2025/2/3]**

**Issues:** The camera's image is distorted. The camera's calibration intrinsic data is something wrong. 

**Solution:** Change the calibration data send into the grasp detector module differently (Shown in the code).



**[2025/1/25]**

**Issues:** The motion planning method including RRT and Rmpflow does not work very well. 

**Solution:** Deprecate them for now. Use AKsolver directly instead. 



**[2025/1/20]**

**Issues:** The transformation is not always right. 

**Solution:** Tortured. But getting it right at last. 



**[2025/1/5]**

**Issues:** I can't find a module(omni.isaac.motion_planning).

**Solution:** Activate it and add its path to let it be found.



## Project Overview

A universal and modular grasp-related task simulation platform built in Isaac SIm. 



## Repository Structure



```
my_Isaac/
├── scripts/
│   ├── modules/
│   │   ├── control.py            # Control the robot, camera and gripper
│   │   ├── grasp_generator.py    # Generate grasps
│   │   ├── initial_set.py        # Initial settings
│   │   ├── motion_planning.py    # Plan the path
│   │   ├── transform.py          # Coordinate Transform 
│   │   └── record_data.py        # Collect data for training
│   └── ...
├── controller/                   # yaml file
│   ├── ur10e_description.yaml
│   └── rrt_config.yaml
├── urdf/                         # urdf file
│   └── ur10e_gripper.urdf
├── usd/
└── ...
```



## Set up 



### Install Omniverse Launcher:

https://developer.nvidia.com/omniverse#section-getting-started

```sh
sudo apt update
sudo apt install libfuse2
```

```sh
cd Downloads
sudo ./omniverse-launcher-linux.AppImage
```



### Install Isaac Lab

**Follow the tutorial:**

https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html



### (Optional) Set ".bashrc": 

Add codes in `./referencec_settings/add_settings.sh` into .bashrc. 



### Build the environment: 

Each package used in this project is listed. Reference to `./referencec_settings/reference_env.md`.

#### Requirements
- Python
- PyTorch
- Open3d
- NumPy
- SciPy
- Pillow
- MinkowskiEngine

#### Installation
Get the code.
```bash
git clone https://github.com/rhett-chen/graspness_implementation.git
cd graspnet-graspness
```
Install packages via Pip.
```bash
pip install -r requirements.txt
```
Compile and install pointnet2 operators (code adapted from [votenet](https://github.com/facebookresearch/votenet)).
```bash
cd pointnet2
python setup.py install
```
Compile and install knn operator (code adapted from [pytorch_knn_cuda](https://github.com/chrischoy/pytorch_knn_cuda)).
```bash
cd knn
python setup.py install
```
Install graspnetAPI for evaluation.
```bash
git clone https://github.com/graspnet/graspnetAPI.git
cd graspnetAPI
pip install .
```
For MinkowskiEngine, please refer https://github.com/NVIDIA/MinkowskiEngine



### Run:

```sh
cd ./scripts
python main.py
```


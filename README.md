# my_Isaac

## Logs 

**[2025/1/5]**

**Issues:** can't find a module(omni.isaac.motion_planning).

**Solution:** activate it, add its path to let it be found.


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



### Set ".bashrc": add the python path written in 'setup_python_env.sh'

for example:

```sh
## Isaac Python Path
export ISAAC_SIM_PATH=/home/ani/.local/share/ov/pkg/isaac-sim-4.2.0
export PYTHONPATH=$PYTHONPATH:$ISAAC_SIM_PATH/kit/python/lib/python3.10/site-packages:$ISAAC_SIM_PATH/python_packages:$ISAAC_SIM_PATH/exts/omni.isaac.kit:$ISAAC_SIM_PATH/kit/kernel/py:$ISAAC_SIM_PATH/kit/plugins/bindings-python:$ISAAC_SIM_PATH/exts/omni.isaac.lula/pip_prebundle:$ISAAC_SIM_PATH/exts/omni.exporter.urdf/pip_prebundle:$ISAAC_SIM_PATH/extscache/omni.kit.pip_archive-0.0.0+10a4b5c0.lx64.cp310/pip_prebundle:$ISAAC_SIM_PATH/exts/omni.isaac.core_archive/pip_prebundle:$ISAAC_SIM_PATH/exts/omni.isaac.ml_archive/pip_prebundle:$ISAAC_SIM_PATH/exts/omni.pip.compute/pip_prebundle:$ISAAC_SIM_PATH/exts/omni.pip.cloud/pip_prebundle
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ISAAC_SIM_PATH/kit:$ISAAC_SIM_PATH/kit/kernel/plugins:$ISAAC_SIM_PATH/kit/libs/iray:$ISAAC_SIM_PATH/kit/plugins:$ISAAC_SIM_PATH/kit/plugins/bindings-python:$ISAAC_SIM_PATH/kit/plugins/carb_gfx:$ISAAC_SIM_PATH/kit/plugins/rtx:$ISAAC_SIM_PATH/kit/plugins/gpu.foundation:$ISAAC_SIM_PATH/exts/omni.usd.schema.isaac/plugins/IsaacSensorSchema/lib:$ISAAC_SIM_PATH/exts/omni.usd.schema.isaac/plugins/RangeSensorSchema/lib:$ISAAC_SIM_PATH/exts/omni.isaac.lula/pip_prebundle:$ISAAC_SIM_PATH/exts/omni.exporter.urdf/pip_prebundle

# Self Added Isaac python path
export PYTHONPATH=$PYTHONPATH:$ISAAC_SIM_PATH/exts/omni.isaac.motion_generation

# Isaac lab related
export ISAACSIM_DIR="/home/ani/miniconda3/envs/anygrasp/bin/isaacsim"
export ISAACLAB_DIR="/home/ani/IsaacLab"
export EXP_PATH="/home/ani/.local/share/ov/pkg/isaac-sim-4.2.0/apps"
export CARB_APP_PATH="/home/ani/.local/share/ov/pkg/isaac-sim-4.2.0/kit"
export ISAAC_PATH="/home/ani/.local/share/ov/pkg/isaac-sim-4.2.0"
```

<!-- ### Set my_ur_ws

```sh
# install ros2 pkg
sudo apt install ros-humble-ur-robot-driver
sudo apt install ros-humble-moveit

# in the environment
pip install catkin_pkg

# build my_ur_ws
cd my_ur_ws
colcon build
source install/setup.bash

# check if my_ur_driver can be found by rospack
sudo apt install rospack-tools
rospack find my_ur_driver

# if not
export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:~/balance/my_Isaac/my_ur_ws/install/my_ur_driver/share
```

### The directory license/ and log/ need to be changed according to the machine you use.  -->



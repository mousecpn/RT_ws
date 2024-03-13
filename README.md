# Shared autonomy experiment of Robot Trajectron

## Requirements

-   A computer with the Ubuntu 20.04.
-   ROS Noetic
-   [Apriltag ROS](https://github.com/AprilRobotics/apriltag_ros)
-   pybullet
-   qpsolver (daqp)
-   roboticstooblox-python
-   clarabel
-   scs

[How To Install ROS packages](https://industrial-training-master.readthedocs.io/en/melodic/_source/session1/Installing-Existing-Packages.html)

## Installation

#### 1. Clone Repo 

#### 2. Compile the environment

```bash
cd RT_ws
catkin_make
source ./devel/setup.bash
cd ..
```

## Usage
#### Activate trajectron

``` bash
python src/trajectron/src/trajectron_node.py --model_path /path/to/model
```

#### Activate simulation (Recommend to run it in VSCode)

``` bash
python src/franka_share_control/src/franka_sim_world.py 
```

#### Activate shared control

``` bash
python src/shared_control/src/shared_control/SharedControlPybullet.py
```

The "up", "down", "left", "right" represent the moving in x-axis positive/negative direction and y-axis positive/negative direction. "A" and "Z" represent the moving in z-axis positive/negative direction. "H" is homing. 

## Acknowledgement

This repo is based on the code in https://github.com/Shared-control/improved-apf

Thanks for the help from Jordan Antypas@KU Leuven and Maria Papadopoulou@KU Leuven.
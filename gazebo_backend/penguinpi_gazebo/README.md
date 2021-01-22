# Penguin Pi Gazebo Package

This package launches the Penguin Pi robot description to create a gazebo simulation environment for the Penguin Pi. 

## Requirements

### ROS
gazebo (Gazebo 11 preferred)
gazebo_ros

### Python
flask (pip install flask)
gevent (pip install gevent)


## Installation

change to the src directory in your catkin_ws 
```
cd ~/catkin_ws/src
```

Clone the required repositories
```sh

git clone https://bitbucket.org/cirrusrobotics/penguinpi_description.git
git clone https://bitbucket.org/cirrusrobotics/penguinpi_gazebo.git
```

Move to your catkin_ws and build the packages
```sh
cd ~/catkin_ws
catkin_make
```

Source your catkin_ws to make sure ROS can find the penguinpi packages
```
source ~/catkin_ws/devel/setup.bash
```

## Launching 

```sh
roslaunch penguinpi_gazebo penguinpi.launch
```
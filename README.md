## ROS2 Package Dependencies

This project requires the following ROS2 packages:

- `moveit2`
- `moveit_msgs`
- `moveit_resources`
- `Universal_Robots_ROS2_Driver`
- `ur_description`


## WARNING!!

* All users must add `source <YOUR-WORKSPACE>/install/setup.bash` to their environment before launching Isaac Sim (selector)

* This ensures that the necessary environment variables and dependencies are correctly set up for the simulation to run.

## HOW TO RUN

### 1. Launch Moveit2

```bash
ros2 launch ur_moveit_config ur_moveit.launch.py ur_type:=ur5e
```

### 2. Run script

```bash
python3 src/scripts/subscribe_mesh_from_isaac.py
```

> Please check the subscribers' topic names in **subscribe_mesh_from_isaac.py**.
They must match the publisher topic names used in Isaac Sim to ensure correct communication.

## Results

![Isaac Sim Setup Screenshot](src/assets/스크린샷%202025-09-29%2017-23-52.png)

![RVIZ Screenshot](src/assets/스크린샷%202025-09-29%2017-24-05.png)
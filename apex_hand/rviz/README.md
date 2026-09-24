# Apex Hand URDF Viewer

A single-file script used only to visualize this package's URDF. It has no dependencies on the rest of this repository and requires no colcon build.

## Requirements

- **Ubuntu 22.04 + ROS 2 Humble** (the script automatically loads the environment from `/opt/ros/*/setup.bash`, no need to source it beforehand)
- Python 3.9+ (the 3.10 bundled with Humble is fine; no third-party Python libraries needed besides ROS)
- A graphical environment (X11/Wayland desktop; RViz requires a display)

## Installation

**Option 1: Install ROS 2 Desktop (simplest, includes everything needed)**

```bash
# Install ros-humble-desktop following the official docs:
# https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html
sudo apt install ros-humble-desktop
```

**Option 2: Install only the minimal set actually used by this tool**

```bash
sudo apt install \
    ros-humble-robot-state-publisher \   # Publishes TF from URDF + joint angles
    ros-humble-rviz2 \                   # Visualization GUI
    ros-humble-ros2cli \                 # Provides the ros2 command
    ros-humble-rclpy \                   # Used by the joint publisher (subprocess)
    ros-humble-sensor-msgs               # JointState message definitions
```

**Slider feature (optional, only needed for `--slider`)**

```bash
sudo apt install python3-tk
```

## Verify

```bash
ros2 run robot_state_publisher robot_state_publisher --help   # Any output means OK
python3 -c "import rclpy, tkinter"                            # tkinter only needed for --slider
```

## Use

```bash
python3 rviz_viewer.py                          # Left hand, static zero pose
python3 rviz_viewer.py --side right --tactile   # Right hand with tactile points
python3 rviz_viewer.py --slider                 # Add joint slider window
```

Closing the RViz window, closing the slider window, or pressing Ctrl+C exits the program; processes and temporary files are cleaned up automatically.

# Overtaking 🚗💨

This repository contains a ROS 2 package implementing multiple **path planning and overtaking algorithms** for autonomous robots.
It is designed for use with mobile robots or Ackermann-steered vehicles in simulation (e.g., Gazebo, RViz) or real-world experiments.

---

## ✨ Features

* **Best First Search (BestFS):**
  Graph-based search algorithm for planning collision-free paths.

* **Dynamic Window Approach (DWA):**
  Local trajectory optimization based on dynamic feasibility of the robot.

* **Artificial Potential Field (APF):**
  Reactive planner using attractive and repulsive forces.

* **ROS 2 Integration:**

  * Launch files for quick startup
  * Parameterized nodes
  * Compatible with `rclpy`

* **Testing & Style Checks:**
  Includes tests for copyright, flake8, and PEP257.

---

## 📂 Repository Structure

```
overtaking/
├── best_first_search/         # BestFS node implementation
│   └── best_first_search_node.py
├── dynamic_window_approach/   # Dynamic Window Approach implementation
│   └── dwa.py
├── potential_field/           # Potential Field method
│   └── potential_field_node.py
├── launch/                    # Launch files
│   ├── bestFS.launch.py
│   └── dwa.launch.py
├── test/                      # Linting and style tests
├── package.xml                # ROS 2 package manifest
├── setup.py                   # Python setup file
├── setup.cfg                  # Configurations
└── LICENSE
```

---

## 🚀 Installation

Clone the repository into your ROS 2 workspace:

```bash
cd ~/ros2_ws/src
git clone https://github.com/<your-username>/overtaking.git
cd ~/ros2_ws
colcon build
source install/setup.bash
```

Dependencies (make sure you have):

* ROS 2 Humble (or later)
* Python ≥ 3.8
* `rclpy`, `geometry_msgs`, `nav_msgs`, `visualization_msgs`

---

## ▶️ Usage

### Run Best First Search

```bash
ros2 launch overtaking bestFS.launch.py
```

### Run Dynamic Window Approach

```bash
ros2 launch overtaking dwa.launch.py
```

### Run Potential Field

(Currently no dedicated launch file – run node directly)

```bash
ros2 run overtaking potential_field_exe
```

---

## ⚙️ Parameters

Each node supports configurable parameters (tuning for robot dynamics, environment, etc.).
You can pass parameters via the launch file or YAML configs.

---

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

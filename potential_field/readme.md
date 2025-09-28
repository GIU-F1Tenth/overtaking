# Potential Field Local Planner

## Overview

The **Potential Field planner** is a reactive local planner inspired by physics-based models. It treats the robot as a particle in a field:

* **Attractive force** pulls the robot towards the goal.
* **Repulsive forces** push the robot away from obstacles (detected via LiDAR).

The combination of these forces determines the control commands for steering and velocity. This makes the planner intuitive and smooth, but it can suffer from local minima.

---

## Features

* Physics-inspired **force-based navigation**.
* Attractive force towards goal, repulsive forces from obstacles.
* Scales velocity based on force magnitude.
* Subscribes to laser scan, odometry, and goals.
* Publishes Ackermann drive commands and visualization markers for forces.
* Provides a simple way to achieve smooth overtaking or navigation behavior.

---

## ROS 2 Interfaces

### Subscribed Topics

* **`/scan`** (`sensor_msgs/LaserScan`): LiDAR scan for obstacle detection.
* **`/goal_pose`** (`geometry_msgs/PoseStamped`): Global target pose.
* **`/lookahead_point`** (`geometry_msgs/PoseStamped`): Local sub-goal (dynamic).

### Published Topics

* **`/ackermann_cmd`** (`ackermann_msgs/AckermannDriveStamped`): Velocity and steering commands.
* **`/goal_marker`** (`visualization_msgs/Marker`): Visualization of current goal in RViz.
* **`/force_marker`** (`visualization_msgs/MarkerArray`): Visualization of repulsive and attractive forces.

---

## Parameters

* **`k_att`** *(float)*: Attractive force scaling constant.
* **`k_rep`** *(float)*: Repulsive force scaling constant.
* **`repulsion_radius`** *(float)*: Distance within which obstacles exert repulsive force.
* **`max_vel`** *(float)*: Maximum forward velocity.
* **`max_steering_angle`** *(float)*: Steering angle limit.
* **`lookahead_distance`** *(float)*: Distance for sub-goal selection.

---

## Algorithm Details

1. **Attractive Force**

   * Points from robot → goal.
   * Magnitude increases with distance, capped for stability.

2. **Repulsive Force**

   * Computed for each LiDAR point within a threshold radius.
   * Magnitude inversely proportional to obstacle distance.
   * Summed to produce a total repulsion vector.

3. **Resultant Force**

   * The vector sum of attractive and repulsive forces.
   * Direction determines steering angle.
   * Magnitude modulates forward velocity.

4. **Control Output**

   * Publishes an `AckermannDriveStamped` command.
   * Publishes visualization markers to show forces and goals in RViz.

---

## Usage

Launch with:

```bash
ros2 run overtaking potential_field_node
```
---

## Limitations

* Suffers from **local minima** (e.g., robot may get stuck between obstacles).
* Strongly dependent on **parameter tuning** for force scaling.
* Assumes **static obstacle map** (dynamic obstacles may cause oscillations).
* Not a true global planner – works best as a **local reactive layer**.
* Sensitive to noise at high speeds which doesn't make it the best option for racing.
---

## Notes

The potential field method provides a **smooth and natural navigation style**, making it suitable for **overtaking maneuvers** or dynamic avoidance. However, for reliable navigation.

---

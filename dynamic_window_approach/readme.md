# Dynamic Window Approach (DWA) for Ackermann Vehicles

## Overview

The **Dynamic Window Approach (DWA)** is a local planner for reactive collision avoidance and trajectory selection. Unlike grid-based planners, DWA operates directly in the **continuous velocity space** by simulating multiple forward trajectories and choosing the optimal one based on cost functions.

This implementation is optimized for **Ackermann steering vehicles** (e.g., cars) and is heavily vectorized with **NumPy** for high performance.
It is further adapted for **high-speed racing applications** by integrating with an external **optimal path/raceline generator**.

---

## Features

* Fully vectorized **trajectory rollout** for multiple velocities and angular rates.
* Predicts motion over a **time horizon** using Euler integration.
* Evaluates candidate trajectories based on:

  * **Goal heading alignment**
  * **Obstacle avoidance (from LiDAR)**
  * **Velocity preference**
* Requires a **lookahead point** (`x, y, v`) from an **optimal path generator** currently v is stored in orientation.w.
* Chooses final velocity as the **minimum between the simulated velocity** and the **optimal raceline velocity** (ensures safety at high speeds).
* Subscribes to odometry, laser scan, and goals.
* Publishes:

  * Control commands (`AckermannDriveStamped`)
  * Visualization markers for goals and trajectory horizon

---

## ROS 2 Interfaces

### Subscribed Topics

* **`/scan`** (`sensor_msgs/LaserScan`): LiDAR scan for obstacle detection.
* **`/odom`** (`nav_msgs/Odometry`): Current robot odometry.
* **`/goal_pose`** (`geometry_msgs/PoseStamped`): Final navigation goal.
* **`/lookahead_point`** (`geometry_msgs/PoseStamped` with `x, y, v`):
  Local sub-goal from an **optimal raceline generator**, including target position and velocity.

### Published Topics

* **`/ackermann_cmd`** (`ackermann_msgs/AckermannDriveStamped`): Steering and velocity commands.
* **`/goal_marker`** (`visualization_msgs/Marker`): Visualization of current goal.
* **`/horizon_markers`** (`visualization_msgs/MarkerArray`): Visualization of predicted trajectories.

---

## Parameters

* **`prediction_horizon`** *(int)*: Number of simulation steps into the future.
* **`dt`** *(float)*: Time step for trajectory rollout.
* **`n_v_omega`** *(int)*: Number of candidate velocities/omegas sampled.
* **`opt_vel`** *(float)*: Preferred (target) velocity.
* **`max_vel`** *(float)*: Maximum velocity allowed.
* **`max_omega`** *(float)*: Maximum steering angle rate.
* **`accel_limit`** *(float)*: Maximum acceleration constraint.
* **`lookahead_distance`** *(float)*: Distance for selecting local sub-goal.

---

## Algorithm Details

1. **Velocity Sampling**

   * Samples a set of forward velocities (`v`) and angular velocities (`ω`).
   * Respects max velocity and acceleration limits.
   * Final chosen velocity is the **minimum of**:

     * The best simulated velocity from DWA
     * The **raceline-provided optimal velocity**

   This ensures **safe driving at racing speeds** while respecting the global path dynamics.

2. **Trajectory Rollout**

   * Uses **Euler integration** to simulate the vehicle’s motion over a fixed horizon.
   * Each candidate trajectory is represented as a sequence of `(x, y, θ)` states.

3. **Cost Evaluation**

   * Each trajectory is scored using:

     * **Heading cost**: Alignment with goal/target.
     * **Obstacle cost**: Distance to nearest obstacle (from LiDAR).
     * **Velocity cost**: Reward for higher speeds.

4. **Trajectory Selection**

   * Chooses the **trajectory with minimum combined cost**.
   * Extracts `(v, ω)` from that trajectory.
   * Adjusts velocity by taking the **minimum with raceline velocity**.

5. **Command & Visualization**

   * Publishes Ackermann drive command.
   * Publishes RViz markers for goal and horizon visualization.

---

## Usage

Launch the node with:

```bash
ros2 launch overtaking dwa.launch.py
```

---

## Limitations

* Sensitive to parameter tuning (weights, horizon, step size).
* Performance depends on **LiDAR resolution and range**.
* Local minima possible if no clear trajectory is found.
* Assumes **flat 2D environment**.
* Computationally expensive if not tuned correctly.

---

## Notes

This DWA implementation is designed for **real-time performance** thanks to NumPy vectorization.
It is **well-suited for racing applications** because it integrates with an **optimal raceline generator** and adapts its velocity to ensure safe yet aggressive maneuvering.

---
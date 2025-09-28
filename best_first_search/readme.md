# Best-First Search Path Planner

## Overview

The **Best-First Search (BestFS) planner** is a grid-based path planning algorithm designed for overtaking and navigation in structured environments. It discretizes the world into a costmap grid and explores cells based on a heuristic (Manhattan or Euclidean distance) to find an efficient path from the start pose to the goal.

This planner is lightweight and deterministic, making it easy to debug and visualize while providing a foundation for classical search-based planning.

---

## Features

* Grid-based map representation using `nav_msgs/OccupancyGrid`.
* Supports **both Manhattan and Euclidean distance heuristics**.
* Subscribes to global map, goal pose, and lookahead points.
* Publishes planned path as a `nav_msgs/Path`.
* Provides utilities for:

  * World ↔ Grid coordinate conversion.
  * Collision checking (map occupancy).
  * Pose → grid cell mapping.

---

## ROS 2 Interfaces

### Subscribed Topics

* **`/map`** (`nav_msgs/OccupancyGrid`): Global occupancy grid map.
* **`/goal_pose`** (`geometry_msgs/PoseStamped`): Final target pose.
* **`/lookahead_point`** (`geometry_msgs/PoseStamped`): Intermediate goal for dynamic path updates.

### Published Topics

* **`/pp_path`** (`nav_msgs/Path`): Planned path from current robot position to goal.
* **`/astar_lookahead_marker`** (`visualization_msgs/Marker`): Visualization of the current lookahead point on RViz.

---

## Parameters

* **`resolution`** *(float)*: Grid resolution (meters per cell).
* **`heuristic`** *(string)*: Distance metric (`"manhattan"` or `"euclidean"`).
* **`map_topic`** *(string, default `/map`)*: Topic name for the occupancy grid.
* **`goal_topic`** *(string, default `/goal_pose`)*: Topic name for goal pose.

---

## Algorithm Details

1. **Initialization**:

   * Receives the map, goal, and lookahead point.
   * Converts the robot’s world position into grid indices.

2. **Exploration**:

   * Uses a **priority queue (min-heap)** where nodes are sorted by heuristic cost.
   * Expands neighbors in **8 directions**:

     ```
     [(-1,0), (1,0), (0,-1), (0,1), (-1,1), (1,1), (-1,-1), (1,-1)]
     ```
   * Skips nodes already visited or in collision (occupied in costmap).

3. **Path Construction**:

   * Once goal is reached, backtracks through parent dictionary to construct the path.
   * Converts path from grid coordinates → world coordinates.

4. **Output**:

   * Publishes `nav_msgs/Path` for navigation stack.
   * Publishes `visualization_msgs/Marker` for lookahead visualization.

---

## Usage

Launch the node using the provided launch file:

```bash
ros2 launch overtaking bestFS.launch.py
```
---

## Limitations

* Works only with **discretized maps** (OccupancyGrid).
* Computational cost increases with **map size and resolution**.
* Paths are **grid-aligned** and may not be smooth.
* No dynamic re-planning if map updates frequently.

---

## Notes

This planner is best used for **global path generation** in structured maps before local planners (like DWA) smooth and execute the trajectory.

---

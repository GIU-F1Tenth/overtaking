#!/usr/bin/env python3
"""
Best First Search Path Planner Module

This module implements a Best First Search path planning algorithm for autonomous navigation.
It subscribes to occupancy grids and goal markers, then computes optimal paths
using the Best First Search algorithm.
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, Pose, PointStamped
from visualization_msgs.msg import Marker
from rclpy.qos import QoSProfile, DurabilityPolicy
from tf2_ros import Buffer, TransformListener, LookupException
from queue import PriorityQueue
from std_msgs.msg import Header
import numpy as np
from scipy.ndimage import grey_dilation
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_pose


class GraphNode:
    """
    A node in the search graph for A* pathfinding.

    Represents a single cell in the occupancy grid with associated costs
    and heuristic values for the A* algorithm.
    """

    def __init__(self, x, y, heuristic=0, prev=None):
        """
        Initialize a graph node.

        Args:
            x (int): Grid x-coordinate
            y (int): Grid y-coordinate  
            cost (float): Cost to reach this node from start
            heuristic (float): Heuristic estimate to goal
            prev (GraphNode): Previous node in path
        """
        self.x = x
        self.y = y
        self.heuristic = heuristic
        self.prev = prev

    def __lt__(self, other):
        return (self.heuristic) < (other.heuristic)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))

    def __add__(self, other):
        return GraphNode(self.x + other[0], self.y + other[1])


class BestFirstSearch(Node):
    """
    A ROS2 node implementing the Best-First Search path planning algorithm.

    This node subscribes to occupancy grids and goal markers, then computes
    optimal paths using the Best-First Search algorithm. The computed paths are
    published for use by path following controllers.
    """

    def __init__(self):
        """
        Initialize the AStarPlanner node.

        Sets up subscribers, publishers, parameters, and TF listeners.
        """
        super().__init__("bestFS_node")
        self.get_logger().info("Best First Search Node Initialized")
        # Set up TF2 for coordinate transformations
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Configure QoS for map subscription
        map_qos = QoSProfile(depth=10)
        map_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        # Declare parameters
        self.declare_parameter("is_antiClockwise", True)
        self.declare_parameter("costmap_topic", "/costmap/costmap")
        self.declare_parameter("marker_topic", "/astar_lookahead_marker")
        self.declare_parameter("path_topic", "/pp_path")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "laser")
        self.declare_parameter("visited_map_topic", "/visited_map")
        self.declare_parameter("point_topic", "/clicked_point")

        # Get parameter values
        self.is_antiClockwise = self.get_parameter(
            "is_antiClockwise").get_parameter_value().bool_value
        self.costmap_topic = self.get_parameter(
            "costmap_topic").get_parameter_value().string_value
        self.marker_topic = self.get_parameter(
            "marker_topic").get_parameter_value().string_value
        self.path_topic = self.get_parameter(
            "path_topic").get_parameter_value().string_value
        self.map_frame = self.get_parameter(
            "map_frame").get_parameter_value().string_value
        self.base_frame = self.get_parameter(
            "base_frame").get_parameter_value().string_value
        self.visited_map_topic = self.get_parameter(
            "visited_map_topic").get_parameter_value().string_value
        self.point_topic = self.get_parameter(
            "point_topic").get_parameter_value().string_value

        # Set up subscribers and publishers
        self.map_sub = self.create_subscription(
            OccupancyGrid, self.costmap_topic, self.map_callback, map_qos
        )
        self.lookahead_point_sub = self.create_subscription(
            Marker, self.marker_topic, self.lookahead_point_callback, 10
        )
        self.goal_sub = self.create_subscription(
            PointStamped, self.point_topic, self.goal_callback, 10
        )
        self.path_pub = self.create_publisher(Path, self.path_topic, 10)
        self.map_pub = self.create_publisher(OccupancyGrid, self.visited_map_topic, 10)
        
        # Initialize map storage
        self.map_ = OccupancyGrid()
        self.visited_map_ = OccupancyGrid()

    def map_callback(self, map_msg: OccupancyGrid):
        """
        Callback function for receiving occupancy grid maps.

        Args:
            map_msg (OccupancyGrid): The received occupancy grid message
        """
        self.map_ = map_msg
        self.visited_map_.header.frame_id = map_msg.header.frame_id
        self.visited_map_.info = map_msg.info
        self.visited_map_.data = [-1] * (map_msg.info.height * map_msg.info.width)

    def lookahead_point_callback(self, marker: Marker):
        """
        Callback function for processing goal markers.

        Args:
            marker (Marker): The received marker message containing the goal position
        """
        if self.map_ is None:
            self.get_logger().error("No map received!")
            return
        self.visited_map_.data = [-1] * (self.visited_map_.info.height * self.visited_map_.info.width)

        pose = PoseStamped()
        pose.pose.position.x = marker.pose.position.x
        pose.pose.position.y = marker.pose.position.y

        try:
            map_to_base_tf = self.tf_buffer.lookup_transform(
                self.map_.header.frame_id, self.base_frame, rclpy.time.Time()
            )
        except LookupException:
            self.get_logger().error("Could not transform from map to base_link")
            return

        map_to_base_pose = Pose()
        map_to_base_pose.position.x = map_to_base_tf.transform.translation.x
        map_to_base_pose.position.y = map_to_base_tf.transform.translation.y
        map_to_base_pose.orientation = map_to_base_tf.transform.rotation

        # sending goal pose to the planner
        path = self.plan(map_to_base_pose, pose.pose)
        if path.poses:
            self.get_logger().info("Shortest path found!")
            self.path_pub.publish(path)
        else:
            self.get_logger().warn("No path found to the goal.")

    def goal_callback(self, point: PointStamped):
        if self.map_ is None:
            self.get_logger().error("No map received!")
            return

        self.visited_map_.data = [-1] * (self.visited_map_.info.height * self.visited_map_.info.width)

        try:
            map_to_base_tf = self.tf_buffer.lookup_transform(
                self.map_.header.frame_id, self.base_frame, rclpy.time.Time()
            )
        except LookupException:
            self.get_logger().error("Could not transform from map to base_link")
            return

        map_to_base_pose = Pose()
        map_to_base_pose.position.x = map_to_base_tf.transform.translation.x
        map_to_base_pose.position.y = map_to_base_tf.transform.translation.y
        map_to_base_pose.orientation = map_to_base_tf.transform.rotation

        pose = PoseStamped()
        pose.pose.position.x = point.point.x
        pose.pose.position.y = point.point.y

        # sending goal pose to the planner
        path = self.plan(map_to_base_pose, pose.pose)
        if path.poses:
            self.get_logger().info("Shortest path found!")
            self.path_pub.publish(path)
        else:
            self.get_logger().warn("No path found to the goal.")

    def plan(self, start: Pose, goal: Pose):
        # Define possible movement directions
        explore_directions = [(-1, 0), (1, 0), (-1, 1), (1, 1), (0, -1), (0, 1), (1, -1), (-1, -1)] # my addition for the perfect path

        # Priority queue with custom comparison for Best-First Search based on heuristic
        pending_nodes = PriorityQueue()
        visited_nodes = set()

        start_node = self.world_to_grid(start)
        goal_node = self.world_to_grid(goal)
        start_node.heuristic = self.manhattan_distance(start_node, goal_node)
        pending_nodes.put(start_node)

        while not pending_nodes.empty() and rclpy.ok():
            active_node: GraphNode = pending_nodes.get()

            # Goal found!
            if active_node == goal_node:
                break

            # Explore neighbors
            for dir_x, dir_y in explore_directions:
                new_node: GraphNode = active_node + (dir_x, dir_y)

                if (new_node not in visited_nodes and self.pose_on_map(new_node) and
                        0 <= self.map_.data[self.pose_to_cell(new_node)] < 99):

                    new_node.heuristic = self.manhattan_distance(
                        new_node, goal_node) + self.map_.data[self.pose_to_cell(new_node)]
                    new_node.prev = active_node

                    pending_nodes.put(new_node)
                    visited_nodes.add(new_node)

            self.visited_map_.data[self.pose_to_cell(active_node)] = -106
            self.map_pub.publish(self.visited_map_)

        path = Path()
        path.header.frame_id = self.map_.header.frame_id
        while active_node and active_node.prev and rclpy.ok():
            last_pose: Pose = self.grid_to_world(active_node)
            last_pose_stamped = PoseStamped()
            last_pose_stamped.header.frame_id = self.map_.header.frame_id
            last_pose_stamped.pose = last_pose
            path.poses.append(last_pose_stamped)
            active_node = active_node.prev

        if self.is_antiClockwise == False:
            path.poses.reverse()
        return path

    def manhattan_distance(self, node: GraphNode, goal_node: GraphNode):
        return abs(node.x - goal_node.x) + abs(node.y - goal_node.y)

    def pose_on_map(self, node: GraphNode):
        return 0 <= node.x < self.map_.info.width and 0 <= node.y < self.map_.info.height

    def world_to_grid(self, pose: Pose) -> GraphNode:
        grid_x = int(
            (pose.position.x - self.map_.info.origin.position.x) / self.map_.info.resolution)
        grid_y = int(
            (pose.position.y - self.map_.info.origin.position.y) / self.map_.info.resolution)
        return GraphNode(grid_x, grid_y)

    def grid_to_world(self, node: GraphNode) -> Pose:
        pose = Pose()
        pose.position.x = node.x * self.map_.info.resolution + \
            self.map_.info.origin.position.x
        pose.position.y = node.y * self.map_.info.resolution + \
            self.map_.info.origin.position.y
        return pose

    def pose_to_cell(self, node: GraphNode):
        return node.y * self.map_.info.width + node.x


def main(args=None):
    rclpy.init(args=args)
    node = BestFirstSearch()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()

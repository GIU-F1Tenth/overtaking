#!/usr/bin/env python3
"""
Potential Field obstacle-avoidance controller for Ackermann-drive vehicle.
Repulsive forces are computed only within ±60° to reduce vibrations.
Publishes goal and force markers for RViz visualization.
Receives goal from /lookahead_goal (Marker) in map frame and transforms it to vehicle frame.
"""
import math
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Point, Vector3
from visualization_msgs.msg import Marker
from pynput import keyboard

from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
import tf2_geometry_msgs.tf2_geometry_msgs as tf2_gm
from geometry_msgs.msg import PoseStamped, Pose

def euler_from_quaternion(quaternion):
    """
    Convert quaternion to Euler angles.

    Converts quaternion (w in last place) to euler roll, pitch, yaw.
    This should be replaced when porting for ROS 2 Python tf_conversions is done.

    Args:
        quaternion (list): Quaternion as [x, y, z, w]

    Returns:
        tuple: (roll, pitch, yaw) in radians
    """
    x = quaternion[0]
    y = quaternion[1]
    z = quaternion[2]
    w = quaternion[3]

    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


class PotentialFieldNode(Node):
    def __init__(self):
        super().__init__('ackermann_potential_field')
        # --- Parameters ---
        self.declare_parameter('drive_topic','/drive')
        self.declare_parameter('scan_topic','/scan')
        self.declare_parameter('goal_topic','/lookahead_goal')  # Marker goal
        self.declare_parameter('max_speed',4.0)
        self.declare_parameter('min_speed',1.0)
        self.declare_parameter('max_steering_angle',0.4189)
        self.declare_parameter('influence_radius',2.0)
        self.declare_parameter('repulsive_gain',0.14)
        self.declare_parameter('attractive_gain',15.0)
        self.declare_parameter('speed_reduction_scale',1.0)
        self.declare_parameter('scan_timeout',0.5)
        self.declare_parameter('repulsive_angle_limit_deg',80.0)
        self.declare_parameter('force_arrow_length',1.0)
        self.declare_parameter('force_max_magnitude',40.0)
        self.declare_parameter('base_link', 'ego_racecar/base_link')
        self.declare_parameter('map_frame', 'map')

        # --- Parameter values ---
        self.drive_topic = self.get_parameter('drive_topic').get_parameter_value().string_value
        self.scan_topic = self.get_parameter('scan_topic').get_parameter_value().string_value
        self.goal_topic = self.get_parameter('goal_topic').get_parameter_value().string_value
        self.max_speed = self.get_parameter('max_speed').get_parameter_value().double_value
        self.min_speed = self.get_parameter('min_speed').get_parameter_value().double_value
        self.max_steering = self.get_parameter('max_steering_angle').get_parameter_value().double_value
        self.influence_radius = self.get_parameter('influence_radius').get_parameter_value().double_value
        self.repulsive_gain = self.get_parameter('repulsive_gain').get_parameter_value().double_value
        self.attractive_gain = self.get_parameter('attractive_gain').get_parameter_value().double_value
        self.speed_reduction_scale = self.get_parameter('speed_reduction_scale').get_parameter_value().double_value
        self.scan_timeout = self.get_parameter('scan_timeout').get_parameter_value().double_value
        self.angle_limit = math.radians(self.get_parameter('repulsive_angle_limit_deg').get_parameter_value().double_value)
        self.force_arrow_length = self.get_parameter('force_arrow_length').get_parameter_value().double_value
        self.force_max_magnitude = self.get_parameter('force_max_magnitude').get_parameter_value().double_value
        self.base_link_frame = self.get_parameter('base_link').get_parameter_value().string_value
        self.map_frame = self.get_parameter('map_frame').get_parameter_value().string_value 

        # --- Publishers and Subscribers ---
        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.drive_topic, 10)
        self.scan_sub = self.create_subscription(LaserScan, self.scan_topic, self.scan_cb, 10)
        self.goal_sub = self.create_subscription(Marker, self.goal_topic, self.lookahead_goal_cb, 10)
        self.goal_marker_pub = self.create_publisher(Marker, '/potential_field_goal_marker', 10)
        self.force_marker_pub = self.create_publisher(Marker, '/potential_field_force_marker', 10)
        # Subscriber for Nav2D goals
        self.goal_sub = self.create_subscription(PoseStamped, '/goal_pose', self.nav2d_goal_cb, 10)

        # --- TF2 ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # --- State ---
        self.last_scan = None
        self.last_error = 0.0
        self.goal_x = 0.0
        self.goal_y = 0.0
        self.goal_on_map = Pose()
        self.car_pose_on_map = Pose()
        self.last_scan_time = self.get_clock().now()
        self.activate_autonomous_vel = False
        self.kp = 1.0
        self.kd = 0.5

        # --- Control timer ---
        self.control_timer = self.create_timer(0.1, self.control_loop)
        self.pose_timer = self.create_timer(0.01, self.get_pose)

        # --- Keyboard ---
        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()
        self.get_logger().info('Press and hold "a" to drive. Adjust kd/kp with d/x and b/z. ESC to exit.')

    # ------------------ Callbacks ------------------
    def scan_cb(self, msg: LaserScan):
        self.last_scan = msg
        self.last_scan_time = self.get_clock().now()

    def nav2d_goal_cb(self, goal_msg: PoseStamped):
        self.goal_on_map = goal_msg.pose
        self.publish_goal_marker(x=goal_msg.pose.position.x, y=goal_msg.pose.position.y)
            
    def lookahead_goal_cb(self, marker: Marker):
        self.goal_on_map = marker.pose
        self.publish_goal_marker(x=marker.pose.position.x, y=marker.pose.position.y)
        # self.get_logger().info(f"Received new goal at ({marker.pose.position.x:.2f}, {marker.pose.position.y:.2f}) in {marker.header.frame_id} frame.")

        
    def transform_to_vehicle_frame(self, point_on_map: Pose, car_pose: Pose):
        # Convert quaternion to yaw angle
        orientation_list = [car_pose.orientation.x, car_pose.orientation.y, car_pose.orientation.z, car_pose.orientation.w]
        _, _, yaw = euler_from_quaternion(orientation_list)
        dx = point_on_map.position.x - car_pose.position.x
        dy = point_on_map.position.y - car_pose.position.y
        transformed_x = math.cos(-yaw) * dx - math.sin(-yaw) * dy
        transformed_y = math.sin(-yaw) * dx + math.cos(-yaw) * dy
        return transformed_x, transformed_y

    def get_pose(self):
        """
        Main control loop that gets robot pose and executes pure pursuit control.

        This method is called at the configured control frequency. It:
        1. Gets the current robot pose from TF2
        2. Calculates lookahead distance based on current velocity
        3. Finds the appropriate lookahead point on the path
        4. Executes pure pursuit control to track that point
        5. Publishes visualization markers for debugging
        """
        try:
            now = rclpy.time.Time()
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,          # target_frame
                self.base_link_frame,    # source_frame
                now,
                timeout=rclpy.duration.Duration(seconds=0.5)
            )

            trans = transform.transform.translation
            rot = transform.transform.rotation

            self.car_pose_on_map.position.x = trans.x
            self.car_pose_on_map.position.y = trans.y
            self.car_pose_on_map.orientation = rot

        except Exception as e:
            self.get_logger().warn(f"Transform not available: {e}")

    def find_linear_vel_steering_controlled_linearly(self, steering):
        m = (self.max_speed - self.min_speed)/(0.0 - self.angle_limit)
        c = self.max_speed - m*(0.0)
        linear_vel = m*steering + c
        # Clamp to ensure safety
        linear_vel = max(self.min_speed, min(self.max_speed, linear_vel))

        return linear_vel

    # ------------------ Marker Publishing ------------------
    def publish_goal_marker(self, x, y):
        marker = Marker()
        marker.header.frame_id = self.map_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'goal'
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.1
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        self.goal_marker_pub.publish(marker)

    def publish_force_marker(self, force):
        magnitude = np.linalg.norm(force)
        direction = force / magnitude if magnitude > 1e-6 else np.array([1.0, 0.0])
        endpoint = direction * self.force_arrow_length

        m_clamped = min(magnitude, self.force_max_magnitude) / self.force_max_magnitude
        r = m_clamped
        g = 1.0 - m_clamped
        b = 0.0

        marker = Marker()
        marker.header.frame_id = self.base_link_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'force'
        marker.id = 1
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.points = [Point(x=0.0, y=0.0, z=0.0), Point(x=float(endpoint[0]), y=float(endpoint[1]), z=0.0)]
        marker.scale = Vector3(x=0.05, y=0.1, z=0.1)
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.color.a = 1.0
        self.force_marker_pub.publish(marker)

    # ------------------ Keyboard ------------------
    def on_press(self, key):
        try:
            if key.char == 'a':
                self.activate_autonomous_vel = True
            if key.char == 'd':
                self.kd += 0.1
                self.get_logger().info(f"kd: {self.kd:.2f}, kp: {self.kp:.2f}")
            if key.char == 'x':
                self.kd -= 0.1
                self.get_logger().info(f"kd: {self.kd:.2f}, kp: {self.kp:.2f}")
            if key.char == 'b':
                self.kp += 0.1
                self.get_logger().info(f"kd: {self.kd:.2f}, kp: {self.kp:.2f}")
            if key.char == 'z':
                self.kp -= 0.1
                self.get_logger().info(f"kd: {self.kd:.2f}, kp: {self.kp:.2f}")
        except AttributeError:
            pass

    def on_release(self, key):
        self.activate_autonomous_vel = False
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = 0.0
        self.drive_pub.publish(drive_msg)
        if key == keyboard.Key.esc:
            return False

    # ------------------ Forces ------------------
    def compute_repulsive_force(self, scan):
        if scan is None:
            return np.array([0.0,0.0]), float('inf')

        ranges = np.array(scan.ranges)
        valid = np.isfinite(ranges) & (ranges>0.001)
        if not np.any(valid):
            return np.array([0.0,0.0]), float('inf')

        angles = scan.angle_min + np.arange(len(ranges))*scan.angle_increment
        valid = valid & (np.abs(angles) <= self.angle_limit)

        xs = ranges * np.cos(angles)
        ys = ranges * np.sin(angles)
        forces = np.zeros((len(xs),2))
        min_range = float(np.min(ranges[valid])) if np.any(valid) else float('inf')

        for i,r in enumerate(ranges):
            if not valid[i] or r>=self.influence_radius:
                continue
            ux = xs[i]/(r+1e-9)
            uy = ys[i]/(r+1e-9)
            mag = self.repulsive_gain * max(0.0, (1.0/r - 1.0/self.influence_radius)) / ((r+1e-9)**2)
            forces[i] = [-mag*ux, -mag*uy]

        return np.sum(forces,axis=0), min_range

    def compute_attractive_force(self):
        return np.array([self.attractive_gain*self.goal_x, self.attractive_gain*self.goal_y])

    # ------------------ Control Loop ------------------
    def control_loop(self):
        if not self.activate_autonomous_vel:
            return

        now = self.get_clock().now()
        if (now - self.last_scan_time).nanoseconds*1e-9 > self.scan_timeout:
            pass
        
        # goal_x, goal_y are the goal coordinates in the vehicle frame
        self.goal_x, self.goal_y = self.transform_to_vehicle_frame(self.goal_on_map, self.car_pose_on_map)

        rep_force, min_range = self.compute_repulsive_force(self.last_scan)
        att_force = self.compute_attractive_force()
        total_force = att_force + rep_force
        total_force = [np.abs(total_force[0]), total_force[1]]  # No reverse
        self.publish_force_marker(total_force)
        self.publish_goal_marker(self.goal_on_map.position.x, self.goal_on_map.position.y)

        # PD controller for steering
        error = math.atan2(total_force[1], total_force[0])
        desired_heading = self.kp*error + self.kd*(error - self.last_error)
        self.last_error = error
        steering = float(np.clip(desired_heading, -self.max_steering, self.max_steering))

        # proximity_factor = np.clip(min_range / self.influence_radius, 0.0, 1.0)
        # steering_penalty = np.clip(1.0 - (abs(steering) / self.max_steering), 0.0, 1.0)
        # speed = self.max_speed * proximity_factor * (self.speed_reduction_scale * steering_penalty)

        # speed = float(np.clip(speed, self.min_speed, self.max_speed))

        # speed = self.find_linear_vel_steering_controlled_linearly(steering)

        # speed = self.goal_on_map.orientation.w

        speed = 1.0

        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = steering
        drive_msg.drive.speed = speed
        self.drive_pub.publish(drive_msg)

def main(args=None):
    rclpy.init(args=args)
    node = PotentialFieldNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
import math
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Pose, Point
from visualization_msgs.msg import Marker, MarkerArray
from ackermann_msgs.msg import AckermannDriveStamped
from tf_transformations import euler_from_quaternion
from tf2_ros import Buffer, TransformListener
from sensor_msgs.msg import LaserScan
from pynput import keyboard


class DWAParams:
    def __init__(self):
        self.n_omega = 25
        self.prediction_horizon = 10
        self.omega_min = -1.5
        self.omega_max = 1.5
        self.dt = 0.2
        self.goal = [5.0, 5.0] # in map frame
        self.v_max = 1.0
        self.r_buffer = 0.1
        self.obstacles = np.zeros((0, 3))
        self.obstacles_cost = 0.005


class DWAAckermannNode(Node):
    def __init__(self):
        super().__init__("dwa_ackermann_node")

        # Parameters
        self.v = self.declare_parameter("speed", 1.0).value
        self.base_link_frame = self.declare_parameter("base_link", "ego_racecar/base_link").value
        self.map_frame = self.declare_parameter("map_frame", "map").value
        self.scan_topic = self.declare_parameter("scan_topic", "/scan").value
        self.lookahead_sub_topic = self.declare_parameter("lookahead_sub_topic", "lookahead_goal").value
        self.limit_angle = self.declare_parameter("limit_angle", 50).value
        self.laser_distance_from_base_link = self.declare_parameter("laser_distance_from_base_link", 0.275).value
        self.max_lidar_distance = self.declare_parameter('max_lidar_distance', 1.6).value
        self.kp = self.declare_parameter('kp', 1.5).value
        self.kd = self.declare_parameter('kd', 1.5).value

        self.parms = DWAParams()
        self.parms.obstacles_cost = self.declare_parameter("obstacles_cost", 0.005).value

        # State
        self.pose = [0.0, 0.0, 0.0]
        self.car_pose_in_map = Pose()
        self.vel = 0.0

        # Control
        self.last_error = 0.0

        # ROS interfaces
        self.create_subscription(PoseStamped, "/goal_pose", self.goal_cb, 10)
        self.create_subscription(LaserScan, self.scan_topic, self.scan_cb, 10)
        self.create_subscription(Marker, self.lookahead_sub_topic, self.lookahead_goal_cb, 10)

        self.pub_cmd = self.create_publisher(AckermannDriveStamped, "/drive", 10)
        self.goal_marker_pub = self.create_publisher(Marker, "goal_marker", 10)
        self.horizon_pub = self.create_publisher(MarkerArray, "dwa_horizons", 10)

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Timers
        self.create_timer(0.01, self.control_loop)
        self.create_timer(0.01, self.get_pose)

        # Keyboard listener
        keyboard.Listener(on_press=self.on_press, on_release=self.on_release).start()
        self.get_logger().info('Press "a" to drive. ESC to stop.')

    # ----------------- DWA Internal Logic -----------------
    def _euler_integration(self, z0, u):
        """Integrate motion for a single step."""
        v = min(u[0], self.parms.v_max)
        omega = u[1]
        x0, y0, theta0 = z0
        return [
            x0 + v * math.cos(theta0) * self.parms.dt,
            y0 + v * math.sin(theta0) * self.parms.dt,
            theta0 + omega * self.parms.dt,
        ]

    def _simulate_trajectories(self, x0, y0, theta0):
        """Generate candidate trajectories for all sampled angular velocities."""
        omega_all = np.linspace(self.parms.omega_min, self.parms.omega_max, self.parms.n_omega)
        all_trajs = []
        for omega in omega_all:
            z = [[x0, y0, theta0]] # z is the integrated trajectory for each omega at each step
            z0 = [x0, y0, theta0] # z0 is the initial conditions of the integration
            for _ in range(self.parms.prediction_horizon):
                z0 = self._euler_integration(z0, [self.v, omega])
                z.append(z0)
            all_trajs.append(np.array(z))
        return omega_all, all_trajs

    def _compute_cost(self, traj):
        """Compute combined cost to goal and obstacles for a single trajectory."""
        cost = sum(math.dist((x, y), self.parms.goal) for x, y, _ in traj)
        if self.parms.obstacles.shape[0] > 0:
            for x, y, _ in traj:
                for x_obs, y_obs, r_obs in self.parms.obstacles:
                    dist = max(math.dist((x_obs, y_obs), (x, y)) - (r_obs + self.parms.r_buffer), 0.001)
                    cost += self.parms.obstacles_cost / dist
        return cost

    def _run_dwa(self):
        """Run the full DWA calculation and return chosen omega and trajectory index."""
        x0, y0, theta0 = (0, 0, 0) # since we are computing everything in the vehicle's frame, so our position is (0, 0, 0)
        omega_all, all_trajs = self._simulate_trajectories(x0, y0, theta0)
        costs = [self._compute_cost(traj) for traj in all_trajs]
        chosen_idx = int(np.argmin(costs))
        return omega_all[chosen_idx], chosen_idx, all_trajs

    # ----------------- ROS Callbacks -----------------
    def goal_cb(self, msg: PoseStamped):
        goal_on_map = [msg.pose.position.x, msg.pose.position.y]
        self.publish_goal_marker(*goal_on_map)
        goal_x_in_car, goal_y_in_car = self.transform_to_vehicle_frame(msg.pose, self.car_pose_in_map)
        self.parms.goal = [goal_x_in_car, goal_y_in_car] # goal in vehicle's frame

    def lookahead_goal_cb(self, msg: PoseStamped):
        goal_x_in_car, goal_y_in_car = self.transform_to_vehicle_frame(msg.pose, self.car_pose_in_map)
        self.parms.goal = [goal_x_in_car, goal_y_in_car] # goal in vehicle's frame

    def scan_cb(self, msg: LaserScan):
        angles = np.arange(msg.angle_min, msg.angle_max, msg.angle_increment)
        ranges = np.array(msg.ranges)
        valid = np.isfinite(ranges)
        angles, ranges = angles[valid], ranges[valid]
        mask = (angles >= -math.radians(self.limit_angle)) & (angles <= math.radians(self.limit_angle))
        angles, ranges = angles[mask], ranges[mask]

        points_map = []
        for r, a in zip(ranges, angles):
            if r <= self.max_lidar_distance:
                lx, ly = r * math.cos(a), r * math.sin(a)
                p = Pose()
                p.position.x, p.position.y = lx, ly
                points_map.append([lx, ly, 0.1]) # obstacles in the vehicle's frame
        self.parms.obstacles = np.array(points_map)

    def get_pose(self):
        try:
            now = rclpy.time.Time()
            transform = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_link_frame, now, timeout=rclpy.duration.Duration(seconds=0.5)
            )
            yaw = euler_from_quaternion([
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w,
            ])[2]
            # compensting the base_link -> laser transform
            tx = transform.transform.translation.x + self.laser_distance_from_base_link * math.cos(yaw)
            ty = transform.transform.translation.y + self.laser_distance_from_base_link * math.sin(yaw)
            self.car_pose_in_map.position.x = tx
            self.car_pose_in_map.position.y = ty
            self.car_pose_in_map.orientation = transform.transform.rotation
            self.pose = [tx, ty, yaw]
        except Exception as e:
            self.get_logger().warn(f"Transform not available: {e}")

    def control_loop(self):
        chosen_omega, chosen_idx, all_trajs = self._run_dwa()
        self.publish_horizon_markers(all_trajs, chosen_idx)

        cmd = AckermannDriveStamped()
        cmd.drive.speed = self.vel
        # add a PD controller
        error = math.atan2(chosen_omega * 0.33, self.vel) - 0 if self.vel != 0 else 0.0 
        p_controller = self.kp * error
        d_controller = self.kd * (error - self.last_error)
        cmd.drive.steering_angle = p_controller + d_controller
        self.pub_cmd.publish(cmd)

        # self.get_logger().info(
        #     f"DWA -> omega: {chosen_omega:.3f}, goal: {self.parms.goal}, obstacles: {len(self.parms.obstacles)}"
        # )

    # ----------------- Visualization -----------------
    def publish_goal_marker(self, x, y):
        m = Marker()
        m.header.frame_id = self.map_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.type, m.action = Marker.SPHERE, Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = x, y, 0.1
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.3
        m.color.a, m.color.r = 1.0, 1.0
        self.goal_marker_pub.publish(m)

    def publish_horizon_markers(self, all_trajs, chosen_idx):
        marker_array = MarkerArray()
        for idx, traj in enumerate(all_trajs):
            m = Marker()
            m.header.frame_id = self.base_link_frame
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns, m.id = "dwa_horizons", idx
            m.type, m.action = Marker.LINE_STRIP, Marker.ADD
            m.scale.x = 0.02
            if idx == chosen_idx:
                m.color.r = 1.0
            else:
                m.color.g, m.color.b = 0.5, 1.0
            m.color.a = 0.8
            for x, y, _ in traj:
                m.points.append(Point(x=x, y=y, z=0.05))
            marker_array.markers.append(m)
        self.horizon_pub.publish(marker_array)

    # ----------------- Utilities -----------------
    def transform_to_map_frame(self, point, car_pose):
        yaw = euler_from_quaternion([
            car_pose.orientation.x,
            car_pose.orientation.y,
            car_pose.orientation.z,
            car_pose.orientation.w,
        ])[2]
        tx = math.cos(yaw) * point.position.x - math.sin(yaw) * point.position.y + car_pose.position.x
        ty = math.sin(yaw) * point.position.x + math.cos(yaw) * point.position.y + car_pose.position.y
        p = Pose()
        p.position.x, p.position.y = tx, ty
        return p

    def transform_to_vehicle_frame(self, point_on_map: Pose, car_pose: Pose):
        # Convert quaternion to yaw angle
        orientation_list = [car_pose.orientation.x, car_pose.orientation.y, car_pose.orientation.z, car_pose.orientation.w]
        _, _, yaw = euler_from_quaternion(orientation_list)
        dx = point_on_map.position.x - car_pose.position.x
        dy = point_on_map.position.y - car_pose.position.y
        transformed_x = math.cos(-yaw) * dx - math.sin(-yaw) * dy
        transformed_y = math.sin(-yaw) * dx + math.cos(-yaw) * dy
        return transformed_x, transformed_y
    
    # ----------------- Keyboard -----------------
    def on_press(self, key):
        if hasattr(key, "char") and key.char == "a":
            self.vel = 3.5

    def on_release(self, key):
        self.vel = 0.0
        self.pub_cmd.publish(AckermannDriveStamped())
        if key == keyboard.Key.esc:
            return False


def main(args=None):
    rclpy.init(args=args)
    node = DWAAckermannNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import math
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Pose, Point
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
from ackermann_msgs.msg import AckermannDriveStamped
from tf_transformations import euler_from_quaternion
from tf2_ros import Buffer, TransformListener
from sensor_msgs.msg import LaserScan
import time
from nav_msgs.msg import Odometry


class DWAAckermannNode(Node):
    def __init__(self):
        super().__init__("dwa_ackermann_node")

        # Frame parameters
        self.base_link_frame = self.declare_parameter(
            "base_link", "ego_racecar/base_link").value
        self.map_frame = self.declare_parameter("map_frame", "map").value

        # Topic parameters
        self.scan_topic = self.declare_parameter("scan_topic", "/scan").value
        self.odom_topic = self.declare_parameter(
            'odom_topic', '/ego_racecar/odom').value
        self.lookahead_sub_topic = self.declare_parameter(
            "lookahead_sub_topic", "lookahead_goal").value
        self.goal_topic = self.declare_parameter(
            "goal_topic", "/goal_pose").value
        self.drive_topic = self.declare_parameter(
            "drive_topic", "/drive").value
        self.goal_marker_topic = self.declare_parameter(
            "goal_marker_topic", "goal_marker").value
        self.horizon_marker_topic = self.declare_parameter(
            "horizon_marker_topic", "dwa_horizons").value
        self.start_topic = self.declare_parameter(
            "start_topic", "/control_selector").value

        # General parameters
        self.laser_distance_from_base_link = self.declare_parameter(
            "laser_distance_from_base_link", 0.275).value
        self.wheel_base_length = self.declare_parameter(
            "wheel_base_length", 0.275).value
        self.using_colored_horizons = self.declare_parameter(
            "using_colored_horizons", True).value

        # DWA specific parameters
        self.n_v_omega_max = self.declare_parameter(
            "dwa.n_v_omega_max", 25).value
        self.n_v_omega_min = self.declare_parameter(
            "dwa.n_v_omega_min", 15).value
        self.prediction_horizon = self.declare_parameter(
            "dwa.prediction_horizon", 10).value
        self.omega_max_full = self.declare_parameter(
            "dwa.omega_max_full", 2.0).value
        self.omega_max_min = self.declare_parameter(
            "dwa.omega_max_min", 1.0).value
        self.v_min = self.declare_parameter("dwa.v_min", 0.5).value
        self.v_max = self.declare_parameter("dwa.v_max", 7.5).value
        self.integ_vel_scale = self.declare_parameter(
            "dwa.integ_vel_scale", 1.0).value
        self.integ_vel_scale_offset = self.declare_parameter(
            "dwa.integ_vel_scale_offset", 0.0).value
        self.dt = self.declare_parameter("dwa.dt", 0.2).value
        self.r_buffer = self.declare_parameter("dwa.r_buffer", 0.1).value
        self.obstacles_cost = self.declare_parameter(
            "dwa.obstacles_cost", 0.005).value
        self.goal_cost = self.declare_parameter("dwa.goal_cost", 1.0).value
        self.max_vel_cost = self.declare_parameter(
            "dwa.max_vel_cost", 3.0).value
        self.min_vel_cost = self.declare_parameter(
            "dwa.min_vel_cost", 0.0).value
        self.high_speed_sharp_traj_cost = self.declare_parameter(
            "dwa.high_speed_sharp_traj_cost", 2.0).value
        self.limit_angle = self.declare_parameter("dwa.limit_angle", 90).value
        self.max_lidar_distance = self.declare_parameter(
            'dwa.max_lidar_distance', 4.5).value
        self.min_lidar_distance = self.declare_parameter(
            'dwa.min_lidar_distance', 1.0).value
        self.spread_gaussian = self.declare_parameter(
            'dwa.spread_gaussian', 2.5).value
        self.kp = self.declare_parameter('dwa.kp', 2.2).value
        self.kd = self.declare_parameter('dwa.kd', 1.5).value
        # use NumPy array for vectorized ops; empty shape (0,3) means no obstacles
        self.obstacles = np.zeros((0, 3), dtype=float)

        # State
        self.pose = [0.0, 0.0, 0.0]
        self.car_pose_in_map = Pose()
        self.vel = False

        # Control
        self.last_error = 0.0
        self.opt_vel = 0.0
        self.odom = Odometry()
        self.lidar_cap = 0.0
        # in map frame (vehicle frame used by algorithm)
        self.goal = [0.0, 0.0]

        self.n_v_omega = 0
        self._all_trajs_buffer = np.zeros(
            (self.n_v_omega_max, self.prediction_horizon + 1, 3), dtype=float
        )

        # ROS interfaces
        self.create_subscription(
            PoseStamped, self.goal_topic, self.goal_cb, 10)
        self.create_subscription(LaserScan, self.scan_topic, self.scan_cb, 10)
        self.create_subscription(
            Marker, self.lookahead_sub_topic, self.lookahead_goal_cb, 10)
        self.create_subscription(Odometry, self.odom_topic, self.odom_cb, 10)
        self.create_subscription(String, self.start_topic, self.start_cb, 10)

        self.pub_cmd = self.create_publisher(
            AckermannDriveStamped, self.drive_topic, 10)
        self.goal_marker_pub = self.create_publisher(
            Marker, self.goal_marker_topic, 10)
        self.horizon_pub = self.create_publisher(
            MarkerArray, self.horizon_marker_topic, 10)

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Timers
        self.create_timer(0.01, self.control_loop)
        self.create_timer(0.01, self.get_pose)

        self.scan_time = 0.0
        self.dwa_time = 0.0

        self.get_logger().info('Press "a" to drive. ESC to stop.')

        # Log loaded parameters
        # self.get_logger().info(f'DWA Parameters loaded: v_range=[{self.v_min}, {self.v_max}], omega_range=[{self.omega_min}, {self.omega_max}]')
        self.max_dwa_time = 0.0
        self.max_scan_time = 0.0

    # ----------------- DWA Internal Logic -----------------
    def _euler_integration_step(self, prev, v, omega):
        """Vectorized single integration step for arrays prev shape (n,3), v either scalar or (n,), omega (n,)"""
        x = prev[:, 0] + v * np.cos(prev[:, 2]) * self.dt
        y = prev[:, 1] + v * np.sin(prev[:, 2]) * self.dt
        theta = prev[:, 2] + omega * self.dt
        return np.stack((x, y, theta), axis=1)

    def compute_linear_vel(self):
        # Create Gaussian-shaped velocity distribution
        idx = np.arange(self.n_v_omega)
        center = (self.n_v_omega - 1) / 2.0
        sigma = self.n_v_omega / self.spread_gaussian  # spread of Gaussian
        gaussian_weights = np.exp(-0.5 * ((idx - center) / sigma) ** 2)

        # Normalize to [v_min, v_max]
        gaussian_weights = (gaussian_weights - gaussian_weights.min()) / \
            (gaussian_weights.max() - gaussian_weights.min())
        v_all = self.v_min + gaussian_weights * (self.v_max - self.v_min)
        return v_all

    def compute_dynamic_omega_window(self):
        """
        Nonlinear 'Quadratic' shrinking of omega window based on velocity.

        At v_min  -> omega = [-omega_max, omega_max]
        At v_max  -> omega = [-omega_min, omega_min]
        """
        v = abs(self.odom.twist.twist.linear.x)

        # Normalize velocity between 0 and 1
        alpha = (v - self.v_min) / (self.v_max - self.v_min)
        alpha = np.clip(alpha, 0.0, 1.0)

        # Quadratic nonlinear shrink
        shrink_factor = 1.0 - (1.0 - self.omega_max_min /
                               self.omega_max_full) * (alpha ** 2)

        omega_limit = self.omega_max_full * shrink_factor

        return -omega_limit, omega_limit

    def _compute_velocity_dependent_n_samples(self):
        """
        Nonlinear reduction of n_v_omega based on velocity.

        At v_min  -> n_v_omega
        At v_max  -> n_v_omega_min
        """

        v = abs(self.odom.twist.twist.linear.x)
        alpha = (v - self.v_min) / (self.v_max - self.v_min)
        alpha = np.clip(alpha, 0.0, 1.0)

        n_max = self.n_v_omega_max
        n_min = self.n_v_omega_min

        n_dynamic = n_max - (n_max - n_min) * (alpha ** 2)

        return int(max(n_min, round(n_dynamic)))

    def _run_dwa(self):
        """Fully vectorized DWA over all omegas and trajectory horizon.
        Returns: chosen_v, chosen_omega, chosen_idx, closest_traj_idx,
        all_trajs (np.ndarray shape (n_omega, horizon+1, 3))"""
        start = time.perf_counter()

        self.n_v_omega = self._compute_velocity_dependent_n_samples()
        horizon = self.prediction_horizon

        # prepare grids
        omega_min, omega_max = self.compute_dynamic_omega_window()
        omega_all = np.linspace(omega_min, omega_max, self.n_v_omega)
        v_all = self.compute_linear_vel()

        # Reuse a preallocated workspace to avoid per-cycle trajectory allocations.
        all_trajs = self._all_trajs_buffer[:self.n_v_omega, : horizon + 1, :]
        all_trajs.fill(0.0)

        # Integrate forward for all trajectories in a vectorized loop over time steps
        for t in range(1, horizon + 1):
            prev = all_trajs[:, t - 1, :]
            # integ_vel_scale is factor for integration computation
            integ_ = self.integ_vel_scale_offset + \
                self.odom.twist.twist.linear.x / self.integ_vel_scale
            all_trajs[:, t, :] = self._euler_integration_step(
                prev, integ_, omega_all)

        # Vectorized goal cost
        goal = np.array(self.goal)
        traj_xy = all_trajs[:, :, :2]  # (n_omega, horizon+1, 2)
        dists_to_goal = np.linalg.norm(
            traj_xy - goal[None, None, :], axis=2)  # (n_omega, horizon+1)
        min_goal_dists = np.min(dists_to_goal, axis=1)  # (n_omega,)
        # closer to goal = lower cost
        goal_costs = self.goal_cost * np.sum(dists_to_goal, axis=1)
        closest_traj_idx = int(np.argmin(min_goal_dists))
        vel_scale = 1.0 - \
            np.clip(self.odom.twist.twist.linear.x / self.v_max, 0.0, 1.0)
        # bonus for trajectory that gets closest to the goal, scaled by current velocity (encourages following the goal at low speeds, more freedom at high speeds)
        goal_costs[closest_traj_idx] -= self.goal_cost * vel_scale

        # Vectorized obstacle cost
        if self.obstacles.shape[0] > 0:
            obs_xy = self.obstacles[:, :2]  # (M,2)
            obs_r = self.obstacles[:, 2]    # (M,)

            # Broadcast differences: (n_omega, horizon+1, M, 2)
            diff = traj_xy[:, :, None, :] - obs_xy[None, None, :, :]
            dists = np.linalg.norm(diff, axis=3) - obs_r[None, None, :]
            dists = np.maximum(dists, 0.01)
            inv_dists = self.obstacles_cost / dists
            # sum over time and obstacles -> (n_omega,)
            obs_costs = np.sum(inv_dists, axis=(1, 2))
        else:
            obs_costs = np.zeros_like(goal_costs)

        # Velocity costs (vectorized)
        velocity_costs = (1 - (v_all - self.v_min) / (self.v_max - self.v_min)) * (
            self.max_vel_cost - self.min_vel_cost) + self.min_vel_cost

        # scale by velocity to penalize sharper turns at higher speeds
        velocity_costs *= self.odom.twist.twist.linear.x * self.high_speed_sharp_traj_cost

        total_costs = goal_costs + obs_costs + velocity_costs
        chosen_idx = int(np.argmin(total_costs))

        self.dwa_time = (time.perf_counter() - start) * 1000
        return (
            float(min(v_all[chosen_idx], self.opt_vel)),
            float(omega_all[chosen_idx]),
            chosen_idx,
            closest_traj_idx,
            all_trajs,
            total_costs
        )

    # ----------------- ROS Callbacks -----------------
    def goal_cb(self, msg: PoseStamped):
        goal_on_map = [msg.pose.position.x, msg.pose.position.y]
        self.publish_goal_marker(*goal_on_map)
        goal_x_in_car, goal_y_in_car = self.transform_to_vehicle_frame(
            msg.pose, self.car_pose_in_map)
        self.goal = [goal_x_in_car, goal_y_in_car]  # goal in vehicle's frame

    def lookahead_goal_cb(self, msg: Marker):
        goal_x_in_car, goal_y_in_car = self.transform_to_vehicle_frame(
            msg.pose, self.car_pose_in_map)
        self.opt_vel = msg.pose.orientation.w
        self.goal = [goal_x_in_car, goal_y_in_car]  # goal in vehicle's frame

    def scan_cb(self, msg: LaserScan):
        start = time.perf_counter()
        # Build angles using number of points (safer than arange to avoid floating rounding mismatches)
        n_pts = len(msg.ranges)
        angles = msg.angle_min + np.arange(n_pts) * msg.angle_increment
        ranges = np.array(msg.ranges, dtype=float)
        valid = np.isfinite(ranges)
        angles = angles[valid]
        ranges = ranges[valid]

        mask = (angles >= -np.deg2rad(self.limit_angle)
                ) & (angles <= np.deg2rad(self.limit_angle))
        angles = angles[mask]
        ranges = ranges[mask]

        lidar_capping_distance = self.compute_lidar_max_dist()
        self.lidar_cap = lidar_capping_distance
        distance_mask = ranges <= lidar_capping_distance
        if np.any(distance_mask):
            capped_ranges = ranges[distance_mask]
            capped_angles = angles[distance_mask]
            obs_x = capped_ranges * np.cos(capped_angles)
            obs_y = capped_ranges * np.sin(capped_angles)
            obs_r = np.full(obs_x.shape, self.r_buffer, dtype=float)
            self.obstacles = np.column_stack((obs_x, obs_y, obs_r))
        else:
            self.obstacles = np.zeros((0, 3), dtype=float)

        self.scan_time = (time.perf_counter() - start) * 1000

    def odom_cb(self, msg: Odometry):
        self.odom = msg

    def compute_lidar_max_dist(self):
        linear_vel = self.odom.twist.twist.linear.x
        return np.clip(((linear_vel/self.v_max) * (self.max_lidar_distance - self.min_lidar_distance)) + self.min_lidar_distance,
                       self.min_lidar_distance, self.max_lidar_distance)

    def get_pose(self):
        if not self.vel:
            return
        try:
            now = rclpy.time.Time()
            transform = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_link_frame, now, timeout=rclpy.duration.Duration(
                    seconds=0.5)
            )
            yaw = euler_from_quaternion([
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w,
            ])[2]
            # compensating the base_link -> laser transform
            tx = transform.transform.translation.x + \
                self.laser_distance_from_base_link * math.cos(yaw)
            ty = transform.transform.translation.y + \
                self.laser_distance_from_base_link * math.sin(yaw)
            self.car_pose_in_map.position.x = tx
            self.car_pose_in_map.position.y = ty
            self.car_pose_in_map.orientation = transform.transform.rotation
            self.pose = [tx, ty, yaw]
        except Exception as e:
            self.get_logger().warn(f"Transform not available: {e}")

    def control_loop(self):
        if not self.vel:
            return
        chosen_v, chosen_omega, chosen_idx, closest_traj_idx, all_trajs, total_costs = self._run_dwa()

        if self.using_colored_horizons:
            self.publish_horizon_markers_colored(
                all_trajs, chosen_idx, closest_traj_idx, total_costs)
        else:
            self.publish_horizon_markers(
                all_trajs, chosen_idx, closest_traj_idx)

        cmd = AckermannDriveStamped()
        if self.vel:
            # choose min between DWA velocity and lookahead optimal
            cmd.drive.speed = min(chosen_v, self.opt_vel)
        else:
            cmd.drive.speed = 0.0

        # add a PD controller
        error = math.atan2(chosen_omega * self.wheel_base_length,
                           chosen_v) if chosen_v != 0 else 0.0
        p_controller = self.kp * error
        d_controller = self.kd * (error - self.last_error)
        cmd.drive.steering_angle = p_controller + d_controller
        self.last_error = error

        self.pub_cmd.publish(cmd)

        self.max_dwa_time = max(self.dwa_time, self.max_dwa_time)
        self.max_scan_time = max(self.max_scan_time, self.scan_time)
        # log timings every N cycles conservatively
        self.get_logger().info(
            f"DWA time: {self.dwa_time:.3f} ms, scan time: {self.scan_time:.3f} ms, chosen vel: {chosen_v:.2f}, lidar_cap: {self.lidar_cap:.2f} (max DWA: {self.max_dwa_time:.3f} ms, max scan: {self.max_scan_time:.3f} ms)", throttle_duration_sec=1.0)

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

    def publish_horizon_markers(self, all_trajs, chosen_idx, closest_traj_idx):
        marker_array = MarkerArray()
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        # all_trajs may be numpy array shape (n_omega, horizon+1, 3)
        n_omega = all_trajs.shape[0]
        for idx in range(n_omega):
            traj = all_trajs[idx]
            m = Marker()
            m.header.frame_id = self.base_link_frame
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns, m.id = "dwa_horizons", int(idx)
            m.type, m.action = Marker.LINE_STRIP, Marker.ADD
            m.scale.x = 0.05 if idx == chosen_idx else 0.02
            if idx == closest_traj_idx:
                m.color.r = 0.6
                m.color.g = 0.0
                m.color.b = 0.8
            elif idx == chosen_idx:
                m.color.r = 1.0
            else:
                m.color.g, m.color.b = 0.5, 1.0
            m.color.a = 1.0 if idx in (chosen_idx, closest_traj_idx) else 0.8
            for x, y, _ in traj:
                m.points.append(Point(x=float(x), y=float(y), z=0.05))
            marker_array.markers.append(m)
        self.horizon_pub.publish(marker_array)

    def publish_horizon_markers_colored(self, all_trajs, chosen_idx, closest_traj_idx, total_costs):
        marker_array = MarkerArray()

        # Clear previous markers first, important because n_omega changes over time
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        n_omega = all_trajs.shape[0]

        # Normalize costs to [0, 1]
        c_min = float(np.min(total_costs))
        c_max = float(np.max(total_costs))
        if c_max - c_min < 1e-9:
            normalized_costs = np.zeros_like(total_costs, dtype=float)
        else:
            normalized_costs = (total_costs - c_min) / (c_max - c_min)

        for idx in range(n_omega):
            traj = all_trajs[idx]
            m = Marker()
            m.header.frame_id = self.base_link_frame
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = "dwa_horizons"
            m.id = int(idx)
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD

            # Slightly thicker chosen trajectory
            m.scale.x = 0.05 if idx == chosen_idx else 0.02

            if idx == closest_traj_idx:
                m.color.r = 0.6
                m.color.g = 0.0
                m.color.b = 0.8
                m.color.a = 1.0
            else:
                # Cost-based coloring:
                # low cost  -> green
                # high cost -> red
                norm = float(normalized_costs[idx])
                m.color.r = norm
                m.color.g = 1.0 - norm
                m.color.b = 0.0
                m.color.a = 1.0 if idx == chosen_idx else 0.8

            # Optional: make chosen one blue instead of cost-colored
            # if idx == chosen_idx:
            #     m.color.r = 0.0
            #     m.color.g = 0.4
            #     m.color.b = 1.0
            #     m.color.a = 1.0

            for x, y, _ in traj:
                m.points.append(Point(x=float(x), y=float(y), z=0.05))

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
        tx = math.cos(yaw) * point.position.x - math.sin(yaw) * \
            point.position.y + car_pose.position.x
        ty = math.sin(yaw) * point.position.x + math.cos(yaw) * \
            point.position.y + car_pose.position.y
        p = Pose()
        p.position.x, p.position.y = tx, ty
        return p

    def transform_to_vehicle_frame(self, point_on_map: Pose, car_pose: Pose):
        # Convert quaternion to yaw angle
        orientation_list = [car_pose.orientation.x, car_pose.orientation.y,
                            car_pose.orientation.z, car_pose.orientation.w]
        _, _, yaw = euler_from_quaternion(orientation_list)
        dx = point_on_map.position.x - car_pose.position.x
        dy = point_on_map.position.y - car_pose.position.y
        transformed_x = math.cos(-yaw) * dx - math.sin(-yaw) * dy
        transformed_y = math.sin(-yaw) * dx + math.cos(-yaw) * dy
        return transformed_x, transformed_y

    def start_cb(self, msg: String):
        if msg.data == "dwa":
            self.vel = True
        else:
            self.vel = False


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

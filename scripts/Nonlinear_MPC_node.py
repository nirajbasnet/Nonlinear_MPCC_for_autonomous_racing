#!/usr/bin/env python
from casadi import *
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
import csv
import os
import time
import numpy as np
import copy

import rospy
from geometry_msgs.msg import Pose, PoseStamped, Quaternion
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Path, Odometry
from std_msgs.msg import Duration, Header
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from visualization_msgs.msg import Marker, MarkerArray
from Nonlinear_MPC import MPC
from osuf1_common.msg import MPC_metadata, MPC_trajectory, MPC_prediction

class MPCKinematicNode:
    def __init__(self):
        rospy.init_node('mpc_node')
        self.param = {'dT': rospy.get_param('dT', 0.2),
                      'N': rospy.get_param('mpc_steps_N', 20),
                      'L': rospy.get_param('vehicle_L', 0.325),
                      'theta_max': rospy.get_param('mpc_max_steering', 0.523),
                      'v_max': rospy.get_param('max_speed', 2.0),  # 5
                      'p_min': rospy.get_param('p_min', 0),
                      'p_max': rospy.get_param('p_max', 3.0),
                      'x_min': rospy.get_param('x_min', -200),
                      'x_max': rospy.get_param('x_max', 200),
                      'y_min': rospy.get_param('y_min', -200),
                      'y_max': rospy.get_param('y_max', 200),
                      'psi_min': rospy.get_param('psi_min', -1000),
                      'psi_max': rospy.get_param('psi_max', 1000),
                      's_min': rospy.get_param('s_min', 0),
                      's_max': rospy.get_param('s_max', 200),
                      'd_v_bound': rospy.get_param('d_v_bound', 2.0),
                      'd_theta_bound': rospy.get_param('d_theta_bound', 0.5),
                      'd_p_bound': rospy.get_param('d_p_bound', 2.0),
                      'ref_vel': rospy.get_param('mpc_ref_vel', 2.0),
                      'mpc_w_cte': rospy.get_param('mpc_w_cte', 750),
                      'mpc_w_s': rospy.get_param('mpc_w_s', 0),
                      'mpc_w_lag': rospy.get_param('mpc_w_lag', 750),
                      'mpc_w_epsi': rospy.get_param('mpc_w_epsi', 400),
                      'mpc_w_vel': rospy.get_param('mpc_w_vel', 0.75),
                      'mpc_w_delta': rospy.get_param('mpc_w_delta', 50),
                      'mpc_w_p': rospy.get_param('mpc_w_p', 5),  # 1
                      'mpc_w_accel': rospy.get_param('mpc_w_accel', 4),
                      'mpc_w_delta_d': rospy.get_param('mpc_w_delta_d', 750),
                      'mpc_w_delta_p': rospy.get_param('mpc_w_delta_p', 0),
                      'spline_poly_order': rospy.get_param('spline_poly_order', 3),
                      'INTEGRATION_MODE': rospy.get_param('integration_mode','Euler'), #can be 'RK4' or 'RK3'
                      'ipopt_verbose': rospy.get_param('ipopt_verbose', True)
                      }

        dirname = os.path.dirname(__file__)
        path_folder_name = rospy.get_param('path_folder_name', 'kelley')
        self.CENTER_TRACK_FILENAME = os.path.join(dirname, path_folder_name + '/centerline_waypoints.csv')
        self.CENTER_DERIVATIVE_FILENAME = os.path.join(dirname, path_folder_name + '/center_spline_derivatives.csv')
        self.RIGHT_TRACK_FILENAME = os.path.join(dirname, path_folder_name + '/right_waypoints.csv')
        self.LEFT_TRACK_FILENAME = os.path.join(dirname, path_folder_name + '/left_waypoints.csv')
        self.CONTROLLER_FREQ = rospy.get_param('controller_freq', 20)
        self.GOAL_THRESHOLD = rospy.get_param('goal_threshold', 0.75)
        self.CAR_WIDTH = rospy.get_param('car_width', 0.30)
        self.INFLATION_FACTOR = rospy.get_param('inflation_factor', 0.9)
        self.LAG_TIME = rospy.get_param('lag_time', 0.1)  # 100ms

        self.DEBUG_MODE = rospy.get_param('debug_mode', True)
        self.DELAY_MODE = rospy.get_param('delay_mode', True)
        self.THROTTLE_MODE = rospy.get_param('throttle_mode',True)
        # Topic name related parameters
        pose_topic = rospy.get_param('localized_pose_topic_name', 'pf/viz/inferred_pose')
        cmd_vel_topic = rospy.get_param('cmd_vel_topic_name', 'vesc/high_level/ackermann_cmd_mux/input/nav_0')
        odom_topic = rospy.get_param('odom_topic_name', 'vesc/odom')
        goal_topic = rospy.get_param('goal_topic_name', '/move_base_simple/goal')
        prediction_pub_topic = rospy.get_param('mpc_prediction_topic', 'mpc_prediction')
        meta_pub_topic = rospy.get_param('mpc_metadata_topic', 'mpc_metadata')
        self.car_frame = rospy.get_param('car_frame', 'base_link')


        # Path related variables
        self.path_points = None
        self.center_lane = None
        self.center_point_angles = None
        self.center_lut_x, self.center_lut_y = None, None
        self.center_lut_dx, self.center_lut_dy = None, None
        self.right_lut_x, self.right_lut_y = None, None
        self.left_lut_x, self.left_lut_y = None, None
        self.element_arc_lengths = None
        self.element_arc_lengths_orig = None


        # Plot related variables
        self.current_time = 0
        self.t_plot = []
        self.v_plot = []
        self.steering_plot = []
        self.cte_plot = []
        self.time_plot = []

        # Minimum distance search related variables
        self.ARC_LENGTH_MIN_DIST_TOL = rospy.get_param('arc_length_min_dist_tol',0.05) #minimum distance between current pose and point on path to calculate arc length travelled without projection

        # Publishers
        self.ackermann_pub = rospy.Publisher(cmd_vel_topic, AckermannDriveStamped, queue_size=10)
        self.mpc_trajectory_pub = rospy.Publisher('mpc_trajectory', Path, queue_size=10)
        self.center_path_pub = rospy.Publisher('center_path', Path, queue_size=10)
        self.right_path_pub = rospy.Publisher('right_path', Path, queue_size=10)
        self.left_path_pub = rospy.Publisher('left_path', Path, queue_size=10)
        self.center_tangent_pub = rospy.Publisher('center_tangent', PoseStamped, queue_size=10)
        self.path_boundary_pub = rospy.Publisher('boundary_marker', MarkerArray, queue_size=10)
        self.prediction_pub = rospy.Publisher(prediction_pub_topic, MPC_trajectory, queue_size=1)
        self.meta_pub = rospy.Publisher(meta_pub_topic, MPC_metadata, queue_size=1)


        # MPC related initializations
        self.mpc = MPC()
        self.mpc.boundary_pub = self.path_boundary_pub
        self.initialize_MPC()
        self.current_pos_x, self.current_pos_y, self.current_yaw, self.current_s = 0.0, 0.0, 0.0, 0.0
        self.current_pose = None
        self.current_vel_odom = 0.0
        self.projected_vel = 0.0
        self.steering_angle = 0.0
        # Goal status related variables
        self.goal_pos = None
        self.goal_reached = False
        self.goal_received = False

        # Subscribers
        rospy.Subscriber(pose_topic, PoseStamped, self.pf_pose_callback, queue_size=1)
        rospy.Subscriber(goal_topic, PoseStamped, self.goalCB, queue_size=1)
        rospy.Subscriber(odom_topic, Odometry, self.odomCB, queue_size=1)
        # Timer callback function for the control loop
        rospy.Timer(rospy.Duration(1.0 / self.CONTROLLER_FREQ), self.controlLoopCB)

    def initialize_MPC(self):
        self.preprocess_track_data()
        self.param['s_max'] = self.element_arc_lengths[-1]
        self.mpc.set_initial_params(self.param)
        self.mpc.set_track_data(self.center_lut_x, self.center_lut_y, self.center_lut_dx, self.center_lut_dy,
                                self.right_lut_x, self.right_lut_y, self.left_lut_x, self.left_lut_y,
                                self.element_arc_lengths, self.element_arc_lengths_orig[-1])
        self.mpc.setup_MPC()

    def create_header(self, frame_id):
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = frame_id
        return header

    def find_nearest_index(self, car_pos):
        distances_array = np.linalg.norm(self.center_lane - car_pos, axis=1)
        min_dist_idx = np.argmin(distances_array)
        return min_dist_idx, distances_array[min_dist_idx]

    def heading(self, yaw):
        q = quaternion_from_euler(0, 0, yaw)
        return Quaternion(*q)

    def quaternion_to_euler_yaw(self, orientation):
        _, _, yaw = euler_from_quaternion((orientation.x, orientation.y, orientation.z, orientation.w))
        return yaw

    def read_waypoints_array_from_csv(self, filename):
        '''read waypoints from given csv file and return the data in the form of numpy array'''
        if filename == '':
            raise ValueError('No any file path for waypoints file')
        with open(filename) as f:
            path_points = [tuple(line) for line in csv.reader(f, delimiter=',')]
        path_points = np.array([[float(point[0]), float(point[1])] for point in path_points])
        return path_points

    def pf_pose_callback(self, msg):
        '''acquire estimated pose of car from particle filter'''
        self.current_pos_x = msg.pose.position.x
        self.current_pos_y = msg.pose.position.y
        self.current_yaw = self.quaternion_to_euler_yaw(msg.pose.orientation)
        self.current_pose = [self.current_pos_x, self.current_pos_y, self.current_yaw]
        if self.goal_received:
            car2goal_x = self.goal_pos.x - self.current_pos_x
            car2goal_y = self.goal_pos.y - self.current_pos_y
            dist2goal = sqrt(car2goal_x * car2goal_x + car2goal_y * car2goal_y)
            if dist2goal < self.GOAL_THRESHOLD:
                self.goal_reached = True
                self.goal_received = False
                self.mpc.WARM_START = False
                self.mpc.init_mpc_start_conditions()
                rospy.loginfo("Goal Reached !")
                self.plot_data()

    def odomCB(self, msg):
        '''Get odometry data especially velocity from the car'''
        self.current_vel_odom = msg.twist.twist.linear.x

    def goalCB(self, msg):
        '''Get goal pose from the user'''
        self.goal_pos = msg.pose.position
        self.goal_received = True
        self.goal_reached = False
        if self.DEBUG_MODE:
            print("Goal pos=", self.goal_pos)

    def publish_path(self, waypoints, publisher):
        # Visualize path derived from the given waypoints in the path
        path = Path()
        path.header = self.create_header('map')
        path.poses = []
        for point in waypoints:
            tempPose = PoseStamped()
            tempPose.header = path.header
            tempPose.pose.position.x = point[0]
            tempPose.pose.position.y = point[1]
            tempPose.pose.orientation.w = 1.0
            path.poses.append(tempPose)
        publisher.publish(path)

    def get_interpolated_path(self, pts, arc_lengths_arr, smooth_value=0.1, scale=2, derivative_order=0):
        # tck represents vector of knots, the B-spline coefficients, and the degree of the spline.
        tck, u = splprep(pts.T, u=arc_lengths_arr, s=smooth_value, per=1)
        u_new = np.linspace(u.min(), u.max(), len(pts) * scale)
        x_new, y_new = splev(u_new, tck, der=derivative_order)
        interp_points = np.concatenate((x_new.reshape((-1, 1)), y_new.reshape((-1, 1))), axis=1)
        return interp_points, tck

    def get_interpolated_path_casadi(self, label_x, label_y, pts, arc_lengths_arr):
        u = arc_lengths_arr
        V_X = pts[:, 0]
        V_Y = pts[:, 1]
        lut_x = interpolant(label_x, 'bspline', [u], V_X)
        lut_y = interpolant(label_y, 'bspline', [u], V_Y)
        return lut_x, lut_y

    def get_arc_lengths(self, waypoints):
        d = np.diff(waypoints, axis=0)
        consecutive_diff = np.sqrt(np.sum(np.power(d, 2), axis=1))
        dists_cum = np.cumsum(consecutive_diff)
        dists_cum = np.insert(dists_cum, 0, 0.0)
        return dists_cum

    def inflate_track_boundaries(self, center_lane, side_lane, car_width=0.325, inflation_factor=1.2):
        for idx in range(len(center_lane)):
            lane_vector = side_lane[idx, :] - center_lane[idx, :]
            side_track_width = np.linalg.norm(lane_vector)
            side_unit_vector = lane_vector / side_track_width
            side_lane[idx, :] = center_lane[idx, :] + side_unit_vector * (
                    side_track_width - car_width * inflation_factor)
        return side_lane

    def preprocess_track_data(self):
        center_lane = self.read_waypoints_array_from_csv(self.CENTER_TRACK_FILENAME)
        center_derivative_data = self.read_waypoints_array_from_csv(self.CENTER_DERIVATIVE_FILENAME)
        right_lane = self.read_waypoints_array_from_csv(self.RIGHT_TRACK_FILENAME)
        left_lane = self.read_waypoints_array_from_csv(self.LEFT_TRACK_FILENAME)

        right_lane = self.inflate_track_boundaries(center_lane, right_lane, self.CAR_WIDTH, self.INFLATION_FACTOR)
        left_lane = self.inflate_track_boundaries(center_lane, left_lane, self.CAR_WIDTH, self.INFLATION_FACTOR)

        self.center_lane = np.row_stack((center_lane, center_lane[1:int(center_lane.shape[0] / 2), :]))
        right_lane = np.row_stack((right_lane, right_lane[1:int(center_lane.shape[0] / 2), :]))
        left_lane = np.row_stack((left_lane, left_lane[1:int(center_lane.shape[0] / 2), :]))
        center_derivative_data = np.row_stack(
            (center_derivative_data, center_derivative_data[1:int(center_lane.shape[0] / 2), :]))
        # print self.center_lane.shape,right_lane.shape,left_lane.shape,center_derivative_data

        # Interpolate center line upto desired resolution
        self.element_arc_lengths_orig = self.get_arc_lengths(center_lane)
        self.element_arc_lengths = self.get_arc_lengths(self.center_lane)
        self.center_lut_x, self.center_lut_y = self.get_interpolated_path_casadi('lut_center_x', 'lut_center_y',
                                                                                 self.center_lane,
                                                                                 self.element_arc_lengths)
        self.center_lut_dx, self.center_lut_dy = self.get_interpolated_path_casadi('lut_center_dx', 'lut_center_dy',
                                                                                   center_derivative_data,
                                                                                   self.element_arc_lengths)
        self.center_point_angles = np.arctan2(center_derivative_data[:, 1], center_derivative_data[:, 0])

        # Interpolate right and left wall line
        self.right_lut_x, self.right_lut_y = self.get_interpolated_path_casadi('lut_right_x', 'lut_right_y', right_lane,
                                                                               self.element_arc_lengths)
        self.left_lut_x, self.left_lut_y = self.get_interpolated_path_casadi('lut_left_x', 'lut_left_y', left_lane,
                                                                             self.element_arc_lengths)
        for i in range(5):
            self.publish_path(center_lane, self.center_path_pub)
            self.publish_path(right_lane, self.right_path_pub)
            self.publish_path(left_lane, self.left_path_pub)
            rospy.sleep(0.2)

    def find_current_arc_length(self, car_pos):
        nearest_index, minimum_dist = self.find_nearest_index(car_pos)
        if minimum_dist > self.ARC_LENGTH_MIN_DIST_TOL:
            if nearest_index == 0:
                next_idx = 1
                prev_idx = self.center_lane.shape[0] - 1
            elif nearest_index == (self.center_lane.shape[0] - 1):
                next_idx = 0
                prev_idx = self.center_lane.shape[0] - 2
            else:
                next_idx = nearest_index + 1
                prev_idx = nearest_index - 1
            dot_product_value = np.dot(car_pos - self.center_lane[nearest_index, :],
                                       self.center_lane[prev_idx, :] - self.center_lane[nearest_index, :])
            if dot_product_value > 0:
                nearest_index_actual = prev_idx
            else:
                nearest_index_actual = nearest_index
                nearest_index = next_idx
            new_dot_value = np.dot(car_pos - self.center_lane[nearest_index_actual, :],
                                   self.center_lane[nearest_index, :] - self.center_lane[nearest_index_actual, :])
            projection = new_dot_value / np.linalg.norm(
                self.center_lane[nearest_index, :] - self.center_lane[nearest_index_actual, :])
            current_s = self.element_arc_lengths[nearest_index_actual] + projection
        else:
            current_s = self.element_arc_lengths[nearest_index]

        ###temporary fix##
        if nearest_index==0:
            current_s=0.0
        return current_s, nearest_index


    def controlLoopCB(self, event):
        '''Control loop for car MPC'''
        if self.goal_received and not self.goal_reached:
            control_loop_start_time = time.time()
            # Update system states: X=[x, y, psi]
            px = copy.deepcopy(self.current_pos_x)
            py = copy.deepcopy(self.current_pos_y)
            car_pos = np.array([px, py])
            psi = copy.deepcopy(self.current_yaw)

            # Update system inputs: U=[speed(v), steering]
            v = copy.deepcopy(self.current_vel_odom)
            steering = self.steering_angle  # radian
            L = self.mpc.L

            current_s, near_idx = self.find_current_arc_length(car_pos)
            print "pre",current_s,near_idx
            if self.DELAY_MODE:
                dt_lag = self.LAG_TIME
                px = px + v * np.cos(psi) * dt_lag
                py = py + v * np.sin(psi) * dt_lag
                psi = psi + (v / L) * tan(steering) * dt_lag
                current_s = current_s + self.projected_vel * dt_lag

            current_state = np.array([px, py, psi, current_s])

            centerPose = PoseStamped()
            centerPose.header = self.create_header('map')
            centerPose.pose.position.x = float(self.center_lane[near_idx, 0])
            centerPose.pose.position.y = float(self.center_lane[near_idx, 1])
            centerPose.pose.orientation = self.heading(self.center_point_angles[near_idx])
            self.center_tangent_pub.publish(centerPose)

            # Solve MPC Problem
            mpc_time = time.time()
            first_control, trajectory, control_inputs = self.mpc.solve(current_state)
            mpc_compute_time = time.time() - mpc_time

            # MPC result (all described in car frame)
            speed = float(first_control[0])  # speed
            steering = float(first_control[1])  # radian
            self.projected_vel = speed

            #throttle calculation
            throttle = 0.03*(speed - v)/ self.param['dT']

            if throttle>1:
                throttle=1
            elif throttle<-1:
                throttle=-1
            if speed ==0:
                throttle=0

            if not self.mpc.WARM_START:
                speed, steering,throttle = 0, 0, 0
                self.mpc.WARM_START = True
            if (speed >= self.param['v_max']):
                speed = self.param['v_max']
            elif (speed <= (- self.param['v_max'] / 2.0)):
                speed = - self.param['v_max'] / 2.0

            # Display the MPC predicted trajectory
            mpc_traj = Path()
            mpc_traj.header = self.create_header('map')
            mpc_traj.poses = []
            for i in range(trajectory.shape[0]):
                tempPose = PoseStamped()
                tempPose.header = mpc_traj.header
                tempPose.pose.position.x = trajectory[i, 0]
                tempPose.pose.position.y = trajectory[i, 1]
                tempPose.pose.orientation = self.heading(trajectory[i, 2])
                mpc_traj.poses.append(tempPose)
            self.mpc_trajectory_pub.publish(mpc_traj)

            # publish the mpc related metadata to hylaa node
            #metadata message
            meta = MPC_metadata()
            meta.header = self.create_header('map')
            meta.dt = self.param['dT']
            meta.horizon = self.param['N']*self.param['dT']
            self.meta_pub.publish(meta)

            #publish mpc prediction results message
            mpc_prediction_results = []
            for i in range(control_inputs.shape[0]):
                mpc_prediction_states = [trajectory[i, 0], trajectory[i, 1], trajectory[i, 2],0.1]
                mpc_prediction_inputs = [control_inputs[i, 0], control_inputs[i, 1]]
                mpc_prediction_results.append((mpc_prediction_states, mpc_prediction_inputs))

            trajectory = MPC_trajectory()
            trajectory.header = meta.header
            trajectory.trajectory = [MPC_prediction(pred[0], pred[1]) for pred in mpc_prediction_results]
            self.prediction_pub.publish(trajectory)

            total_time = time.time() - control_loop_start_time
            if self.DEBUG_MODE:
                rospy.loginfo("DEBUG")
                rospy.loginfo("psi: %s ", psi)
                rospy.loginfo("V: %s", v)
                rospy.loginfo("Throttle: %s", throttle)
                rospy.loginfo("Control loop time mpc= %s:", mpc_compute_time)
                rospy.loginfo("Control loop time=: %s", total_time)

            self.current_time += 1.0 / self.CONTROLLER_FREQ
            # self.cte_plot.append(cte)
            self.t_plot.append(self.current_time)
            self.v_plot.append(speed)
            self.steering_plot.append(np.rad2deg(steering))
            self.time_plot.append(mpc_compute_time * 1000)
        else:
            steering = 0.0
            speed = 0.0
            throttle=0.0

        # publish cmd
        ackermann_cmd = AckermannDriveStamped()
        ackermann_cmd.header = self.create_header(self.car_frame)
        ackermann_cmd.drive.steering_angle = steering
        self.steering_angle = steering
        ackermann_cmd.drive.speed = speed
        if self.THROTTLE_MODE:
            ackermann_cmd.drive.acceleration = throttle
        self.ackermann_pub.publish(ackermann_cmd)

    def plot_data(self):
        plt.figure(1)
        plt.subplot(411)
        plt.step(self.t_plot, self.v_plot, 'k', linewidth=1.5)
        # plt.ylim(-0.2, 0.8)
        plt.ylabel('v m/s')
        plt.xlabel('time(s)')
        plt.subplot(412)
        plt.step(self.t_plot, self.steering_plot, 'r', linewidth=1.5)
        # plt.ylim(-0.5, 1.0)
        plt.ylabel('steering angle(degrees)')
        plt.xlabel('time(s)')
        # plt.subplot(413)
        # plt.step(self.t_plot, self.cte_plot, 'g', linewidth=1.5)
        # # plt.ylim(-0.5, 1.0)
        # plt.ylabel('cte in m')
        plt.subplot(414)
        plt.step(self.t_plot, self.time_plot, 'b', linewidth=1.5)
        plt.ylim(0.0, 100)
        plt.ylabel('mpc_compute_time in ms')
        plt.xlabel('time(s)')
        plt.show()

        self.t_plot = []
        self.steering_plot = []
        self.v_plot = []
        self.cte_plot = []
        self.time_plot = []
        self.current_time = 0


if __name__ == '__main__':
    mpc_node = MPCKinematicNode()
    rospy.spin()

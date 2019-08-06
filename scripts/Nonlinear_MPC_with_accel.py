#!/usr/bin/env python
from casadi import *
import matplotlib.pyplot as plt
import csv
import time
import numpy as np

import rospy
import copy
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion, Twist
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Path, Odometry
from std_msgs.msg import Duration, Header
from tf.transformations import quaternion_from_euler, euler_from_quaternion


# ideas
# 1) send polynomial coefficients through P and formulate objective function with parameters P and x0. This way, there is no need to initialize optimization function everytime.

class MPC:
    def __init__(self):
        self.dT = 0.1
        self.N = 30
        self.L = 0.325
        self.v_max = 0.6
        self.v_min = -self.v_max
        self.theta_max = pi / 6
        self.theta_min = -self.theta_max
        self.x_min, self.x_max, self.y_min, self.y_max, self.psi_min, self.psi_max = None, None, None, None, None, None
        self.n_states, self.n_controls, self.T_V = None, None, None
        self.f = None
        self.U = None
        self.P = None
        self.X = None
        self.obj = 0
        self.f0 = 0
        self.psides0 = 0
        self.coeffs = None
        self.X0 = None  # initial estimate for the states solution
        self.u0 = None  # initial estimate for the controls solution
        self.g = []
        self.Q = None
        self.R = None
        self.opts = {}
        self.param = {}
        self.lbg, self.ubg = None, None
        self.lbx, self.ubx = None, None
        self.nlp = None
        self.solver = None
        self.NDP = 4   #3 degree polynomial coefficients


    def setup_MPC(self):
        self.init_system_model()
        self.init_constraints()
        self.compute_optimization_cost()
        self.init_ipopt_solver()
        self.init_mpc_start_conditions()


    def init_system_model(self):
        x = SX.sym('x')
        y = SX.sym('y')
        psi = SX.sym('psi')
        v = SX.sym('v')
        cte = SX.sym('cte')
        epsi = SX.sym('epsi')

        a = SX.sym('a')
        theta = SX.sym('theta')

        states = vertcat(x, y, psi,v)
        controls = vertcat(a, theta)
        self.n_states = states.size1()
        self.n_controls = controls.size1()
        self.T_V = self.n_states + self.n_controls
        rhs = vertcat(v * cos(psi), v * sin(psi), (v / self.L) * theta,a)
        self.f = Function('f', [states, controls], [rhs])  # nonlinear mapping function f(x,u)
        self.U = SX.sym('U', self.n_controls, self.N)

        self.P = SX.sym('P', self.NDP + self.n_states + self.N * (self.n_states + self.n_controls))
        # parameters (which include the nth degree polynomial coefficients, initial state and the reference along the predicted trajectory (reference states and reference controls))

        self.X = SX.sym('X', self.n_states, (self.N + 1))
        # A vector that represents the states over the optimization problem

        self.Q = SX.zeros(self.n_states, self.n_states)
        self.Q[0, 0] = 0
        self.Q[1, 1] = 70
        self.Q[2, 2] = 70  # weighing matrices (states)
        self.Q[3,3]=5

        self.R = SX.zeros(self.n_controls, self.n_controls)
        self.R[0, 0] = 5
        self.R[1, 1] = 5  # weighing matrices (controls)

        self.obj = 0  # Objective function
        self.g = []  # constraints vector



    def set_initial_params(self, param):
        '''Set initial parameters related to MPC'''
        self.param = param
        self.dT = param['dT']
        self.N = param['N']
        self.L = param['L']
        self.theta_max, self.v_max = param['theta_max'], param['v_max']
        self.x_min, self.x_max = param['x_min'], param['x_max']
        self.y_min, self.y_max = param['y_min'], param['y_max']
        self.psi_min, self.psi_max = param['psi_min'], param['psi_max']

    def get_f0(self, x0):
        f0 = 0
        n = self.NDP - 1
        for i in range(n + 1):
            f0 += self.P[i] * power(x0, n - i)
        return f0

    def get_psides(self, x0):
        psides0 = 0
        n = self.NDP - 1
        for i in range(0, n):
            psides0 += (n - i) * self.P[i] * power(x0, n - i - 1)
        angle = atan(psides0)
        return angle

    def compute_optimization_cost(self):
        st = self.X[:, 0]  # initial state
        self.g = vertcat(self.g, st - self.P[self.NDP:self.NDP+self.n_states])  # initial condition constraints

        for k in range(self.N):
            st = self.X[:, k]
            con = self.U[:, k]

            self.f0 = self.get_f0(st[0])
            self.psides0 = self.get_psides(st[0])
            cte = self.f0 - st[1]
            epsi = st[2] - self.psides0
            ref = vertcat(0, self.f0, self.psides0,1.0)

            self.obj = self.obj + mtimes(mtimes((st - ref).T, self.Q), (st - ref)) + mtimes(
                mtimes((con - self.P[self.NDP+2 * self.n_states + self.T_V * k:self.NDP+2 * self.n_states + self.n_controls + self.T_V * k]).T, self.R),
                (con - self.P[self.NDP+2 * self.n_states + self.T_V * k:self.NDP+2 * self.n_states + self.n_controls + self.T_V * k]))

            st_next = self.X[:, k + 1]
            f_value = self.f(st, con)
            st_next_euler = st + (self.dT * f_value)
            self.g = vertcat(self.g, st_next - st_next_euler)  # compute constraints

        # print(self.obj)

    def init_ipopt_solver(self):
        # Make the decision/optimization variables one large single column  vector
        OPT_variables = vertcat(reshape(self.X, self.n_states * (self.N + 1), 1), reshape(self.U, 2 * self.N, 1))
        self.opts["ipopt"] = {}
        self.opts["ipopt"]["max_iter"] = 2000
        self.opts["ipopt"]["print_level"] = 0
        # self.opts["verbose"] = True
        # self.opts["verbose_init"] = True
        self.opts["print_time"] = 0
        self.opts["ipopt"]["acceptable_tol"] = 1e-8
        self.opts["ipopt"]["acceptable_obj_change_tol"] = 1e-6

        self.nlp_prob = {'f': self.obj, 'x': OPT_variables, 'g': self.g, 'p': self.P}
        self.solver = nlpsol('solver', 'ipopt', self.nlp_prob, self.opts)

    def init_constraints(self):
        '''Initialize constraints for states, dynamic model state transitions and control inputs of the system'''
        self.lbg = np.zeros((self.n_states * (self.N + 1), 1))
        self.ubg = np.zeros((self.n_states * (self.N + 1), 1))
        self.lbx = np.zeros((self.n_states + (self.n_states + self.n_controls) * self.N, 1))
        self.ubx = np.zeros((self.n_states + (self.n_states + self.n_controls) * self.N, 1))
        for k in range(self.N + 1):
            self.lbx[self.n_states * k:self.n_states * (k + 1), 0] = np.array([[self.x_min, self.y_min, self.psi_min,-2.0]])
            self.ubx[self.n_states * k:self.n_states * (k + 1), 0] = np.array([[self.x_max, self.y_max, self.psi_max,2.0]])
        state_count = self.n_states * (self.N + 1)
        for k in range(self.N):
            self.lbx[state_count:state_count + self.n_controls, 0] = np.array(
                [[self.v_min, self.theta_min]])  # v and theta lower bound
            self.ubx[state_count:state_count + self.n_controls, 0] = np.array(
                [[self.v_max, self.theta_max]])  # v and theta upper bound
            state_count += self.n_controls

    def polyeval(self, coeffs, x):
        '''Evalulatea a polynomial at a given point'''
        result = 0.0
        n = len(coeffs) - 1
        for i in range(n + 1):
            result += coeffs[i] * pow(x, n - i)
        return result

    def poly_heading_eval(self, coeffs, x):
        result = 0.0
        n = len(coeffs) - 1
        for i in range(0, n):
            result += (n - i) * coeffs[i] * pow(x, n - i - 1)
        angle = atan(result)
        return angle

    def init_mpc_start_conditions(self):
        self.u0 = np.zeros((self.N, self.n_controls))
        self.X0 = np.zeros((self.n_states, self.N + 1))

    def solve(self, initial_state):
        p = np.zeros(self.NDP+ self.n_states + self.N * (self.n_states + self.n_controls))
        p[0:self.NDP]=self.coeffs
        p[self.NDP:self.NDP+self.n_states] = initial_state  # initial condition of the robot posture
        for k in range(self.N):  # new - set the reference to track
            x_ref = 0
            y_ref = 0
            psi_ref = 0
            v_ref = 0.75
            a_ref=0
            theta_ref = 0
            p[self.NDP+ self.n_states + self.T_V * k:self.NDP+ 2 * self.n_states + self.T_V * k] = [x_ref, y_ref, psi_ref,v_ref]
            p[self.NDP+2 * self.n_states + self.T_V * k:self.NDP+2 * self.n_states + self.n_controls + self.T_V * k] = [a_ref,
                                                                                                         theta_ref]

        # Initial value of the optimization variables
        x_init = vertcat(reshape(self.X0.T, self.n_states * (self.N + 1), 1),reshape(self.u0.T, self.n_controls * self.N, 1))
        # Solve using ipopt by giving the initial estimate and parameters as well as bounds for constraints
        sol = self.solver(x0=x_init, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg, p=p)
        # Get state and control solution
        self.X0 = reshape(sol['x'][0:self.n_states * (self.N + 1)], self.n_states, self.N + 1).T  # get soln trajectory
        u = reshape(sol['x'][self.n_states * (self.N + 1):], self.n_controls, self.N).T  # get controls solution
        # Get the first control
        con_first = u[0, :].T
        trajectory = self.X0.full()  # size is (N+1,n_states)
        # full converts casadi data type to python data type(numpy array)

        # Shift trajectory and control solution to initialize the next step
        self.X0 = vertcat(self.X0[1:, :], self.X0[self.X0.size1() - 1, :])
        self.u0 = vertcat(u[1:, :], u[u.size1() - 1, :])
        return con_first, trajectory


########################################################################################################################
########################################################################################################################
class MPCKinematicNode:
    def __init__(self):
        rospy.init_node('mpc_node')
        self.param = {'dT': rospy.get_param('dT', 0.1),
                      'N': rospy.get_param('N', 15),
                      'L': rospy.get_param('vehicle_L', 0.325),
                      'theta_max': rospy.get_param('theta_max', 0.523),
                      'v_max': rospy.get_param('v_max', 1.0),
                      'x_min': rospy.get_param('x_min', -30),
                      'x_max': rospy.get_param('x_max', 30),
                      'y_min': rospy.get_param('y_min', -3),
                      'y_max': rospy.get_param('y_max', 3),
                      'psi_min': rospy.get_param('psi_min', -3.14),
                      'psi_max': rospy.get_param('psi_max', 3.14)}
        filename = rospy.get_param('~waypoints_filepath', '')
        self.LOCAL_PATHLENGTH = rospy.get_param('local_path_length', 4.0)
        self.WAYPOINT_FOV = rospy.get_param('waypoints_fov', 1.57)
        self.GOAL_THRESHOLD = rospy.get_param('goal_threshold', 0.2)
        self.DEBUG_MODE = rospy.get_param('debug_mode', False)
        self.TWIST_PUB_MODE = rospy.get_param('pub_twist_cmd', False)

        pose_topic = rospy.get_param('localized_pose_topic_name', '/pf/viz/inferred_pose')
        cmd_vel_topic = rospy.get_param('cmd_vel_topic_name', '/vesc/high_level/ackermann_cmd_mux/input/nav_0')
        odom_topic = rospy.get_param('odom_topic_name', '/vesc/odom')
        goal_topic = rospy.get_param('goal_topic_name', '/move_base_simple/goal')
        self.car_frame = rospy.get_param('car_frame', 'base_link')

        self.mpc = MPC()
        self.mpc.set_initial_params(self.param)
        self.mpc.setup_MPC()
        self.current_pos_x, self.current_pos_y, self.current_yaw = 0.0, 0.0, 0.0
        self.current_pose = None
        self.current_vel_odom = 0.0
        self.steering_angle = 0.0

        self.total_path = self.read_waypoints_from_csv(filename)
        self.local_path = Path()
        self.path_from_coeffs = Path()

        self.goal_pos = None
        self.goal_reached = False
        self.goal_received = False

        self.ackermann_pub = rospy.Publisher(cmd_vel_topic, AckermannDriveStamped, queue_size=1)
        self.mpc_trajectory_pub = rospy.Publisher('/mpc_trajectory', Path, queue_size=1)
        self.mpc_reference_pub = rospy.Publisher('/mpc_reference', Path, queue_size=1)
        self.mpc_coeffs_path_pub = rospy.Publisher('/mpc_coeffs_path', Path, queue_size=1)
        if self.TWIST_PUB_MODE:
            self.twist_pub = rospy.Publisher(cmd_vel_topic, Twist, queue_size=1)

        rospy.Subscriber(pose_topic, PoseStamped, self.pf_pose_callback, queue_size=1)
        rospy.Subscriber(goal_topic, PoseStamped, self.goalCB, queue_size=1)
        rospy.Subscriber(odom_topic, Odometry, self.odomCB, queue_size=1)
        rospy.Timer(rospy.Duration(0.1), self.controlLoopCB)

    def path_to_local_coord(self):
        '''convert path to vehicle coordinate system'''

    def create_header(self, frame_id):
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = frame_id
        return header

    def get_nearby_waypoints(self):
        '''Get nearby waypoints on the path'''
        self.local_path.poses = []
        self.local_path.header = self.create_header('map')
        for i in range(len(self.total_path.poses)):
            dx = self.total_path.poses[i].pose.position.x - self.current_pos_x
            dy = self.total_path.poses[i].pose.position.y - self.current_pos_y
            dist = sqrt(dx * dx + dy * dy)
            delta_yaw = atan2(dy, dx) - self.current_yaw
            if (((dist < self.LOCAL_PATHLENGTH) and (fabs(delta_yaw) < self.WAYPOINT_FOV)) or (dist < 1.25)):
                tempPose = PoseStamped()
                tempPose.header = self.local_path.header
                tempPose.pose.position.x = self.total_path.poses[i].pose.position.x
                tempPose.pose.position.y = self.total_path.poses[i].pose.position.y
                tempPose.pose.orientation.w = 1.0
                self.local_path.poses.append(tempPose)

    def heading(self, yaw):
        q = quaternion_from_euler(0, 0, yaw)
        return Quaternion(*q)

    def quaternion_to_euler_yaw(self, orientation):
        _, _, yaw = euler_from_quaternion((orientation.x, orientation.y, orientation.z, orientation.w))
        return yaw

    def read_waypoints_from_csv(self, filename):
        '''read waypoints from given csv file and return the data in the form of nav_msgs::Path'''
        path = Path()
        path.header = self.create_header('map')
        if filename == '':
            raise ValueError('No any file path for waypoints file')
        with open(filename) as f:
            path_points = [tuple(line) for line in csv.reader(f, delimiter=',')]
        path_points = [(float(point[0]), float(point[1]), float(point[2])) for point in path_points]
        for point in path_points:
            header = self.create_header('map')
            waypoint = Pose(Point(float(point[0]), float(point[1]), 0), self.heading(0.0))
            path.poses.append(PoseStamped(header, waypoint))
        return path

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
                rospy.loginfo("Goal Reached !")
        # if self.DEBUG_MODE:
        #     print("Robot pose=",self.current_pose)

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

    def publish_path_from_coeffs(self, coeffs):
        x_coords = np.linspace(0., self.LOCAL_PATHLENGTH, int((self.LOCAL_PATHLENGTH - 0) / 0.1))
        self.path_from_coeffs.header = self.create_header(self.car_frame)
        self.path_from_coeffs.poses = []
        for x in x_coords:
            tempPose = PoseStamped()
            tempPose.header = self.path_from_coeffs.header
            tempPose.pose.position.x = x
            tempPose.pose.position.y = np.polyval(self.mpc.coeffs,x)
            tempPose.pose.orientation.w = 1.0
            self.path_from_coeffs.poses.append(tempPose)
        self.mpc_coeffs_path_pub.publish(self.path_from_coeffs)

    def controlLoopCB(self, event):
        '''Control loop for car MPC'''
        if self.goal_received and not self.goal_reached:
            control_loop_start_time = time.time()
            # Update system states: X=[x, y, psi, v]
            px = self.current_pos_x
            py = self.current_pos_y
            psi = self.current_yaw
            v = self.current_vel_odom

            # Update system inputs: U=[steering, throttle]
            steering = self.steering_angle  # radian
            dt = self.mpc.dT
            L = self.mpc.L

            # Waypoints related parameters
            self.get_nearby_waypoints()
            self.mpc_reference_pub.publish(self.local_path)

            NW = len(self.local_path.poses)  # Number of waypoints
            cospsi = cos(psi)
            sinpsi = sin(psi)
            print("Control loop time1=:", time.time() - control_loop_start_time)

            # Convert to the vehicle coordinate system
            x_veh = np.zeros(NW)
            y_veh = np.zeros(NW)
            for i in range(NW):
                dx = self.local_path.poses[i].pose.position.x - px
                dy = self.local_path.poses[i].pose.position.y - py
                x_veh[i] = dx * cospsi + dy * sinpsi
                y_veh[i] = dy * cospsi - dx * sinpsi

            # Fit waypoints

            self.mpc.coeffs = np.polyfit(x_veh, y_veh, 3)
            if self.DEBUG_MODE:
                print("poly coeffs=",self.mpc.coeffs)
            self.publish_path_from_coeffs(self.mpc.coeffs)
            print("Control loop time2=:", time.time() - control_loop_start_time)
            # cte = self.polyeval(coeffs, 0.0)
            # epsi = atan(coeffs[1])

            current_state= np.array([0.0,0.0,0.0,v])


            print("Control loop time_mpc=:", time.time() - control_loop_start_time)
            # Solve MPC Problem
            first_control,trajectory = self.mpc.solve(current_state)
            print("Control loop time3=:", time.time() - control_loop_start_time)

            # MPC result (all described in car frame)
            steering = first_control[1] #radian
            speed = v+first_control[0]*dt  # speed
            if (speed >= self.param['v_max']):
                speed = self.param['v_max']
            elif (speed <= (- self.param['v_max'] / 2.0)):
                speed = - self.param['v_max'] / 2.0

            # Display the MPC predicted trajectory
            mpc_traj = Path()
            mpc_traj.header= self.create_header(self.car_frame)
            mpc_traj.poses=[]
            for i in range(trajectory.shape[0]):
                tempPose = PoseStamped()
                tempPose.header = mpc_traj.header
                tempPose.pose.position.x = trajectory[i,0]
                tempPose.pose.position.y = trajectory[i,1]
                tempPose.pose.orientation.w = 1.0
                mpc_traj.poses.append(tempPose)
            print("Control loop time4=:", time.time() - control_loop_start_time)

            # publish the mpc trajectory
            self.mpc_trajectory_pub.publish(mpc_traj)
            if self.DEBUG_MODE:
                print("DEBUG")
                print("psi: ",psi)
                print("accel: ",first_control[0] )
                print("V: ",v)
                print("coeffs: ",self.mpc.coeffs)
                print("_steering:",steering)
                # print("_throttle: \n",throttle)
                print("_speed: ",speed)
                # print("mpc_speed: \n",mpc_results[2])
                print("Control loop time=:",time.time()-control_loop_start_time)

        else:
            steering = 0.0
            speed = 0.0
            if (self.goal_reached and self.goal_received):
                print("Goal Reached !")

        # publish cmd
        ackermann_cmd = AckermannDriveStamped()
        ackermann_cmd.header = self.create_header(self.car_frame)
        ackermann_cmd.drive.steering_angle = steering
        ackermann_cmd.drive.speed = speed
        # ackermann_cmd.drive.acceleration = throttle
        self.ackermann_pub.publish(ackermann_cmd)

        #
        # if self.TWIST_PUB_MODE:
        #     twist_msg = Twist()
        #     twist_msg.linear.x = speed
        #     twist_msg.angular.z = steering
        #     self.twist_pub.publish(twist_msg)


if __name__ == '__main__':
    mpc_node = MPCKinematicNode()
    rospy.spin()

#!/usr/bin/env python
from casadi import *
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
import csv
import time
import numpy as np

import rospy
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion,Vector3
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Path, Odometry
from std_msgs.msg import Duration, Header,ColorRGBA
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from visualization_msgs.msg import Marker,MarkerArray


class MPC:
    def __init__(self):
        self.dT = 0.1
        self.N = 30
        self.L = 0.325
        self.v_max = 0.6
        self.v_min = -self.v_max
        self.theta_max = pi / 6
        self.theta_min = -self.theta_max
        self.s_min, self.s_max = 0, 10
        self.p_min = 0
        self.p_max = 0.6
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
        self.NDP = 4  # number of coefficients for curve fitting
        self.N_OBST = 1

        self.center_lut_x, self.center_lut_y = None, None
        self.center_lut_dx, self.center_lut_dy = None, None
        self.right_lut_x, self.right_lut_y = None, None
        self.left_lut_x, self.left_lut_y = None, None
        self.element_arc_lengths = None
        self.arc_lengths_orig_l= None

        self.WARM_START = False
        self.p_initial = 2.75      #projected centerline vel can set to desired value for initial estimation

        self.boundary_pub = None

    def setup_MPC(self):
        self.init_system_model()
        self.init_constraints()
        self.compute_optimization_cost()
        self.init_ipopt_solver()
        self.init_mpc_start_conditions()

    def init_system_model(self):
        # States
        x = MX.sym('x')
        y = MX.sym('y')
        psi = MX.sym('psi')
        s = MX.sym('s')
        # Controls
        v = MX.sym('v')
        theta = MX.sym('theta')
        p = MX.sym('p')

        states = vertcat(x, y, psi, s)
        controls = vertcat(v, theta, p)
        self.n_states = states.size1()
        self.n_controls = controls.size1()
        self.T_V = self.n_states + self.n_controls
        rhs = vertcat(v * cos(psi), v * sin(psi), (v / self.L) * tan(theta), p)  # dynamic equations of the states
        self.f = Function('f', [states, controls], [rhs])  # nonlinear mapping function f(x,u)
        self.U = MX.sym('U', self.n_controls, self.N)

        self.P = MX.sym('P',self.n_states+ 2*self.N+4*self.N_OBST+ self.N *self.n_controls)
        # parameters (which include the nth degree polynomial coefficients, initial state and the reference along the predicted trajectory (reference states and reference controls))

        self.X = MX.sym('X', self.n_states, (self.N + 1))
        # A vector that represents the states over the optimization problem

        self.Q = MX.zeros(2, 2)
        self.Q[0, 0] = self.param['mpc_w_cte']  # cross track error
        self.Q[1, 1] = self.param['mpc_w_lag']  # lag error
        # self.Q[2, 2] = self.param['mpc_w_epsi']  # heading error. weighing matrices (states)
        # self.Q[3, 3] = self.param['mpc_w_s']
        self.R = MX.zeros(3, 3)
        self.R[0, 0] = self.param['mpc_w_vel']  # use of velocity control
        self.R[1, 1] = self.param['mpc_w_delta']  # use of steering actuator.  weighing matrices (controls)
        self.R[2, 2] = self.param['mpc_w_p']  # projected velocity input for progress along the track

        self.S = MX.zeros(3, 3)
        self.S[0, 0] = self.param['mpc_w_accel']  # change in velocity i.e, acceleration
        self.S[1, 1] = self.param['mpc_w_delta_d']  # change in steering angle. weighing matrices (change in controls)
        self.S[2, 2] = self.param['mpc_w_delta_p']

        self.obj = 0  # Objective function
        self.g = []  # constraints vector

    def set_initial_params(self, param):
        '''Set initial parameters related to MPC'''
        self.param = param
        self.dT = param['dT']
        self.N = param['N']
        self.L = param['L']
        self.theta_max, self.v_max = param['theta_max'], param['v_max']
        self.theta_min = -self.theta_max
        self.v_min = -self.v_max
        self.x_min, self.x_max = param['x_min'], param['x_max']
        self.y_min, self.y_max = param['y_min'], param['y_max']
        self.psi_min, self.psi_max = param['psi_min'], param['psi_max']
        self.s_min, self.s_max = param['s_min'], param['s_max']*2
        self.p_min, self.p_max = param['p_min'], param['p_max']
        self.NDP = int(self.param['spline_poly_order']) + 1
        print self.param

    def set_track_data(self, c_x, c_y, c_dx, c_dy, r_x, r_y, l_x, l_y, element_arc_lengths,original_arc_length_total):
        self.center_lut_x, self.center_lut_y = c_x, c_y
        self.center_lut_dx, self.center_lut_dy = c_dx, c_dy
        self.right_lut_x, self.right_lut_y = r_x, r_y
        self.left_lut_x, self.left_lut_y = l_x, l_y
        self.element_arc_lengths = element_arc_lengths
        self.arc_lengths_orig_l = original_arc_length_total

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
        self.g = vertcat(self.g, st - self.P[0:self.n_states])  # initial condition constraints
        for k in range(self.N):
            st = self.X[:, k]
            st_next = self.X[:, k + 1]
            con = self.U[:, k]

            # st[3] = mod(st[3], self.element_arc_lengths[-1])
            # st_next[3]= mod(st_next[3], self.element_arc_lengths[-1])
            dx, dy = self.center_lut_dx(st_next[3]), self.center_lut_dy(st_next[3])
            t_angle = atan2(dy, dx)
            ref_x, ref_y = self.center_lut_x(st_next[3]), self.center_lut_y(st_next[3])
            e_c = sin(t_angle) * (st_next[0] - ref_x) - cos(t_angle) * (st_next[1] - ref_y)
            e_l = -cos(t_angle) * (st_next[0] - ref_x) - sin(t_angle) * (st_next[1] - ref_y)
            error = vertcat(e_c, e_l)

            self.obj = self.obj + mtimes(mtimes(error.T, self.Q), error) + mtimes(
                mtimes((con - self.P[self.n_states+2*self.N+4*self.N_OBST+self.n_controls*k:self.n_states+2*self.N+4*self.N_OBST+self.n_controls*(k+1)]).T,
                       self.R),
                (con - self.P[self.n_states+2*self.N+4*self.N_OBST+self.n_controls*k:self.n_states+2*self.N+4*self.N_OBST+self.n_controls*(k+1)]))
            if k < self.N - 1:
                con_next = self.U[:, k + 1]
                self.obj += mtimes(mtimes((con_next - con).T, self.S), (con_next - con))

            f_value = self.f(st, con)
            st_next_euler = st + (self.dT * f_value)
            self.g = vertcat(self.g, st_next - st_next_euler)  # compute constraints

            # path boundary constraints
            self.g=vertcat(self.g,self.P[self.n_states+2*k]*st_next[0]-self.P[self.n_states+2*k+1]*st_next[1])   #LB<=ax-by<=UB  --represents half space planes

            # print error, self.obj, self.g
            # input('str')

            # Obstacle avoidance constraints
            #should include [pos_x,pos_y,velocity(ob_vel),orientation(ob_theta)]
            # obs_x = 0.5
            # obs_y = 0.5
            # obs_diam = 0.3
            # rob_diam = 0.3
            # self.g = vertcat(self.g,-sqrt((st[0] - obs_x)** 2 + (st[1] - obs_y) ** 2) + (rob_diam / 2 + obs_diam / 2))

    def init_ipopt_solver(self):
        # Optimization variables(States+controls) across the prediction horizon
        OPT_variables = vertcat(reshape(self.X, self.n_states * (self.N + 1), 1),
                                reshape(self.U, self.n_controls * self.N, 1))
        self.opts["ipopt"] = {}
        self.opts["ipopt"]["max_iter"] = 2000
        self.opts["ipopt"]["print_level"] =0
        self.opts["verbose"] = self.param['ipopt_verbose']
        # self.opts["expand"] = True    #Error- 'eval_sx' not defined for BSplineInterpolant
        self.opts["jit"]=True
        self.opts["print_time"] = 0
        self.opts["ipopt"]["acceptable_tol"] = 1e-8
        self.opts["ipopt"]["acceptable_obj_change_tol"] = 1e-6
        self.opts["ipopt"]["fixed_variable_treatment"] = "make_parameter"
        self.opts["ipopt"]["linear_solver"] = "ma57"
        # Nonlinear problem formulation with solver initialization
        self.nlp_prob = {'f': self.obj, 'x': OPT_variables, 'g': self.g, 'p': self.P}
        self.solver = nlpsol('solver', 'ipopt', self.nlp_prob, self.opts)

    def init_constraints(self):
        '''Initialize constraints for states, dynamic model state transitions and control inputs of the system'''
        # for k in range(self.N):
        #     self.lbg[2 * self.n_states * (k + 1):2 * self.n_states * (k + 1) + 4, 0] = np.array([[0, -inf, 0, -inf]])
        #     self.ubg[2 * self.n_states * (k + 1):2 * self.n_states * (k + 1) + 4, 0] = np.array([[inf, 0, inf, 0]])

        self.lbg = np.zeros((self.n_states * (self.N + 1)+self.N, 1))
        self.ubg = np.zeros((self.n_states * (self.N + 1)+self.N, 1))
        self.lbx = np.zeros((self.n_states + (self.n_states + self.n_controls) * self.N, 1))
        self.ubx = np.zeros((self.n_states + (self.n_states + self.n_controls) * self.N, 1))
        # Upper and lower bounds for the state optimization variables

        for k in range(self.N + 1):
            self.lbx[self.n_states * k:self.n_states * (k + 1), 0] = np.array(
                [[self.x_min, self.y_min, self.psi_min, self.s_min]])
            self.ubx[self.n_states * k:self.n_states * (k + 1), 0] = np.array(
                [[self.x_max, self.y_max, self.psi_max, self.s_max]])
        state_count = self.n_states * (self.N + 1)
        # Upper and lower bounds for the control optimization variables
        for k in range(self.N):
            self.lbx[state_count:state_count + self.n_controls, 0] = np.array(
                [[self.v_min, self.theta_min, self.p_min]])  # v and theta lower bound
            self.ubx[state_count:state_count + self.n_controls, 0] = np.array(
                [[self.v_max, self.theta_max, self.p_max]])  # v and theta upper bound
            state_count += self.n_controls



    def init_mpc_start_conditions(self):
        self.u0 = np.zeros((self.N, self.n_controls))
        self.X0 = np.zeros((self.N + 1,self.n_states))

    def get_angle_at_centerline(self,s):
        dx, dy = self.center_lut_dx(s), self.center_lut_dy(s)
        return np.arctan2(dy, dx)

    def get_point_at_centerline(self,s):
        return self.center_lut_x(s), self.center_lut_y(s)

    def get_path_constraints_points(self,prev_soln):
        right_points = np.zeros((self.N, 2))
        left_points = np.zeros((self.N, 2))
        for k in range(1,self.N+1):
            right_points[k-1,:] = [self.right_lut_x(prev_soln[k,3]), self.right_lut_y(prev_soln[k,3])]  # Right boundary
            left_points[k-1,:] = [self.left_lut_x(prev_soln[k,3]), self.left_lut_y(prev_soln[k,3])]  # Left boundary
        return right_points,left_points

    def construct_warm_start_soln(self,initial_state):
        #Construct an initial estimated solution to warm start the optimization problem with valid path constraints
        s_now = initial_state[3]
        initial_state[2] = self.get_angle_at_centerline(s_now)
        self.X0[0,:] = initial_state
        for k in range(1,self.N+1):
            s_next = self.X0[k-1,3] + self.p_initial*self.dT
            psi_next = self.get_angle_at_centerline(s_next)
            x_next,y_next = self.get_point_at_centerline(s_next)
            self.X0[k,:]=[x_next,y_next,psi_next,s_next]

    def filter_estimate(self,initial_s):
        # print self.X0
        if self.X0[0,3] > self.arc_lengths_orig_l:
            # print "Not ADjusted", self.X0
            self.X0[:,3]=self.X0[:,3]-self.arc_lengths_orig_l
            # print "ADjusted",self.X0
            # print initial_s

    def solve(self, initial_state):
        p = np.zeros( self.n_states+2*self.N+4*self.N_OBST+ self.N *self.n_controls)
        if initial_state[3]>self.arc_lengths_orig_l:
            initial_state[3] =initial_state[3] - self.arc_lengths_orig_l
        if self.X0[0,2]-initial_state[2] >np.pi:
            initial_state[2]=initial_state[2]+2*np.pi
        elif self.X0[0, 2] - initial_state[2] < -np.pi:
            initial_state[2] = initial_state[2] - 2 * np.pi


        if not self.WARM_START:
            rospy.loginfo("Warm start started")
            self.construct_warm_start_soln(initial_state)
            rospy.loginfo("Warm start accomplished")
        # print self.X0
        self.filter_estimate(initial_state[3])
        initial_state[3]=self.X0[0,3]
        p[0:self.n_states] = initial_state  # initial condition of the robot posture
        right_points, left_points = self.get_path_constraints_points(self.X0)
        self.publish_boundary_markers(right_points,left_points)

        for k in range(self.N):  #set the reference controls and path boundary conditions to track
            delta_x_path = right_points[k,0]-left_points[k,0]
            delta_y_path = right_points[k,1]-left_points[k,1]
            p[self.n_states+2*k:self.n_states+2*k+2]=[-delta_x_path,delta_y_path]
            up_bound= max(-delta_x_path*right_points[k,0]-delta_y_path*right_points[k,1],-delta_x_path*left_points[k,0]-delta_y_path*left_points[k,1])
            low_bound = min(-delta_x_path*right_points[k,0]-delta_y_path*right_points[k,1],-delta_x_path*left_points[k,0]-delta_y_path*left_points[k,1])
            self.lbg[self.n_states-1+(self.n_states+1)*(k+1),0]=low_bound
            self.ubg[self.n_states - 1 + (self.n_states + 1) * (k + 1),0] = up_bound

            #obstacle parameters

            #Control parameters
            v_ref = self.param['ref_vel']
            p_ref = self.param['p_max']
            theta_ref = 0
            p[self.n_states+2*self.N+4*self.N_OBST+self.n_controls*k:self.n_states+2*self.N+4*self.N_OBST+self.n_controls*(k+1)] = [v_ref,theta_ref, p_ref]

        # Initial value of the optimization variables
        x_init = vertcat(reshape(self.X0.T, self.n_states * (self.N + 1), 1),
                         reshape(self.u0.T, self.n_controls * self.N, 1))
        # Solve using ipopt by giving the initial estimate and parameters as well as bounds for constraints
        sol = self.solver(x0=x_init, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg, p=p)
        # Get state and control solution
        self.X0 = reshape(sol['x'][0:self.n_states * (self.N + 1)], self.n_states, self.N + 1).T  # get soln trajectory
        u = reshape(sol['x'][self.n_states * (self.N + 1):], self.n_controls, self.N).T  # get controls solution
        # Get the first control
        con_first = u[0, :].T
        trajectory = self.X0.full()  # size is (N+1,n_states)
        # full converts casadi data type to python data type(numpy array)
        # print self.X0
        # print u
        # input('str')

        # Shift trajectory and control solution to initialize the next step
        self.X0 = vertcat(self.X0[1:, :], self.X0[self.X0.size1() - 1, :])
        self.u0 = vertcat(u[1:, :], u[u.size1() - 1, :])
        return con_first, trajectory

    def heading(self, yaw):
        q = quaternion_from_euler(0, 0, yaw)
        return Quaternion(*q)

    def publish_boundary_markers(self,right_points,left_points):
        boundary_array = MarkerArray()
        combined_points = np.row_stack((right_points,left_points))
        delta = right_points - left_points
        angles= np.arctan2(delta[:,0],-delta[:,1])

        for i in range(combined_points.shape[0]):
            path_marker = Marker()
            path_marker.header.frame_id = "map"
            path_marker.id = i
            path_marker.type = path_marker.ARROW
            path_marker.action = path_marker.ADD
            path_marker.scale= Vector3(0.25,0.05,0.05)
            path_marker.color = ColorRGBA(0.0,0.0,1.0,0.8)
            path_marker.pose.orientation = self.heading(angles[i%right_points.shape[0]])
            path_marker.pose.position= Point(float(combined_points[i,0]),float(combined_points[i,1]),0.0)
            boundary_array.markers.append(path_marker)
        self.boundary_pub.publish(boundary_array)

########################################################################################################################
########################################################################################################################
class MPCKinematicNode:
    def __init__(self):
        rospy.init_node('mpc_node')
        self.param = {'dT': rospy.get_param('dT', 0.25),
                      'N': rospy.get_param('mpc_steps_N', 18),
                      'L': rospy.get_param('vehicle_L', 0.325),
                      'theta_max': rospy.get_param('mpc_max_steering', 0.523),
                      'v_max': rospy.get_param('max_speed', 6.0), #5
                      'p_min': rospy.get_param('p_min', 0),
                      'p_max': rospy.get_param('p_max', 6.0),
                      'x_min': rospy.get_param('x_min', -200),
                      'x_max': rospy.get_param('x_max', 200),
                      'y_min': rospy.get_param('y_min', -200),
                      'y_max': rospy.get_param('y_max', 200),
                      'psi_min': rospy.get_param('psi_min', -1000),
                      'psi_max': rospy.get_param('psi_max', 1000),
                      's_min': rospy.get_param('s_min', 0),
                      's_max': rospy.get_param('s_max', 200),
                      'ref_vel': rospy.get_param('mpc_ref_vel', 2.5),
                      'mpc_w_cte': rospy.get_param('mpc_w_cte', 1000), #500
                      'mpc_w_epsi': rospy.get_param('mpc_w_epsi', 50),
                      'mpc_w_s': rospy.get_param('mpc_w_s', 0),
                      'mpc_w_lag': rospy.get_param('mpc_w_lag', 1000),  #1000
                      'mpc_w_vel': rospy.get_param('mpc_w_vel', 0.05),
                      'mpc_w_delta': rospy.get_param('mpc_w_delta', 50),
                      'mpc_w_p': rospy.get_param('mpc_w_p', 0.75),  #1
                      'mpc_w_accel': rospy.get_param('mpc_w_accel', 4),
                      'mpc_w_delta_d': rospy.get_param('mpc_w_delta_d', 1300), #600
                      'mpc_w_delta_p': rospy.get_param('mpc_w_delta_p', 0),
                      'spline_poly_order': rospy.get_param('spline_poly_order', 3),
                      'ipopt_verbose': rospy.get_param('ipopt_verbose', True)
                      }

        filename = rospy.get_param('~waypoints_filepath', '')
        self.CENTER_TRACK_FILENAME = rospy.get_param('center_track_filpath', './test/centerline_waypoints.csv')
        self.CENTER_DERIVATIVE_FILENAME = rospy.get_param('center_derivative_filpath',
                                                          './test/center_spline_derivatives.csv')
        self.RIGHT_TRACK_FILENAME = rospy.get_param('right_track_filpath', './test/right_waypoints.csv')
        self.LEFT_TRACK_FILENAME = rospy.get_param('left_track_filpath', './test/left_waypoints.csv')
        self.LOCAL_PATHLENGTH = rospy.get_param('local_path_length', 4.0)
        self.WAYPOINT_FOV = rospy.get_param('waypoints_fov', 1.57)
        self.CONTROLLER_FREQ = rospy.get_param('controller_freq', 20)
        self.GOAL_THRESHOLD = rospy.get_param('goal_threshold', 1.0)
        self.DEBUG_MODE = rospy.get_param('debug_mode', True)
        self.DELAY_MODE = rospy.get_param('delay_mode', True)
        # Topic name related parameters
        pose_topic = rospy.get_param('localized_pose_topic_name', '/pf/viz/inferred_pose')
        cmd_vel_topic = rospy.get_param('cmd_vel_topic_name', '/vesc/high_level/ackermann_cmd_mux/input/nav_0')
        odom_topic = rospy.get_param('odom_topic_name', '/vesc/odom')
        goal_topic = rospy.get_param('goal_topic_name', '/move_base_simple/goal')
        self.car_frame = rospy.get_param('car_frame', 'base_link')

        # Path related variables
        self.path_points = None
        # self.total_path = self.read_waypoints_from_csv(filename)
        self.local_path = Path()
        self.path_from_coeffs = Path()
        self.center_lane = None
        self.center_point_angles =None
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
        self.last_search_index = 0
        self.SEARCH_WINDOW = 100
        self.SEARCH_TOL = 1.0
        self.ARC_LENGTH_MIN_DIST_TOL = 0.02

        # Publishers
        self.ackermann_pub = rospy.Publisher(cmd_vel_topic, AckermannDriveStamped, queue_size=10)
        self.mpc_trajectory_pub = rospy.Publisher('/mpc_trajectory', Path, queue_size=10)
        self.mpc_reference_pub = rospy.Publisher('/mpc_reference', Path, queue_size=10)
        self.mpc_coeffs_path_pub = rospy.Publisher('/mpc_coeffs_path', Path, queue_size=10)
        self.center_path_pub = rospy.Publisher('/center_path', Path, queue_size=100)
        self.right_path_pub = rospy.Publisher('/right_path', Path, queue_size=100)
        self.left_path_pub = rospy.Publisher('/left_path', Path, queue_size=100)
        self.center_tangent_pub = rospy.Publisher('/center_tangent', PoseStamped, queue_size=100)

        # MPC related initializations
        self.mpc = MPC()
        self.mpc.boundary_pub = rospy.Publisher('/boundary_marker', MarkerArray, queue_size=100)
        self.initialize_MPC()
        self.current_pos_x, self.current_pos_y, self.current_yaw, self.current_s = 0.0, 0.0, 0.0, 0.0
        self.current_pose = None
        self.current_vel_odom = 0.0
        self.projected_vel=0.0
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
                                self.element_arc_lengths,self.element_arc_lengths_orig[-1])
        self.mpc.setup_MPC()

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
            if (((dist < self.LOCAL_PATHLENGTH) and (fabs(delta_yaw) < self.WAYPOINT_FOV)) or (dist < 1.5)):
                tempPose = PoseStamped()
                tempPose.header = self.local_path.header
                tempPose.pose.position.x = self.total_path.poses[i].pose.position.x
                tempPose.pose.position.y = self.total_path.poses[i].pose.position.y
                tempPose.pose.orientation.w = 1.0
                self.local_path.poses.append(tempPose)

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

    def read_waypoints_from_csv(self, filename):
        '''read waypoints from given csv file and return the data in the form of nav_msgs::Path'''
        path = Path()
        path.header = self.create_header('map')
        if filename == '':
            raise ValueError('No any file path for waypoints file')
        with open(filename) as f:
            path_points = [tuple(line) for line in csv.reader(f, delimiter=',')]
        self.path_points = np.array([(float(point[0]), float(point[1])) for point in path_points])
        skip = 8
        for idx, point in enumerate(self.path_points):
            if idx % skip == 0:
                header = self.create_header('map')
                waypoint = Pose(Point(float(point[0]), float(point[1]), 0), self.heading(0.0))
                path.poses.append(PoseStamped(header, waypoint))
        return path

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
                self.mpc.WARM_START=False
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

    def publish_path_from_coeffs(self, coeffs):
        # Visualize path derived from fitted spline to given waypoints in the path
        x_coords = np.linspace(-1.0, self.LOCAL_PATHLENGTH * 2, int((self.LOCAL_PATHLENGTH * 2 + 1) / 0.1))
        self.path_from_coeffs.header = self.create_header(self.car_frame)
        self.path_from_coeffs.poses = []
        for x in x_coords:
            tempPose = PoseStamped()
            tempPose.header = self.path_from_coeffs.header
            tempPose.pose.position.x = x
            tempPose.pose.position.y = np.polyval(self.mpc.coeffs, x)
            tempPose.pose.orientation.w = 1.0
            self.path_from_coeffs.poses.append(tempPose)
        self.mpc_coeffs_path_pub.publish(self.path_from_coeffs)

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
        print np.where(consecutive_diff==0)[0]
        dists_cum = np.insert(dists_cum, 0, 0.0)
        return dists_cum

    def preprocess_track_data(self):
        center_lane = self.read_waypoints_array_from_csv(self.CENTER_TRACK_FILENAME)
        center_derivative_data = self.read_waypoints_array_from_csv(self.CENTER_DERIVATIVE_FILENAME)
        right_lane = self.read_waypoints_array_from_csv(self.RIGHT_TRACK_FILENAME)
        left_lane = self.read_waypoints_array_from_csv(self.LEFT_TRACK_FILENAME)

        self.center_lane = np.row_stack((center_lane,center_lane[1:int(center_lane.shape[0]/2),:]))
        right_lane = np.row_stack((right_lane, right_lane[1:int(center_lane.shape[0] / 2), :]))
        left_lane = np.row_stack((left_lane, left_lane[1:int(center_lane.shape[0] / 2), :]))
        center_derivative_data = np.row_stack((center_derivative_data, center_derivative_data[1:int(center_lane.shape[0] / 2), :]))
        # print self.center_lane.shape,right_lane.shape,left_lane.shape,center_derivative_data

        # Interpolate center line upto desired resolution
        self.element_arc_lengths_orig = self.get_arc_lengths(center_lane)
        self.element_arc_lengths = self.get_arc_lengths(self.center_lane)
        print self.element_arc_lengths
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
        return current_s, nearest_index

    def controlLoopCB(self, event):
        '''Control loop for car MPC'''
        if self.goal_received and not self.goal_reached:
            control_loop_start_time = time.time()
            # Update system states: X=[x, y, psi]
            px = self.current_pos_x
            py = self.current_pos_y
            car_pos = np.array([self.current_pos_x, self.current_pos_y])
            psi = self.current_yaw

            # Update system inputs: U=[speed(v), steering]
            v = self.current_vel_odom
            steering = self.steering_angle  # radian
            dt = 1.0 / self.CONTROLLER_FREQ
            L = self.mpc.L

            current_s, near_idx = self.find_current_arc_length(car_pos)

            if self.DELAY_MODE:
                dt_lag=3* dt
                px = px+ v*np.cos(psi)*dt_lag
                py = py+ v*np.sin(psi)*dt_lag
                psi= psi+ (v/L)*tan(steering)*dt_lag
                current_s = current_s+self.projected_vel*dt_lag
            current_state = np.array([px, py, psi, current_s])

            centerPose = PoseStamped()
            centerPose.header = self.create_header('map')
            centerPose.pose.position.x = float(self.center_lane[near_idx, 0])
            centerPose.pose.position.y = float(self.center_lane[near_idx, 1])
            centerPose.pose.orientation = self.heading(self.center_point_angles[near_idx])
            self.center_tangent_pub.publish(centerPose)

            print(car_pos,v,near_idx,current_s,psi)

            # Solve MPC Problem
            mpc_time = time.time()
            first_control, trajectory = self.mpc.solve(current_state)
            mpc_compute_time = time.time() - mpc_time
            print("Control loop time mpc=:", mpc_compute_time)

            # MPC result (all described in car frame)
            speed = float(first_control[0])  # speed
            steering = float(first_control[1])  # radian
            self.projected_vel = float(first_control[2])
            if not self.mpc.WARM_START:
                speed, steering = 0, 0
                self.mpc.WARM_START= True
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
            print("Control loop time4=:", time.time() - control_loop_start_time)

            # publish the mpc trajectory
            self.mpc_trajectory_pub.publish(mpc_traj)
            total_time = time.time() - control_loop_start_time
            if self.DEBUG_MODE:
                print("DEBUG")
                print("psi: ", psi)
                print("V: ", v)
                print("coeffs: ", self.mpc.coeffs)
                print("_steering:", steering)
                print("_speed: ", speed)
                print("Control loop time=:", total_time)

            self.current_time += 1.0 / self.CONTROLLER_FREQ
            # self.cte_plot.append(cte)
            self.t_plot.append(self.current_time)
            self.v_plot.append(speed)
            self.steering_plot.append(np.rad2deg(steering))
            self.time_plot.append(mpc_compute_time * 1000)

            # input('str')

        else:
            steering = 0.0
            speed = 0.0

        # publish cmd
        ackermann_cmd = AckermannDriveStamped()
        ackermann_cmd.header = self.create_header(self.car_frame)
        ackermann_cmd.drive.steering_angle = steering
        ackermann_cmd.drive.speed = speed
        # ackermann_cmd.drive.acceleration = throttle
        self.ackermann_pub.publish(ackermann_cmd)

    def plot_data(self):
        plt.figure(1)
        plt.subplot(411)
        plt.step(self.t_plot, self.v_plot, 'k', linewidth=1.5)
        # plt.ylim(-0.2, 0.8)
        plt.ylabel('v m/s')
        plt.subplot(412)
        plt.step(self.t_plot, self.steering_plot, 'r', linewidth=1.5)
        # plt.ylim(-0.5, 1.0)
        plt.ylabel('steering angle(degrees)')
        # plt.subplot(413)
        # plt.step(self.t_plot, self.cte_plot, 'g', linewidth=1.5)
        # # plt.ylim(-0.5, 1.0)
        # plt.ylabel('cte in m')
        plt.subplot(414)
        plt.step(self.t_plot, self.time_plot, 'b', linewidth=1.5)
        plt.ylim(0.0, 70)
        plt.ylabel('mpc_compute_time in ms')
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

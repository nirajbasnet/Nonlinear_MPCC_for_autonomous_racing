#!/usr/bin/env python
from casadi import *
import numpy as np

import rospy
from geometry_msgs.msg import Point, Quaternion, Vector3
from std_msgs.msg import ColorRGBA
from tf.transformations import quaternion_from_euler
from visualization_msgs.msg import Marker, MarkerArray


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
        self.N_OBST = 1

        self.center_lut_x, self.center_lut_y = None, None
        self.center_lut_dx, self.center_lut_dy = None, None
        self.right_lut_x, self.right_lut_y = None, None
        self.left_lut_x, self.left_lut_y = None, None
        self.element_arc_lengths = None
        self.arc_lengths_orig_l = None
        self.WARM_START = False
        self.INTEGRATION_MODE = "Euler"  # RK4 and RK3 method are the other two choices
        self.p_initial = 2.0  # projected centerline vel can set to desired value for initial estimation
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

        self.P = MX.sym('P', self.n_states + 2 * self.N + 4 * self.N_OBST + self.N * self.n_controls)
        # parameters (which include the nth degree polynomial coefficients, initial state and the reference along the predicted trajectory (reference states and reference controls))

        self.X = MX.sym('X', self.n_states, (self.N + 1))
        # A vector that represents the states over the optimization problem

        self.Q = MX.zeros(2, 2)
        self.Q[0, 0] = self.param['mpc_w_cte']  # cross track error
        self.Q[1, 1] = self.param['mpc_w_lag']  # lag error

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
        self.p_initial = self.v_max
        self.theta_min = -self.theta_max
        self.v_min = -self.v_max
        self.x_min, self.x_max = param['x_min'], param['x_max']
        self.y_min, self.y_max = param['y_min'], param['y_max']
        self.psi_min, self.psi_max = param['psi_min'], param['psi_max']
        self.s_min, self.s_max = param['s_min'], param['s_max']
        self.p_min, self.p_max = param['p_min'], param['p_max']
        self.INTEGRATION_MODE = param['INTEGRATION_MODE']
        print self.param

    def set_track_data(self, c_x, c_y, c_dx, c_dy, r_x, r_y, l_x, l_y, element_arc_lengths, original_arc_length_total):
        self.center_lut_x, self.center_lut_y = c_x, c_y
        self.center_lut_dx, self.center_lut_dy = c_dx, c_dy
        self.right_lut_x, self.right_lut_y = r_x, r_y
        self.left_lut_x, self.left_lut_y = l_x, l_y
        self.element_arc_lengths = element_arc_lengths
        self.arc_lengths_orig_l = original_arc_length_total

    def compute_optimization_cost(self):
        st = self.X[:, 0]  # initial state
        self.g = vertcat(self.g, st - self.P[0:self.n_states])  # initial condition constraints
        for k in range(self.N):
            st = self.X[:, k]
            st_next = self.X[:, k + 1]
            con = self.U[:, k]
            dx, dy = self.center_lut_dx(st_next[3]), self.center_lut_dy(st_next[3])
            t_angle = atan2(dy, dx)
            ref_x, ref_y = self.center_lut_x(st_next[3]), self.center_lut_y(st_next[3])
            e_c = sin(t_angle) * (st_next[0] - ref_x) - cos(t_angle) * (st_next[1] - ref_y)
            e_l = -cos(t_angle) * (st_next[0] - ref_x) - sin(t_angle) * (st_next[1] - ref_y)
            error = vertcat(e_c, e_l)

            con_start_idx = self.n_states + 2 * self.N + 4 * self.N_OBST + self.n_controls * k
            con_end_idx = self.n_states + 2 * self.N + 4 * self.N_OBST + self.n_controls * (k + 1)

            self.obj = self.obj + mtimes(mtimes(error.T, self.Q), error) + \
                       mtimes(mtimes((con - self.P[con_start_idx:con_end_idx]).T,self.R), (con - self.P[con_start_idx:con_end_idx]))
            if k < self.N - 1:
                con_next = self.U[:, k + 1]
                self.obj += mtimes(mtimes((con_next - con).T, self.S),(con_next - con))

            k1 = self.f(st, con)
            if self.INTEGRATION_MODE == "Euler":
                st_next_euler = st + (self.dT * k1)
            elif self.INTEGRATION_MODE == "RK3":
                k2 = self.f(st + self.dT / 2 * k1, con)
                k3 = self.f(st + self.dT * (2 * k2 - k1), con)
                st_next_euler = st + self.dT / 6 * (k1 + 4 * k2 + k3)
            elif self.INTEGRATION_MODE == "RK4":
                k2 = self.f(st + self.dT / 2 * k1, con)
                k3 = self.f(st + self.dT / 2 * k2, con)
                k4 = self.f(st + self.dT * k3, con)
                st_next_euler = st + self.dT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

            self.g = vertcat(self.g, st_next - st_next_euler)  # compute constraints

            # path boundary constraints
            self.g = vertcat(self.g,
                             self.P[self.n_states + 2 * k] * st_next[0] - self.P[self.n_states + 2 * k + 1] * st_next[
                                 1])  # LB<=ax-by<=UB  --represents half space planes

            # Obstacle avoidance constraints
            # should include [pos_x,pos_y,velocity(ob_vel),orientation(ob_theta)]
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
        self.opts["ipopt"]["print_level"] = 0
        self.opts["verbose"] = self.param['ipopt_verbose']
        self.opts["jit"] = True
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
        self.lbg = np.zeros((self.n_states * (self.N + 1) + self.N, 1))
        self.ubg = np.zeros((self.n_states * (self.N + 1) + self.N, 1))
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
        self.X0 = np.zeros((self.N + 1, self.n_states))

    def get_angle_at_centerline(self, s):
        dx, dy = self.center_lut_dx(s), self.center_lut_dy(s)
        return np.arctan2(dy, dx)

    def get_point_at_centerline(self, s):
        return self.center_lut_x(s), self.center_lut_y(s)

    def get_path_constraints_points(self, prev_soln):
        right_points = np.zeros((self.N, 2))
        left_points = np.zeros((self.N, 2))
        for k in range(1, self.N + 1):
            right_points[k - 1, :] = [self.right_lut_x(prev_soln[k, 3]),
                                      self.right_lut_y(prev_soln[k, 3])]  # Right boundary
            left_points[k - 1, :] = [self.left_lut_x(prev_soln[k, 3]),
                                     self.left_lut_y(prev_soln[k, 3])]  # Left boundary
        return right_points, left_points

    def construct_warm_start_soln(self, initial_state):
        # Construct an initial estimated solution to warm start the optimization problem with valid path constraints
        if initial_state[3] >= self.arc_lengths_orig_l:
            initial_state[3] -= self.arc_lengths_orig_l
        initial_state[2] = self.get_angle_at_centerline(initial_state[3])
        self.X0[0, :] = initial_state
        for k in range(1, self.N + 1):
            s_next = self.X0[k - 1, 3] + self.p_initial * self.dT
            psi_next = self.get_angle_at_centerline(s_next)
            x_next, y_next = self.get_point_at_centerline(s_next)
            self.X0[k, :] = [x_next, y_next, psi_next, s_next]

    def filter_estimate(self, initial_arc_pos):
        if (self.X0[0, 3] >= self.arc_lengths_orig_l) and (
                (initial_arc_pos >= self.arc_lengths_orig_l) or (initial_arc_pos <= 5)):
            self.X0[:, 3] = self.X0[:, 3] - self.arc_lengths_orig_l
        if initial_arc_pos >= self.arc_lengths_orig_l:
            initial_arc_pos -= self.arc_lengths_orig_l
        return initial_arc_pos

    def solve(self, initial_state):
        p = np.zeros(self.n_states + 2 * self.N + 4 * self.N_OBST + self.N * self.n_controls)
        delta_yaw = self.X0[1, 2] - initial_state[2]
        if abs(delta_yaw) >= np.pi:
            new_val_ceil = initial_state[2] + np.ceil(delta_yaw / (2 * np.pi)) * (2 * np.pi)
            new_val_floor = initial_state[2] + np.floor(delta_yaw / (2 * np.pi)) * (2 * np.pi)
            if abs(new_val_ceil - self.X0[1, 2]) < abs(new_val_floor - self.X0[1, 2]):
                initial_state[2] = new_val_ceil
            else:
                initial_state[2] = new_val_floor
        if not self.WARM_START:
            rospy.loginfo("Warm start started")
            self.construct_warm_start_soln(initial_state)
            rospy.loginfo("Warm start accomplished")
        # print "init1=", initial_state
        initial_state[3] = self.filter_estimate(initial_state[3])
        # print "init2=",initial_state
        # print self.X0
        p[0:self.n_states] = initial_state  # initial condition of the robot posture
        right_points, left_points = self.get_path_constraints_points(self.X0)
        self.publish_boundary_markers(right_points, left_points)

        for k in range(self.N):  # set the reference controls and path boundary conditions to track
            delta_x_path = right_points[k, 0] - left_points[k, 0]
            delta_y_path = right_points[k, 1] - left_points[k, 1]
            p[self.n_states + 2 * k:self.n_states + 2 * k + 2] = [-delta_x_path, delta_y_path]
            up_bound = max(-delta_x_path * right_points[k, 0] - delta_y_path * right_points[k, 1],
                           -delta_x_path * left_points[k, 0] - delta_y_path * left_points[k, 1])
            low_bound = min(-delta_x_path * right_points[k, 0] - delta_y_path * right_points[k, 1],
                            -delta_x_path * left_points[k, 0] - delta_y_path * left_points[k, 1])
            self.lbg[self.n_states - 1 + (self.n_states + 1) * (k + 1), 0] = low_bound
            self.ubg[self.n_states - 1 + (self.n_states + 1) * (k + 1), 0] = up_bound

            # obstacle parameters

            # Control parameters
            v_ref = self.param['ref_vel']
            p_ref = self.param['p_max']
            theta_ref = 0
            p[
            self.n_states + 2 * self.N + 4 * self.N_OBST + self.n_controls * k:self.n_states + 2 * self.N + 4 * self.N_OBST + self.n_controls * (
                    k + 1)] = [v_ref, theta_ref, p_ref]

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
        inputs = u.full()
        # full converts casadi data type to python data type(numpy array)
        # Shift trajectory and control solution to initialize the next step
        self.X0 = vertcat(self.X0[1:, :], self.X0[self.X0.size1() - 1, :])
        self.u0 = vertcat(u[1:, :], u[u.size1() - 1, :])
        return con_first, trajectory, inputs

    def heading(self, yaw):
        q = quaternion_from_euler(0, 0, yaw)
        return Quaternion(*q)

    def publish_boundary_markers(self, right_points, left_points):
        boundary_array = MarkerArray()
        combined_points = np.row_stack((right_points, left_points))
        delta = right_points - left_points
        angles = np.arctan2(delta[:, 0], -delta[:, 1])
        for i in range(combined_points.shape[0]):
            path_marker = Marker()
            path_marker.header.frame_id = "map"
            path_marker.id = i
            path_marker.type = path_marker.ARROW
            path_marker.action = path_marker.ADD
            path_marker.scale = Vector3(0.25, 0.05, 0.05)
            path_marker.color = ColorRGBA(0.0, 0.0, 1.0, 0.8)
            path_marker.pose.orientation = self.heading(angles[i % right_points.shape[0]])
            path_marker.pose.position = Point(float(combined_points[i, 0]), float(combined_points[i, 1]), 0.0)
            boundary_array.markers.append(path_marker)
        self.boundary_pub.publish(boundary_array)


########################################################################################################################
########################################################################################################################

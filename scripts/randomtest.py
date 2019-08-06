from casadi import *
import matplotlib.pyplot as plt

import time
import numpy as np

T = 0.1
N = 50
rob_diam = 0.3

v_max = 0.6
v_min = -v_max
omega_max = pi / 4
omega_min = -omega_max

x = SX.sym('x')
y = SX.sym('y')
theta = SX.sym('theta')
states = vertcat(x, y, theta)
n_states = states.size1()

v = SX.sym('v')
omega = SX.sym('omega')
controls = vertcat(v, omega)
n_controls = controls.size1()
rhs = vertcat(v * cos(theta), v * sin(theta), omega)

f = Function('f', [states, controls], [rhs])  # nonlinear mapping function f(x,u)
U = SX.sym('U', n_controls, N)

P = SX.sym('P', n_states + N * (n_states + n_controls))
# parameters (which include the initial state and the reference along the predicted trajectory (reference states and reference controls))
print(P.size())
X = SX.sym('X', n_states, (N + 1))
# A vector that represents the states over the optimization problem.

obj = 0  # Objective function
g = []  # constraints vector

Q = SX.zeros(3, 3)
Q[0, 0] = 2
Q[1, 1] = 1
Q[2, 2] = 0.5  # weighing matrices (states)

R = SX.zeros(2, 2)
R[0, 0] = 0.5
R[1, 1] = 0.05  # weighing matrices (controls)

S = SX.zeros(5, 5)
S[0, 0] = 2
S[1, 1] = 1
S[2, 2] = 0.5  # weighing matrices (states)
S[3, 3] = 0.5  # weighing matrices (states)
S[4, 4] = 0.05  # weighing matrices (states)



st = X[:, 0]  # initial state
up_time = time.time()
g = vertcat(g, st - P[0:3])  # initial condition constraints

for k in range(N):
    st = X[:, k]
    # print(st)
    con = U[:, k]
    stt=vertcat(st,con)
    # print(con)
    # obj = obj+(st-P(4:6))'*Q*(st-P(4:6)) + con'*R*con; % calculate obj
    # obj = obj + mtimes(mtimes((stt - P[3 + 5 * k:7 + 5 * k + 1]).T, S), (stt - P[3 + 5 * k:7 + 5 * k + 1]))


    obj = obj +  mtimes(mtimes((con - P[6 + 5 * k:7 + 5 * k + 1]).T, R), (con - P[6 + 5 * k:7 + 5 * k + 1]))

    # obj = obj + mtimes(mtimes((st - P[3 + 5 * k:5 + 5 * k + 1]).T, Q), (st - P[3 + 5 * k:5 + 5 * k + 1])) + mtimes(
    #     mtimes((con - P[6 + 5 * k:7 + 5 * k + 1]).T, R), (con - P[6 + 5 * k:7 + 5 * k + 1]))  # % calculate obj
    # #
    # obj = obj + Q[0, 0] * (st[0] - P[3 + 5 * k])*(st[0] - P[3 + 5 * k]) \
    #           + Q[1, 1] * (st[1] - P[4 + 5 * k])*(st[1] - P[4 + 5 * k]) \
    #           + Q[2, 2] * (st[2] - P[5 + 5 * k])*(st[2] - P[5 + 5 * k]) \
    #           + R[0, 0] * (con[0] - P[6 + 5 * k])*(con[0] - P[6 + 5 * k]) \
    #           + R[1, 1] * (con[1] - P[7 + 5 * k])*(con[1] - P[7 + 5 * k])
    # the number 5 is (n_states+n_controls)
    # st_next = X[:, k + 1]
    # f_value = f(st, con)
    # st_next_euler = st + (T * f_value)
    # g = vertcat(g, st_next - st_next_euler)  # compute constraints
    # print(st_next)
    # print(f_value)
    # print(st_next_euler)
    # print(g)

# make the decision variable one column  vector
OPT_variables = vertcat(reshape(X, 3 * (N + 1), 1), reshape(U, 2 * N, 1))

nlp_prob = {'f': obj, 'x': OPT_variables, 'g': g, 'p': P}
print("up2_time=", time.time() - up_time)
opts = {}
opts["ipopt"] = {}
opts["ipopt"]["max_iter"] = 2000
opts["ipopt"]["print_level"] = 0
opts["verbose"] = False
opts["verbose_init"] = False

opts["print_time"] = 0
opts["ipopt"]["acceptable_tol"] = 1e-8
opts["ipopt"]["acceptable_obj_change_tol"] = 1e-6
print("up3_time=", time.time() - up_time)
solver = nlpsol('solver', 'ipopt', nlp_prob, opts)
print("up_time=", time.time() - up_time)
input('str')
args = {}

lbg = np.zeros((n_states * (N + 1), 1))
ubg = np.zeros((n_states * (N + 1), 1))
lbx = np.zeros((n_states + (n_states + n_controls) * N, 1))
ubx = np.zeros((n_states + (n_states + n_controls) * N, 1))

for k in range(N + 1):
    lbx[n_states * k:n_states * (k + 1), 0] = np.array([[-20, -2, -3.14]])
    ubx[n_states * k:n_states * (k + 1), 0] = np.array([[20, 2, 3.14]])

state_count = n_states * (N + 1)
for k in range(N):
    lbx[state_count:state_count + n_controls, 0] = np.array([[v_min, omega_min]])  # v and omega lower bound
    ubx[state_count:state_count + n_controls, 0] = np.array([[v_max, omega_max]])  # v and omega upper bound
    state_count += n_controls
# ALL OF THE ABOVE IS JUST A PROBLEM SET UP
#
# print(lbg)
# print(lbx)
# print(ubx)

# THE SIMULATION LOOP SHOULD START FROM HERE
sim_tim = 30  # Maximum simulation time
total_iterations = int(sim_tim / T)
t0 = 0
x0 = vertcat(0, 0, 0.0)  # initial condition.
xs = vertcat(1.5, 1.5, 0.0)  # Reference posture.

xx = SX.zeros(1, N)
t = np.zeros(total_iterations + 1)
# xx = x0 # xx contains the history of states
t[0] = t0
u_init = vertcat(0, 0)
u0 = transpose(repmat(u_init, 1, N))
# u0 = SX.zeros(N, 2)  # two control inputs for each robot
X0 = transpose(repmat(x0, 1, N + 1))  # initialization of the states decision variables

# Start MPC
mpciter = 0
xx1 = []
u_cl = np.zeros((total_iterations + 1, 2))

# the main simulaton loop... it works as long as the error is greater
# than 10^-6 and the number of mpc steps is less than its maximum
# value.


p = np.zeros((n_states + N * (n_states + n_controls), 1))
loop_start_time = time.time()
while (mpciter < total_iterations):  # new - condition for ending the loop
    main_loop = time.time()
    obj = 0  # Objective function
    g = []  # constraints vector
    st = X[:, 0]  # initial state
    g = vertcat(g, st - P[0:3])  # initial condition constraints

    for k in range(N):
        st = X[:, k]
        # print(st)
        con = U[:, k]
        # print(con)
        # obj = obj+(st-P(4:6))'*Q*(st-P(4:6)) + con'*R*con; % calculate obj
        obj = obj + mtimes(mtimes((st - P[3 + 5 * k:5 + 5 * k + 1]).T, Q), (st - P[3 + 5 * k:5 + 5 * k + 1])) + mtimes(
            mtimes((con - P[6 + 5 * k:7 + 5 * k + 1]).T, R), (con - P[6 + 5 * k:7 + 5 * k + 1]))  # % calculate obj
        # the number 5 is (n_states+n_controls)
        # print(obj)
        st_next = X[:, k + 1]
        f_value = f(st, con)
        st_next_euler = st + (T * f_value)
        g = vertcat(g, st_next - st_next_euler)  # compute constraints
        # print(st_next)
        # print(f_value)
        # print(st_next_euler)
        # print(g)
    # print(time.time()-main_loop)

    # make the decision variable one column  vector
    OPT_variables = vertcat(reshape(X, 3 * (N + 1), 1), reshape(U, 2 * N, 1))

    nlp_prob = {'f': obj, 'x': OPT_variables, 'g': g, 'p': P}
    # print(time.time() - main_loop)

    solver = nlpsol('solver', 'ipopt', nlp_prob, opts)
    print(time.time() - main_loop)
    # input('str')
    current_time = mpciter * T  # new - get the current time
    p[0:3] = x0  # initial condition of the robot posture
    for k in range(N):  # new - set the reference to track
        t_predict = current_time + k * T  # predicted time instant
        x_ref = 0.5 * t_predict
        y_ref = 1
        theta_ref = 0
        u_ref = 0.5
        omega_ref = 0
        if x_ref >= 12:  # the trajectory end is reached
            x_ref = 12
            y_ref = 1
            theta_ref = 0
            u_ref = 0
            omega_ref = 0
        p[3 + 5 * k:5 + 5 * k + 1, 0] = [x_ref, y_ref, theta_ref]
        p[6 + 5 * k:7 + 5 * k + 1, 0] = [u_ref, omega_ref]
    # print(p)

    #
    #     # initial value of the optimization variables
    x_init = vertcat(reshape(X0.T, 3 * (N + 1), 1), reshape(u0.T, 2 * N, 1))
    sol = solver(x0=x_init, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=p)

    u = reshape(sol['x'][3 * (N + 1):], 2, N).T
    # xx1(:,1:3,mpciter+1)= reshape(full(sol.x(1:3*(N+1)))',3,N+1)'; % get solution TRAJECTORY
    u_cl[mpciter, :] = u[0, :]
    t[mpciter + 1] = t0
    #     % Apply the control and shift the solution
    st = x0
    con_first = u[0, :].T
    function_value = f(st, con_first)
    st = st + T * function_value
    x0 = st.full()
    t0 = t0 + T
    u0 = vertcat(u[1:, :], u[u.size1() - 1, :])
    # print(x0)
    # print(t0)
    # print(u0)

    #     xx(:,mpciter+2) = x0;
    X0 = reshape(sol['x'][0:3 * (N + 1)], 3, N + 1).T  # get solution TRAJECTORY
    # print(X0)
    # print(X0.size())
    # Shift trajectory to initialize the next step
    X0 = vertcat(X0[1:, :], X0[X0.size1() - 1, :])
    # print(X0)
    mpciter = mpciter + 1
print(mpciter)
main_loop_time = time.time() - loop_start_time
average_mpc_time = main_loop_time / (mpciter + 1)
print("average time=", average_mpc_time)
# Draw_PC_tracking_v1 (t,xx,xx1,u_cl,xs,N,rob_diam)

plt.figure(1)
plt.subplot(211)
plt.step(t, u_cl[:, 0], 'k', linewidth=1.5)
plt.ylim(-0.2, 0.8)
plt.ylabel('v rad/s')
plt.subplot(212)
plt.step(t, u_cl[:, 1], 'r', linewidth=1.5)
plt.ylim(-0.5, 1.0)
plt.ylabel('omega rad/s')
plt.show()

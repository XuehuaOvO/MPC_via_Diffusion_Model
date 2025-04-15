import casadi as ca
import numpy as np
import control
import torch
import os
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool, Manager, Array
import multiprocessing
import scipy.linalg
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, AcadosSimSolver

MAX_CORE_CPU = 1  # 16
FOLDER_PATH = '/root/cartpoleDiff/cart_pole_diffusion_based_on_MPD/scripts/mpc_data_collecting/Acrobots/figure/new'

##### Acrobot Parameters (Gym) #####
LINK_LENGTH_1 = 1.0  # [m]
LINK_LENGTH_2 = 1.0  # [m]
LINK_MASS_1 = 1.0  #: [kg] mass of link 1
LINK_MASS_2 = 1.0  #: [kg] mass of link 2
LINK_COM_POS_1 = 0.5  #: [m] position of the center of mass of link 1
LINK_COM_POS_2 = 0.5  #: [m] position of the center of mass of link 2
LINK_MOI = 1.0  #: moments of inertia for both links
G = 9.81 # [m/s^2]
U_BOUND = 10


##### MPC parameters #####
CONTROL_STEPS = 300

# (for 1 time solving)
N = 128 # mpc prediction horizon
TS = 0.01
TF = N*TS

NUM_X = 4 # theta1, theta2, theta_1_dot, theta_2_dot
NUM_U = 1 # tau
IDX_THETA1_INI = 0
IDX_THETA2_INI = 1

X_GUESS = [-np.pi, np.pi] # [-np.pi, np.pi]
THETHA_1_GUESS_RANGE = np.linspace(np.pi, np.pi, 2)
THETHA_2_GUESS_RANGE = np.linspace(-2*np.pi, 2*np.pi, 2)
U_GUESS = [-10,10]

##### cost function weights #####
"Q = np.diag([0.1, 10, 10, 0.1]), R = 0.1, P = np.diag([1, 1, 1, 1])"
Q = np.diag([100, 100, 1, 1]) # np.diag([0.01, 0.01, 1, 1, 10, 10])   np.diag([1, 1, 10, 10])
Q_E = np.diag([1000, 1000, 10, 10])  # np.diag([0.01, 0.01, 10, 10, 1000, 1000])   np.diag([10, 10, 1000, 1000])
R = 0.1
# Q, R --> W for Acado ocp

# intermediate weights
W = scipy.linalg.block_diag(Q, R) 
# terminal weights
W_TERMINAL = Q_E # 6 states
# initial weights
# W_INI = scipy.linalg.block_diag(Q, R) 

# reference state
X_REF = np.array([0, 0, 0, 0, 0]) # 4 + 1
X_REF_TERMINAL = np.array([0, 0, 0, 0])  # 4s
# X_REF_INI= np.array([np.pi, 0, 0, 0, 0, 0, 0, 0]) # 6 states + 1 u

# initial data range
NUM_INITIAL_THETA1 = 1
Theta1_INITIAL_RANGE = np.linspace(0,0, NUM_INITIAL_THETA1)  # np.linspace(-np.pi/2,np.pi/2, NUM_INITIAL_THETA1) 

NUM_INITIAL_THETA2 = 3
Theta2_INITIAL_RANGE = np.linspace(-np.pi/4,np.pi/4, NUM_INITIAL_THETA2)  # np.linspace(-np.pi/2,np.pi/2, NUM_INITIAL_THETA2) 

# rng_theta1 = np.concatenate([Theta1_INITIAL_RANGE_1, Theta1_INITIAL_RANGE_2])
# rng_theta2 = np.concatenate([Theta2_INITIAL_RANGE_1, Theta2_INITIAL_RANGE_2])

rng0 = []
for idx1 in Theta1_INITIAL_RANGE:
    for idx2 in Theta2_INITIAL_RANGE:
        rng0.append([idx1,idx2])
rng0 = np.array(rng0)
num_datagroup = len(rng0)
print(f'rng0 -- {rng0.shape}')

# initial guess
guess_rng0 = []
for g_idx1 in THETHA_1_GUESS_RANGE:
    for g_idx2 in THETHA_2_GUESS_RANGE:
        guess_rng0.append([g_idx1,g_idx2])
guess_rng0= np.array(guess_rng0)
num_guessgroup = len(guess_rng0)
print(f'guess_rng0 -- {guess_rng0.shape}')

INITIAL_GUESS_NUM = 2
initial_guess_x = guess_rng0
initial_guess_u = U_GUESS
num_u_guessgroup = len(initial_guess_u)

# Theta star
PI_UNDER_2 = 2/np.pi

def Theta1ToThetaStar1(theta1):
    return np.pi - ((theta1)**2/np.pi)

def Theta2ToThetaStar2(theta2):
    return (theta2-np.pi)**2/-np.pi + np.pi


########## dynamics of Acrobot ##########
def Acrobot_dynamic_Casadi(x, u) -> ca.vertcat:
   
   "x[0] theta_1, x[1] theta_2, x[2] theta_1_dot, x[3] theta_2_dot"
   
   # mass matrix elements
   m11 = LINK_MOI + LINK_MOI + LINK_MASS_2*LINK_LENGTH_1**2 + 2*LINK_MASS_2*LINK_LENGTH_1*LINK_COM_POS_2*np.sin(x[1])
   m12 = LINK_MOI + LINK_MASS_2*LINK_LENGTH_1*LINK_COM_POS_2*np.sin(x[1])
   m21 = LINK_MOI + LINK_MASS_2*LINK_LENGTH_1*LINK_COM_POS_2*np.sin(x[1])
   m22 = LINK_MOI 

   Mass = ca.SX([[m11, m12],
              [m21, m22]])
   Mass_inv = ca.inv(Mass)

   # Coriolis matrix elements
   c11 = -2*LINK_MASS_2*LINK_LENGTH_1*LINK_COM_POS_2*np.sin(x[1])*x[3]
   c12 = -LINK_MASS_2*LINK_LENGTH_1*LINK_COM_POS_2*x[3]
   c21 = LINK_MASS_2*LINK_LENGTH_1*LINK_COM_POS_2*x[2]
   c22 = 0

   Cor = ca.SX([[c11, c12],
              [c21, c22]])

   # gravitational torque matrix elements
   taug1 = -LINK_MASS_1*G*LINK_COM_POS_1*np.sin(x[0]) - LINK_MASS_2*G*(LINK_LENGTH_1*np.sin(x[0]) + LINK_COM_POS_2*np.sin(x[0] + x[1]))
   taug2 = -LINK_MASS_2*G*LINK_COM_POS_2*np.sin(x[0] + x[1])

   Taug = ca.SX([[taug1],
                 [taug2]])
   
   # B matrix
   B_matrix = ca.SX([[0],
                     [1]])
   
   # calculate theta_ddot
   theta_dot = ca.SX([x[2],
                      x[3]])
   
   theta_ddot = Mass_inv*(Taug + B_matrix*u - Cor*theta_dot)


   return ca.vertcat(
        x[2], # theta_1_dot

        x[3], # theta_2_dot

        theta_ddot[0], # theta_1_ddot

        theta_ddot[1], # theta_2_ddot
    )

########## define Acado Model ##########
def Acrobot_Acado_model():
   
   theta_1 = ca.SX.sym('theta_1')
   theta_2 = ca.SX.sym('theta_2')
   dtheta_1 = ca.SX.sym('dtheta_1')
   dtheta_2 = ca.SX.sym('dtheta_2')
   # theta_1_star = ca.SX.sym('theta_1_star')
   # theta_2_star = ca.SX.sym('theta_2_star')
   x = ca.vertcat(theta_1, theta_2, dtheta_1, dtheta_2)

   F = ca.SX.sym('F')
   u = ca.vertcat(F)

   theta_1_dot = ca.SX.sym('theta_1_dot')
   theta_2_dot  = ca.SX.sym('theta_2_dot')
   dtheta_1_dot  = ca.SX.sym('dtheta_1_dot')
   dtheta_2_dot  = ca.SX.sym('dtheta_2_dot')
   # theta_1_star_dot  = ca.SX.sym('theta_1_star_dot')
   # theta_2_star_dot  = ca.SX.sym('theta_2_star_dot')
   xdot = ca.vertcat(theta_1_dot, theta_2_dot, dtheta_1_dot, dtheta_2_dot)

   # x = ca.SX.sym('x', 6)  # x[0] theta_1, x[1] theta_2, x[2] theta_1_dot, x[3] theta_2_dot, x[4] theta_star_1, x[5] theta_star_2
   # u = ca.SX.sym('u', 1)  # u tau

   # mass matrix elements
   m11 = LINK_MOI + LINK_MOI + LINK_MASS_2*LINK_LENGTH_1**2 + 2*LINK_MASS_2*LINK_LENGTH_1*LINK_COM_POS_2*ca.cos(theta_2)
   m12 = LINK_MOI + LINK_MASS_2*LINK_LENGTH_1*LINK_COM_POS_2*ca.cos(theta_2)
   m21 = LINK_MOI + LINK_MASS_2*LINK_LENGTH_1*LINK_COM_POS_2*ca.cos(theta_2)
   m22 = LINK_MOI 

   Mass = ca.vertcat(
          ca.horzcat(m11, m12),
          ca.horzcat(m21, m22))
   Mass_inv = ca.solve(Mass, ca.SX.eye(2))
   # mass_func = ca.Function("mass_det", [x], [ca.det(Mass)])
   # test_x = np.array([0, 0, 0, 0, np.pi, 0])
   # print("Mass determinant at test x:", mass_func(test_x))

   # Coriolis matrix elements  
   c11 = -2*LINK_MASS_2*LINK_LENGTH_1*LINK_COM_POS_2*ca.sin(theta_2)*dtheta_2
   c12 = -LINK_MASS_2*LINK_LENGTH_1*LINK_COM_POS_2*ca.sin(theta_2)*dtheta_2
   c21 = LINK_MASS_2*LINK_LENGTH_1*LINK_COM_POS_2*ca.sin(theta_2)*dtheta_1
   c22 = 0

   Cor = ca.vertcat(
          ca.horzcat(c11, c12),
          ca.horzcat(c21, c22))
   
   # gravitational torque matrix elements
   taug1 = -LINK_MASS_1*G*LINK_COM_POS_1*ca.sin(theta_1) - LINK_MASS_2*G*(LINK_LENGTH_1*ca.sin(theta_1) + LINK_COM_POS_2*ca.sin(theta_1 + theta_2))
   taug2 = -LINK_MASS_2*G*LINK_COM_POS_2*ca.sin(theta_1 + theta_2)

   Taug = ca.vertcat(taug1,
                     taug2)
   
   # B matrix
   B_u = ca.vertcat(0, F)
   # B_matrix = ca.SX([[0],[1]])
   
   # calculate theta_ddot
   theta_dot = ca.vertcat(dtheta_1, dtheta_2)
   
   theta_ddot = Mass_inv@(Taug + B_u - Cor@theta_dot)
   # print("theta_ddot shape:", theta_ddot.shape)

   f_expl = ca.vertcat(
        dtheta_1, # theta_1_dot
        dtheta_2, # theta_2_dot
        theta_ddot[0], # theta_1_ddot
        theta_ddot[1], # theta_2_ddot
        # -PI_UNDER_2 * (theta_1) * dtheta_1, # theta_1_star_dot
        # -PI_UNDER_2 * (theta_2-ca.pi) * dtheta_2, # theta_2_star_dot
    )
   f_impl = xdot - f_expl
   # x_dot = ca.SX.sym('xdot', 6)
   # f_impl = x_dot - dynam_acrobot
 
   model = AcadosModel()
   model.name = "acrobot_acado"
   model.x    = x
   model.xdot = xdot
   model.u    = u
   # model.p    = []
   model.f_expl_expr = f_expl
   model.f_impl_expr = f_impl

   return model

def Acrobot_gym_model():
   theta1 = ca.SX.sym('theta_1')
   theta2 = ca.SX.sym('theta_2')
   dtheta1 = ca.SX.sym('dtheta_1')
   dtheta2 = ca.SX.sym('dtheta_2')
   theta_1_star = ca.SX.sym('theta_1_star')
   theta_2_star = ca.SX.sym('theta_2_star')
   x = ca.vertcat(theta1, theta2, dtheta1, dtheta2, theta_1_star, theta_2_star)

   F = ca.SX.sym('F')
   u = ca.vertcat(F)

   theta_1_dot = ca.SX.sym('theta_1_dot')
   theta_2_dot  = ca.SX.sym('theta_2_dot')
   dtheta_1_dot  = ca.SX.sym('dtheta_1_dot')
   dtheta_2_dot  = ca.SX.sym('dtheta_2_dot')
   theta_1_star_dot  = ca.SX.sym('theta_1_star_dot')
   theta_2_star_dot  = ca.SX.sym('theta_2_star_dot')
   xdot = ca.vertcat(theta_1_dot, theta_2_dot, dtheta_1_dot, dtheta_2_dot, theta_1_star_dot, theta_2_star_dot)

   # x = ca.SX.sym('x', 6)  # x[0] theta_1, x[1] theta_2, x[2] theta_1_dot, x[3] theta_2_dot, x[4] theta_star_1, x[5] theta_star_2
   # u = ca.SX.sym('u', 1)  # u tau

   # theta1 = x[0]
   # theta2 = x[1]
   # dtheta1 = x[2]
   # dtheta2 = x[3]

   m1 = LINK_MASS_1
   m2 = LINK_MASS_2
   l1 = LINK_LENGTH_1
   lc1 = LINK_COM_POS_1
   lc2 = LINK_COM_POS_2
   I1 = LINK_MOI
   I2 = LINK_MOI
   g = 9.8

   d1 = m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * ca.cos(theta2))+ I1 + I2

   d2 = m2 * (lc2**2 + l1 * lc2 * ca.cos(theta2)) + I2
   phi2 = m2 * lc2 * g * ca.cos(theta1 + theta2 - ca.pi / 2.0)
   phi1 = -m2 * l1 * lc2 * dtheta2**2 * ca.sin(theta2) - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * ca.sin(theta2) + (m1 * lc1 + m2 * l1) * g * ca.cos(theta1 - ca.pi / 2) + phi2

   ddtheta2 = (F + d2 / d1 * phi1 - phi2) / (m2 * lc2**2 + I2 - d2**2 / d1)
   ddtheta1 = -(d2 * ddtheta2 + phi1) / d1

   f_expl = ca.vertcat(
       dtheta1, # theta_1_dot
       dtheta2, # theta_2_dot
       ddtheta1, # theta_1_ddot
       ddtheta2, # theta_2_ddot
       -PI_UNDER_2 * theta1 * dtheta1, # theta_1_star_dot
       -PI_UNDER_2 * (theta2-ca.pi) * dtheta2, # theta_2_star_dot
       )

   # x_dot = ca.SX.sym('xdot', 6)
   f_impl = xdot - f_expl

   model = AcadosModel()
   model.name = "acrobot_acado"
   model.x    = x
   model.xdot = xdot
   model.u    = u
   model.f_expl_expr = f_expl
   model.f_impl_expr = f_impl

   return model

########## create Acado ocp (optimal control problem) solver ##########
def Acado_ocp_solver(x0):
   ocp = AcadosOcp()
   
   # ocp solver
   ocp.solver_options.N_horizon = N
   ocp.solver_options.tf = TF
   
   # load acrobot acado model
   model = Acrobot_Acado_model()
   ocp.model = model

   # ocp.model.x = model.x
   # ocp.model.u = model.u
   ocp.model.cost_y_expr = ca.vertcat(1 + ca.cos(model.x[0]), 1 - ca.cos(model.x[1]), model.x[2], model.x[3], model.u)# ca.vertcat(model.x, model.u)    ocp.model.cost_y_expr = ca.vertcat(model.x[2], model.x[3], model.x[4], model.x[5], model.u)
   ocp.model.cost_y_expr_e = ca.vertcat(1 + ca.cos(model.x[0]), 1 - ca.cos(model.x[1]), model.x[2], model.x[3]) # terminal cost   model.x    ca.vertcat(model.x[2], model.x[3], model.x[4], model.x[5])

   # set dimensions
   nx = model.x.rows()
   nu = model.u.rows()
   ny = nx + nu
   ny_e = nx

   # cost function
   ocp.cost.cost_type = 'NONLINEAR_LS'
   ocp.cost.cost_type_e = 'NONLINEAR_LS'

   # dyn_func = ca.Function('f', [model.x, model.u], [model.f_expl_expr])
   # for test_theta1 in np.linspace(-np.pi, np.pi, 10):
    # test_x = np.array([test_theta1, 0, 0, 0, 0, 0])
    # test_u = 5
    # val = dyn_func(test_x, test_u)
    # print(test_theta1, val)
   # test_x = np.array([0, 0, 0, 0, np.pi, 0])
   # test_u = 10
   # test_val = dyn_func(test_x, test_u) 
   # print("Dynamics output:", test_val)

   
   # weights
   ocp.cost.W = W
   # ocp.cost.W_0 = W_INI
   ocp.cost.W_e = W_TERMINAL
 
   # y = V_x*x + Vu*u
   # Vx = np.zeros((ny, nx)) # 7*6
   # Vx[:nx, :nx] = np.eye(nx)
   # ocp.cost.Vx = Vx

   # Vx = np.zeros((ny-2, nx)) # 5*6
   # Vx[0, 2] = 1.0
   # Vx[1, 3] = 1.0
   # Vx[2, 4] = 1.0
   # Vx[3, 5] = 1.0
   # ocp.cost.Vx = Vx

   # Vu = np.zeros((ny, nu)) # 7*1
   # Vu[-1, 0] = 1.0
   # ocp.cost.Vu = Vu

   # Vu = np.zeros((ny-2, nu)) # 7*1
   # Vu[-1, 0] = 1.0
   # ocp.cost.Vu = Vu

   # y_e = Vxe*x_e
   
   # Vx_e = np.zeros((ny_e, nx)) #6*6
   # Vx_e[:nx, :nx] = np.eye(nx)
   # ocp.cost.Vx_e = Vx_e

   # Vx_e = np.zeros((ny_e-2, nx)) #6*6
   # Vx_e[0, 2] = 1.0
   # Vx_e[1, 3] = 1.0
   # Vx_e[2, 4] = 1.0
   # Vx_e[3, 5] = 1.0
   # ocp.cost.Vx_e = Vx_e

   # reference state
   ocp.cost.yref = X_REF
   # ocp.cost.yref_0 = X_REF_INI
   ocp.cost.yref_e = X_REF_TERMINAL

   # constraints
   ocp.constraints.x0 = x0 # initial states

    
   ocp.constraints.lbx = np.array([-2*np.pi, -2*np.pi]) # , -4*np.pi, -9*np.pi # np.array([-2*np.pi, -2*np.pi])
   ocp.constraints.ubx = np.array([2*np.pi, 2*np.pi]) # , 4*np.pi, 9*np.pi # np.array([-2*np.pi, -2*np.pi])
   ocp.constraints.idxbx = np.array([0, 1])  # 2 constraints

   # ocp.constraints.lbu = np.array([-U_BOUND])
   # ocp.constraints.ubu = np.array([U_BOUND])
   # ocp.constraints.idxbu = np.array([0])

   # solver setting
   ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
   ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
   ocp.solver_options.integrator_type = 'IRK'
   ocp.solver_options.nlp_solver_type = 'SQP_RTI'
   # ocp.solver_options.nlp_solver_max_iter = 10
   ocp.solver_options.sim_method_newton_iter = 10

   # build solver
   acados_solver = AcadosOcpSolver(ocp, json_file="acados_ocp_acrobots.json")

   acados_integrator = AcadosSimSolver(ocp, json_file = "acados_ocp_acrobots.json")

   return ocp, acados_solver, acados_integrator



########## single closed control loop ##########
def RunMPCForSingle_IniState_IniGuess(x_ini_guess: float, u_ini_guess:float,idx_group_of_control_step:int,x0_state:np.array, 
                                      u_ini_memory: np.array, x_ini_memory: np.array, j_ini_memory: np.array):

    ################ generate data for 0th step ##########################################################
    try:

        ########## Acado ##########
        ocp, ocp_solver, integrator = Acado_ocp_solver(x0_state)

        nx = ocp.model.x.size()[0]  # number of states (6)
        nu = ocp.model.u.size()[0]  # number of states (1)

        # x, u saving parameters
        X_result = np.zeros((CONTROL_STEPS+1, nx))
        U_result = np.zeros((CONTROL_STEPS, nu))
        X_result[0, :] = x0_state
        print(f'x0 -- {x0_state}')
        print(f'idx_group_of_control_step -- {idx_group_of_control_step}')

        # cost
        cost = np.zeros((CONTROL_STEPS, 1))

        # initial guess
        print(f'u guess -- {u_ini_guess}')
        print(f'x guess -- {x_ini_guess}')
        x_guess = np.array([x_ini_guess[0], x_ini_guess[1], 0, 0])
        u_guess = u_ini_guess
    
        # set initial guess
        ocp_solver.set(0, "u", u_guess) 
        ocp_solver.set(0, "x", x_guess)
        ocp_solver.set(0, "lbx", x0_state)
        ocp_solver.set(0, "ubx", x0_state)
        # cost_0 = ocp_solver.get_cost()
        # cost[0,:] = cost_0

        # ocp solving
        for i in range(0, CONTROL_STEPS):
            # stage 0 of the prediction horizon is the current state
            # ocp_solver.set(i, "lbx", x_next) 
            # ocp_solver.set(i, "ubx", x_next)
            for j in range(N):
                ocp_solver.set(j, "yref", np.array([0, 0, 0, 0, 0])) # np.array([np.pi, 0, 0, 0, 0, 0, 0])   np.array([0, 0, 0, 0, 0])
            ocp_solver.set(N, "yref", np.array([0, 0, 0, 0])) # np.array([np.pi, 0, 0, 0, 0, 0, 0])    np.array([0, 0, 0, 0])
            # ocp_solver.set("yref", X_REF)
            status = ocp_solver.solve()

            # ocp.solver_options.print_level = 3
            # ocp_solver.print_statistics()

            if status != 0:
               print(f"Solver failed at step {i} with status {status}")

            u_solve = ocp_solver.get(0, "u")  
            U_result[i,:] = u_solve
            cost_solve = ocp_solver.get_cost()
            cost[i,:] = cost_solve

            # state updating
            # integrator.set("x", X_result[i, :])
            # integrator.set("u", u_solve)
            # integrator.solve()
            # x_next = integrator.get("x")
            x_next = ocp_solver.get(1, "x")
            ocp_solver.set(0, "lbx", x_next)
            ocp_solver.set(0, "ubx", x_next)
       
            X_result[i+1,:] = x_next
            if i+1 <= N:
               ocp_solver.set(i+1, "x", x_guess)
        
        print(f'X_last_result -- {X_result[-1,:]}')
        print(f'U_last_result -- {U_result[-1,:]}')
        print(f'cost_last_result -- {cost[-1,:]}')

        time = np.arange(X_result.shape[0])  # time steps

        fig, axs = plt.subplots(6, 1, figsize=(8, 12), sharex=True)

        state_labels = [
            'theta1 (rad)',
            'theta2 (rad)',
            'dtheta1 (rad/s)',
            'dtheta2 (rad/s)',
            # 'theta1_star',
            # 'theta2_star'
        ]

        # Plot states
        for i in range(4):
            axs[i].plot(time, X_result[:, i], label=state_labels[i])
            axs[i].set_ylabel(state_labels[i])
            axs[i].grid(True)

        # Plot control input
        axs[4].plot(time[:-1], U_result[:, 0], label='Torque u', color='r')
        axs[4].set_ylabel('Control u (Nm)')
        # axs[6].set_xlabel('Time step')
        axs[4].grid(True)

        # Plot control input
        axs[5].plot(time[:-1], cost[:, 0], label='Cost', color='r')
        axs[5].set_ylabel('Cost')
        axs[5].set_xlabel('Time step')
        axs[5].grid(True)

        plt.suptitle('Acrobot States and Control Input Over Time')
        plt.legend()
        ini_theta1 = f'{x0_state[0]:.2f}'
        ini_theta2 = f'{x0_state[1]:.2f}'
        guess_1 = f'{x_ini_guess[0]:.2f}'
        guess_2 = f'{x_ini_guess[1]:.2f}'
        figure_name = 'idx-' + str(idx_group_of_control_step) + '_x0_' + ini_theta1 + '_' + ini_theta2 + '_x-guess_' + guess_1 + '_' + guess_2 + '.pdf'
        figure_path = os.path.join(FOLDER_PATH, figure_name)
        plt.savefig(figure_path)
        # plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.show()
        
        # noisy at x0
        # MPC_NoiseData_Process(x0_state, x_ini_guess, u_ini_guess, idx_group_of_control_step, u_random_memory, x_random_memory, j_random_memory)
        
        ############################################## generate data for control step loop ##############################################
        # main mpc loop
        # for idx_control_step in range(1, CONTROL_STEPS):
        #     #system dynamic update x 
        #     x0_next = EulerForwardCartpole_virtual(TS,x0_state,u0)
            
        #     ################################################# normal mpc loop to update state #################################################
        #     u0_cur = MPC_NormalData_Process(x0_next, x_ini_guess, u_ini_guess, idx_group_of_control_step, u_ini_memory, j_ini_memory, x_ini_memory, idx_control_step)

        #     ################################## noise  ##################################
        #     MPC_NoiseData_Process(x0_next, x_ini_guess, u_ini_guess, idx_group_of_control_step, u_random_memory, x_random_memory, j_random_memory, idx_control_step, True)
            
        #     # update
        #     x0_state = x0_next
        #     u0 = u0_cur
        

        #################### data saving ####################
        # to tensor
        torch_u_ini_memory_tensor = torch.Tensor(u_ini_memory)
        # torch_u_random_memory_tensor = torch.Tensor(u_random_memory)
        torch_x_ini_memory_tensor = torch.Tensor(x_ini_memory)
        # torch_x_random_memory_tensor = torch.Tensor(x_random_memory)
        torch_j_ini_memory_tensor = torch.Tensor(j_ini_memory)
        # torch_j_random_memory_tensor = torch.Tensor(j_random_memory)

        # torch.save(torch_u_ini_memory_tensor, os.path.join(FOLDER_PATH , f'pure_u_data_' + 'idx-' + str(idx_group_of_control_step) + '_test1.pt'))
        # torch.save(torch_x_ini_memory_tensor , os.path.join(FOLDER_PATH , f'pure_x_data_' + 'idx-' + str(idx_group_of_control_step) + '_test1.pt'))
        # torch.save(torch_j_ini_memory_tensor, os.path.join(FOLDER_PATH , f'pure_j_data_' + 'idx-' + str(idx_group_of_control_step) + '_test1.pt'))
        
        # cat
        # u_data = torch.cat((torch_u_ini_memory_tensor, torch_u_random_memory_tensor), dim=0)
        # x_data = torch.cat((torch_x_ini_memory_tensor, torch_x_random_memory_tensor), dim=0)
        # j_data = torch.cat((torch_j_ini_memory_tensor, torch_j_random_memory_tensor), dim=0)

        print(f'u_size -- {torch_u_ini_memory_tensor.size()}')
        print(f'x_size -- {torch_x_ini_memory_tensor.size()}')
        print(f'j_size -- {torch_j_ini_memory_tensor.size()}')

        # save data in PT file for training
        # torch.save(u_data, os.path.join(FOLDER_PATH , f'u_data_' + 'idx-' + str(idx_group_of_control_step) + '_test1.pt'))
        # torch.save(x_data, os.path.join(FOLDER_PATH , f'x_data_' + 'idx-' + str(idx_group_of_control_step) + '_test1.pt'))
        # torch.save(j_data, os.path.join(FOLDER_PATH , f'j_data_' + 'idx-' + str(idx_group_of_control_step) + '_test1.pt'))

        # # plots
        # t = np.arange(0, CONTROL_STEPS*TS, TS) # np.arange(len(joint_states[1])) * panda.opt.timestep
        # print(f't -- {len(t)}')

        # # plot 1: 5 states
        # plt.figure()
        # # for i in range(5):
        # plt.plot(t, x_ini_memory[:,0], label=f"x")
        # plt.plot(t, x_ini_memory[:,1], label=f"x_dot")
        # plt.plot(t, x_ini_memory[:,2], label=f"theta")
        # plt.plot(t, x_ini_memory[:,3], label=f"theta_dot")
        # plt.plot(t, x_ini_memory[:,4], label=f"theta_star")
        # for z in range(CONTROL_STEPS):
        #         for i in range(5):
        #             noisy_state_each_ctl_step = x_random_memory[z*NUM_NOISY_DATA:z*NUM_NOISY_DATA+NUM_NOISY_DATA,i]
        #             for k in range(0,NUM_NOISY_DATA):
        #                     plt.scatter(t[z], noisy_state_each_ctl_step[k], s = 20, color = 'lightgrey')
        # plt.scatter(t[CONTROL_STEPS-1], noisy_state_each_ctl_step[0], s = 20, color = 'lightgrey', label=f"noise")
                
        # plt.xlabel("Time [s]")
        # plt.ylabel("state")
        # plt.legend()
        # figure_name = 'idx-' + str(idx_group_of_control_step) + '_x_0121' + '.pdf'
        # figure_path = os.path.join(FOLDER_PATH, figure_name)
        # plt.savefig(figure_path)

        # # plot 2: u
        # plt.figure()
        # plt.plot(t, u_ini_memory[:,0,0], label=f"u")
        # for z in range(CONTROL_STEPS):
        #         noisy_u_each_ctl_step = u_random_memory[z*NUM_NOISY_DATA:z*NUM_NOISY_DATA+NUM_NOISY_DATA,0,0]
        #         for k in range(0, NUM_NOISY_DATA):
        #             plt.scatter(t[z], noisy_u_each_ctl_step[k], s = 20, color = 'lightgrey')
        # plt.scatter(t[CONTROL_STEPS-1], noisy_u_each_ctl_step[0], s = 20, color = 'lightgrey', label=f"noise")
                
        # plt.xlabel("Time [s]")
        # plt.ylabel("control input")
        # plt.legend()
        # figure_name = 'idx-' + str(idx_group_of_control_step) + '_u_0121' + '.pdf'
        # figure_path = os.path.join(FOLDER_PATH, figure_name)
        # plt.savefig(figure_path)

        # # plot 3: cost
        # plt.figure()
        # plt.plot(t, j_ini_memory[:,0])
        # for z in range(CONTROL_STEPS):
        #         noisy_j_each_ctl_step = j_random_memory[z*NUM_NOISY_DATA:z*NUM_NOISY_DATA+NUM_NOISY_DATA,0]
        #         for k in range(0, NUM_NOISY_DATA):
        #             plt.scatter(t[z], noisy_j_each_ctl_step[k], s = 20, color = 'lightgrey')
        # plt.scatter(t[CONTROL_STEPS-1], noisy_j_each_ctl_step[0], s = 20, color = 'lightgrey', label=f"noise")
                
        # plt.xlabel("Time [s]")
        # plt.ylabel("cost")
        # plt.legend()
        # figure_name = 'idx-' + str(idx_group_of_control_step) + '_j_0121' + '.pdf'
        # figure_path = os.path.join(FOLDER_PATH, figure_name)
        # plt.savefig(figure_path)

    
    except Exception as e:
        print(f"Error: {e}")



def main():
    # num_seed = NUM_SEED
    # ini_data_start_idx = 0
    # noisy_data_start_idx = NUM_INI_STATES*CONTROL_STEPS

    # initial data generating 50*7, 50*7, 50
    # ini_0_states, random_ini_u_guess, ini_data_idx = ini_data_generating()

    # memories u,x,j


    # memories for data
    u_ini_memory = np.zeros((1*CONTROL_STEPS, N, NUM_U)) # 80 64 1 
    # u_random_memory = np.zeros((NUM_NOISY_DATA*CONTROL_STEPS, HOR, 1))
    x_ini_memory = np.zeros((1*CONTROL_STEPS, NUM_X)) 
    # x_random_memory = np.zeros((NUM_NOISY_DATA*CONTROL_STEPS, 5)) 
    j_ini_memory = np.zeros((1*CONTROL_STEPS, 1)) 
    # j_random_memory = np.zeros((NUM_NOISY_DATA*CONTROL_STEPS, 1)) 

    # initial data groups 50
    argument_each_group = []
    for idx_ini_guess in range(0, num_guessgroup): 
        for turn in range(0,num_datagroup):
            # initial guess
            x_ini_guess = initial_guess_x[idx_ini_guess,:] # initial_guess_x[idx_ini_guess,:]
            u_ini_guess = 0 # initial_guess_u[idx_ini_guess]
            idx_group_of_control_step = idx_ini_guess*num_datagroup+turn
            
            #initial states
            theta_1 = rng0[turn,IDX_THETA1_INI]
            theta_2 = rng0[turn,IDX_THETA2_INI]
            theta1_star_0 = Theta1ToThetaStar1(theta_1)
            theta2_star_0 = Theta2ToThetaStar2(theta_2)

            x0 = np.array([theta_1, theta_2, 0, 0]) # np.array([theta_1, theta_2, 0, 0, theta1_star_0, theta2_star_0])
            
            argument_each_group.append((x_ini_guess, u_ini_guess, idx_group_of_control_step, x0, 
                                        u_ini_memory, x_ini_memory, j_ini_memory))
        

    # test_data_group = initial_data_groups[0:2]
    
    # (noisy)
    # for a in range(NUM_INI_STATES):
    #       for b in range(NOISE_DATA_PER_STATE):
    #             initial_data_groups.append([ini_noisy_data_u_guess[a,b,:], ini_noisy_states[a,b,:]])
    
    #     ini_data_groups_array = np.array(initial_data_groups)
    #     print(f'initial_data_groups_size -- {ini_data_groups_array.shape}')

    with Pool(processes=MAX_CORE_CPU) as pool:
          pool.starmap(RunMPCForSingle_IniState_IniGuess, argument_each_group)





if __name__ == "__main__":
    # multiprocessing.set_start_method("spawn")
    main()
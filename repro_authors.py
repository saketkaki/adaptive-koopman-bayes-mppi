# Faithful script version of the authors' Planar_Quadrotor_Linear_Adaptive_Koopman_MPC.ipynb
# (their model, their MPC class, their adaptation net, their hyperparameters).
# Usage: python repro_authors.py [wind_speed] [traj]
import sys, time, os
import numpy as np
import scipy, scipy.sparse
import torch

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Adaptive-koopman')
sys.path.insert(0, REPO)
os.chdir(REPO)

from models.koop_model import lift
from dynamics.quadsim_2d import QuadSim
from core.adapt_net_linear import AdaptNet_linear
from control_files.nmpc_osqp_adapt import NonlinearMPCController
from dynamics.learned_models_control.linear_dynamics import linear_Dynamics

wind = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
traj_type = sys.argv[2] if len(sys.argv) > 2 else 'lemniscate'
traj_length = {'lemniscate': 1250, 'circle': 1020, 'S': 900, 'adobe': 1050}[traj_type]

quad_params = dict(m=2, g=9.81, L=0.2, Ixx=1, V_w=0, alpha_w=0)
quad_params_changed = dict(m=2.4, g=9.81, L=0.2, Ixx=1.2, V_w=wind, alpha_w=np.pi / 4)

torch.manual_seed(0)
env = QuadSim(quad_params)
env_changed = QuadSim(quad_params_changed)
dt = env.dt
num_states, num_inputs = 6, 2

net_params = dict(first_obs_const=True, override_C=True, encoder_output_dim=10)
model_koop_dnn = torch.load('saved_models/planar_quad/linear/Koop_arch.pth', weights_only=False)
A = np.array(model_koop_dnn.A); B = np.array(model_koop_dnn.B); C = np.array(model_koop_dnn.C)
n_obs = A.shape[0]

# reference
x_ref = np.empty((num_states, traj_length + 1))
env.reset(traj=traj_type)
for j in range(traj_length):
    env.step(env.action_space.sample())
    if j == 0:
        x_ref[:, j] = env.Quadrotor.get_state()
    x_ref[:, j + 1] = env.Quadrotor.get_state()

adap_window = 10
adapt_net_params = dict(state_dim=num_states, ctrl_dim=num_inputs, lift_dim=10, optimizer='adam',
                        lr=1.5e-4, epochs=10, batch_size=adap_window, l2_reg=1e-1, l1_reg=1e-2,
                        first_obs_const=True, override_C=True, warm_start=False,
                        del_A_prev=np.zeros_like(A), del_B_prev=np.zeros_like(B))

solver_settings = dict(gen_embedded_ctrl=False, warm_start=True, polish=True, polish_refine_iter=3,
                       scaling=True, adaptive_rho=False, check_termination=25, max_iter=1000,
                       eps_abs=1e-5, eps_rel=1e-6, eps_prim_inf=1e-4, eps_dual_inf=1e-4, linsys_solver='qdldl')
N = 15
Q_mpc = 1.5e5 * scipy.sparse.diags(np.array([1, 1, 1, 1, 1, 0.5]))
QN_mpc = 1e3 * scipy.sparse.diags(np.ones(6))
R_mpc = scipy.sparse.eye(num_inputs)
umax = 30 * np.ones(num_inputs); umin = -umax
xmax = 20 * np.ones(num_states); xmin = -xmax
x_ref_mpc = np.hstack((x_ref, np.tile(x_ref[:, -1].reshape(-1, 1), N)))


def apply(envq, u):
    u1 = min(max(0, 0.5 * (u[0] - u[1] / envq.L)), 30)
    u2 = min(max(0, 0.5 * (u[0] + u[1] / envq.L)), 30)
    envq.Quadrotor.update(dt, u1, u2)
    return envq.Quadrotor.get_state()


def run(adapt):
    model = linear_Dynamics(scipy.sparse.csc_matrix(A), scipy.sparse.csc_matrix(B), C)
    x0 = x_ref[:, 0]
    x_init = np.linspace(x0, x0, N + 1) + 0.1
    u_init = np.ones((N, num_inputs))
    z_init = np.array([lift(x, model_koop_dnn, net_params) for x in x_init])
    ctrl = NonlinearMPCController(model, N, dt, umin, umax, xmin, xmax, Q_mpc, R_mpc, QN_mpc, solver_settings)
    ctrl.construct_controller(z_init, u_init, x_ref[:, 0])
    env_changed.reset(); env_changed.Quadrotor.state = x0.copy()
    x = np.zeros((traj_length + 1, num_states)); x[0] = x0
    u = np.zeros((traj_length, num_inputs))
    z_pred = np.zeros((traj_length + 1, n_obs)); z_pred[0] = lift(x0, model_koop_dnn, net_params)
    dz = np.zeros((traj_length, n_obs))
    t_tot = 0
    for i in range(traj_length):
        t0 = time.time()
        z = lift(x[i], model_koop_dnn, net_params)
        dz[i] = z - z_pred[i]
        if adapt and i >= adap_window + 1:
            net = AdaptNet_linear(adapt_net_params)
            net.model_pipeline(z_pred[i - (adap_window + 1):i - 1].T, u[i - (adap_window + 1):i - 1].T,
                               dz[i - adap_window:i].T, print_epoch=False)
            dA, dB = net.get_del_matrices()
            model.A = scipy.sparse.csc_matrix(model.A.toarray() + dA)
            model.B = scipy.sparse.csc_matrix(model.B.toarray() + dB)
        ctrl.solve_to_convergence(x_ref_mpc[:, i + 1:i + 2 + N], z, ctrl.z_init, ctrl.u_init, max_iter=1, eps=1e-3)
        ctrl.update_initial_guess_()
        u[i] = ctrl.cur_u[0]
        t_tot += time.time() - t0
        z_pred[i + 1] = ctrl.z_N0
        x[i + 1] = apply(env_changed, u[i])
    return x, u, t_tot / traj_length


def rmse(x):
    e = np.linalg.norm(x[:, :2] - x_ref[:2].T, axis=1)
    return np.linalg.norm(e) / np.sqrt(len(e))


out = {}
for name, adapt in [('nominal', False), ('nn_adapt', True)]:
    x, u, tpi = run(adapt)
    out[name] = x
    print(f'{traj_type} wind={wind}: {name:9s} pos RMSE = {rmse(x):.4f} m   time/iter = {1e3*tpi:.1f} ms')
np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'results/authors_{traj_type}_w{wind:g}.npz'),
         x_ref=x_ref.T, **out)

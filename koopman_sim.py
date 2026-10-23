# Shared pieces: the authors' trained Koopman model, a numpy lift, the planar quadrotor
# (their simulator class, with parameters that can change mid-flight), reference
# trajectories, and a linear Koopman MPC written directly on OSQP.
import os, sys, warnings
import numpy as np
import scipy.sparse as sp
import osqp
import torch

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, '..', 'Adaptive-koopman')
sys.path.insert(0, REPO)

from dynamics.quadrotor_2d import Quadrotor  # noqa: E402

DT = 0.01
NOMINAL = dict(m=2.0, g=9.81, L=0.2, Ixx=1.0, V_w=0.0, alpha_w=np.pi / 4)


# ---------------------------------------------------------------- model
class KoopModel:
    """Authors' trained linear Koopman model z+ = A z + B u, x = C z, with a numpy lift."""

    def __init__(self):
        net = torch.load(os.path.join(REPO, 'saved_models/planar_quad/linear/Koop_arch.pth'),
                         weights_only=False)
        self.A = np.array(net.A.tolist())
        self.B = np.array(net.B.tolist())
        self.C = np.array(net.C.tolist())
        e = net.net
        layers = [e.encoder_fc_in] + list(e.encoder_fc_hid) + [e.encoder_fc_out]
        self.W = [l.weight.detach().numpy().astype(float) for l in layers]
        self.b = [l.bias.detach().numpy().astype(float) for l in layers]
        self.torch_net = net
        self.nz, self.nu = self.B.shape

    def lift(self, x):
        h = np.asarray(x, float)
        for W, b in zip(self.W[:-1], self.b[:-1]):
            h = np.tanh(np.dot(W, h) + b)
        enc = np.dot(self.W[-1], h) + self.b[-1]
        return np.concatenate(([1.0], x, enc))

    def lift_many(self, X):
        h = X
        for W, b in zip(self.W[:-1], self.b[:-1]):
            h = np.tanh(np.dot(h, W.T) + b)
        enc = np.dot(h, self.W[-1].T) + self.b[-1]
        return np.hstack([np.ones((len(X), 1)), X, enc])


def training_stats(model):
    """Scales of [z; u] and the nominal model's one-step residual covariance, from the authors' training data."""
    X = np.load(os.path.join(REPO, 'datasets/quad_state_data.npy'))
    U = np.load(os.path.join(REPO, 'datasets/quad_input_data.npy'))
    Z = model.lift_many(X.reshape(-1, 6)).reshape(X.shape[0], X.shape[1], -1)
    z0, z1, u0 = Z[:, :-1].reshape(-1, model.nz), Z[:, 1:].reshape(-1, model.nz), U.reshape(-1, model.nu)
    res = z1 - np.dot(z0, model.A.T) - np.dot(u0, model.B.T)
    phi = np.hstack([z0, u0])
    mean = phi.mean(axis=0)
    mean[0] = 0.0
    scale = phi.std(axis=0)
    scale[0] = 1.0                      # constant observable
    scale = np.maximum(scale, 1e-3)
    res_cov = np.cov(res[:, 1:].T)      # rows 1.. are adapted, row 0 (constant) is fixed
    return dict(scale=scale, mean=mean, res_cov=res_cov, res_rms=np.sqrt(np.mean(res[:, 1:] ** 2, axis=0)))


# ---------------------------------------------------------------- plant
class Plant:
    """Their Quadrotor ODE; schedule(t) returns the params to use at time t (mid-flight changes)."""

    def __init__(self, x0, schedule):
        self.quad = Quadrotor(pos=x0[:2], orient=x0[2], vel=x0[3:5], orient_dot=x0[5], params=NOMINAL)
        self.quad.state = np.array(x0, float).copy()
        self.schedule = schedule
        self.t = 0.0

    def step(self, u):
        p = self.schedule(self.t)
        q = self.quad
        q.m, q.Ixx, q.V_w, q.alpha_w = p['m'], p['Ixx'], p['V_w'], p['alpha_w']
        t1 = min(max(0.0, 0.5 * (u[0] - u[1] / q.L)), 30.0)   # same mixer + clamp as the authors
        t2 = min(max(0.0, 0.5 * (u[0] + u[1] / q.L)), 30.0)
        q.update(DT, t1, t2)
        self.t += DT
        return q.get_state()


def constant_schedule(**changes):
    p = dict(NOMINAL, **changes)
    return lambda t: p


def jump_schedule(events):
    """events: list of (t_start, dict of param overrides) applied cumulatively."""
    events = sorted(events, key=lambda e: e[0])

    def f(t):
        p = dict(NOMINAL)
        for t0, ch in events:
            if t >= t0:
                p.update(ch)
        return p
    return f


_REF_CACHE = {}


def reference(traj='lemniscate'):
    """Same reference generation as the authors: their PD tracker flying the nominal quad."""
    if traj in _REF_CACHE:
        return _REF_CACHE[traj]
    from dynamics.quadsim_2d import QuadSim
    import contextlib, io
    length = {'lemniscate': 1250, 'circle': 1020, 'S': 900, 'adobe': 1050}[traj]
    with contextlib.redirect_stdout(io.StringIO()):
        env = QuadSim(dict(NOMINAL, alpha_w=0))
    env.reset(traj=traj)
    xr = np.empty((length + 1, 6))
    xr[0] = env.Quadrotor.get_state()
    for j in range(length):
        env.step(None)
        xr[j + 1] = env.Quadrotor.get_state()
    _REF_CACHE[traj] = xr
    return xr


# ---------------------------------------------------------------- MPC
class KoopMPC:
    """Linear Koopman MPC, same cost/constraints/horizon as the authors' controller.

    Variables: z_0..z_N, u_0..u_{N-1}. Cost: sum_{j<N} |C z_j - r_j|_Q^2 + |C z_N - r_N|_QN^2 + sum |u_j|_R^2.
    Optional per-step state bounds (for constraint tightening).
    """

    def __init__(self, nz, nu, C, N=15, umax=30.0, xmax=20.0, QN=None, Q=None):
        self.nz, self.nu, self.C, self.N = nz, nu, C, N
        self.nx = C.shape[0]
        self.Q = 1.5e5 * np.diag([1, 1, 1, 1, 1, 0.5]) if Q is None else Q
        self.QN = 1e3 * np.eye(self.nx) if QN is None else QN
        self.R = np.eye(nu)
        self.umax = umax * np.ones(nu)
        self.xmax = xmax * np.ones(self.nx)
        self.nvar = (N + 1) * nz + N * nu
        self.prev = None
        self.settings = dict(verbose=False, warm_start=True, polish=True, eps_abs=1e-5, eps_rel=1e-6,
                             max_iter=4000, adaptive_rho=True)
        CQC = np.dot(self.C.T, np.dot(self.Q, self.C))
        CQNC = np.dot(self.C.T, np.dot(self.QN, self.C))
        self.P = sp.block_diag([sp.kron(sp.eye(N), CQC), CQNC, sp.kron(sp.eye(N), self.R)], format='csc')
        self.prob = None

    def set_Q(self, Q):
        """Change the stage state weight online (same sparsity pattern, positive diagonal)."""
        self.Q = Q
        CQC = np.dot(self.C.T, np.dot(Q, self.C))
        CQNC = np.dot(self.C.T, np.dot(self.QN, self.C))
        self.P = sp.block_diag([sp.kron(sp.eye(self.N), CQC), CQNC, sp.kron(sp.eye(self.N), self.R)], format='csc')
        if self.prob is not None:
            self.prob.update(Px=sp.triu(self.P, format='csc').data)

    def _A_matrix(self, A, B):
        N, nz, nu = self.N, self.nz, self.nu
        rows, cols, vals = [], [], []
        # dense dynamics blocks written explicitly so the sparsity pattern never changes
        Ai, Aj = np.meshgrid(np.arange(nz), np.arange(nz), indexing='ij')
        Bi, Bj = np.meshgrid(np.arange(nz), np.arange(nu), indexing='ij')
        r0 = 0
        rows += list(range(nz)); cols += list(range(nz)); vals += [1.0] * nz      # z_0 = z_meas
        for j in range(N):
            r = nz * (j + 1)
            rows += list(r + Ai.ravel()); cols += list(nz * j + Aj.ravel()); vals += list(A.ravel())
            rows += list(r + Bi.ravel()); cols += list((N + 1) * nz + nu * j + Bj.ravel()); vals += list(B.ravel())
            rows += list(r + np.arange(nz)); cols += list(nz * (j + 1) + np.arange(nz)); vals += [-1.0] * nz
        r0 = (N + 1) * nz
        for j in range(N):                                                          # input bounds
            rows += list(r0 + nu * j + np.arange(nu)); cols += list((N + 1) * nz + nu * j + np.arange(nu))
            vals += [1.0] * nu
        r0 += N * nu
        Ci, Cj = np.nonzero(self.C)
        for j in range(N + 1):                                                      # state bounds on C z_j
            rows += list(r0 + self.nx * j + Ci); cols += list(nz * j + Cj); vals += list(self.C[Ci, Cj])
        nrows = r0 + (N + 1) * self.nx
        return sp.csc_matrix((vals, (rows, cols)), shape=(nrows, self.nvar))

    def solve(self, A, B, z0, xref, xlo=None, xhi=None):
        """xref: (N+1, nx) targets for z_0..z_N. xlo/xhi: optional (N+1, nx) state bounds."""
        N, nz, nu = self.N, self.nz, self.nu
        q = np.zeros(self.nvar)
        for j in range(N + 1):
            W = self.Q if j < N else self.QN
            q[nz * j: nz * (j + 1)] = -np.dot(self.C.T, np.dot(W, xref[j]))
        lo = np.concatenate([-z0, np.zeros(N * nz), np.tile(-self.umax, N),
                             (np.tile(-self.xmax, N + 1) if xlo is None else xlo.ravel())])
        hi = np.concatenate([-z0, np.zeros(N * nz), np.tile(self.umax, N),
                             (np.tile(self.xmax, N + 1) if xhi is None else xhi.ravel())])
        lo[:nz] = hi[:nz] = z0  # z_0 row has +1 coefficient
        Amat = self._A_matrix(A, B)
        if self.prob is None:
            self.prob = osqp.OSQP()
            self.prob.setup(P=sp.triu(self.P, format='csc'), q=q, A=Amat, l=lo, u=hi, **self.settings)
        else:
            self.prob.update(q=q, l=lo, u=hi, Ax=Amat.data)
        if self.prev is not None:
            self.prob.warm_start(x=self.prev)
        res = self.prob.solve()
        if res.info.status not in ('solved', 'solved inaccurate'):
            raise RuntimeError('OSQP: ' + res.info.status)
        x = res.x
        Z = x[:(N + 1) * nz].reshape(N + 1, nz)
        U = x[(N + 1) * nz:].reshape(N, nu)
        # shift for warm start
        nxt = np.concatenate([Z[1:].ravel(), np.dot(A, Z[-1]) + np.dot(B, U[-1]), U[1:].ravel(), U[-1]])
        self.prev = nxt
        return U, Z


# ---------------------------------------------------------------- closed loop
def snr_noise_std(xref, snr_db):
    if snr_db is None:
        return np.zeros(6)
    power = np.var(xref, axis=0) + 1e-12
    return np.sqrt(power / 10 ** (snr_db / 10))


def run_closed_loop(model, adapter, schedule, traj='lemniscate', snr_db=None, seed=0,
                    tighten=None, floor=None, N=15, QN=None, Q=None, ki=0.0, i_max=0.5, target_fix=False,
                    att_sched=None, mppi=None, tighten_noise=None):
    """Run one closed-loop episode.

    adapter: object with .A, .B, .observe(z_prev, u_prev, z_now), optional .predictive_cov(...)
    tighten: None or kappa (std multiplier) for chance-constraint tightening of `floor`.
    floor: None or (state_index, lower_bound) hard state constraint, e.g. (1, -1.1) for altitude.
    ki: integral action on position error (1/s). The integral of (measured - reference) position shifts the
        position targets the MPC sees by -ki * integral (clipped to +-i_max m). Offset-free tracking without
        touching the QP; applied identically for every adapter.
    target_fix: offset-free target calculation. The learned constant lateral acceleration
        d = C_vy (A[:, 0] - A_nom[:, 0]) (affine column) is cancelled by shifting the attitude target by
        -d / (d vy+/d theta), with the sensitivity taken from the learned model through the lift.
        Without it the MPC pins attitude to the NOMINAL reference and has to accept a lateral offset.
    att_sched: None or (w_lo, w_hi, (p_lo, p_hi), (n_lo, n_hi)). Confidence-scheduled attitude weight,
        on a log scale between w_lo (aggressive) and w_hi (the authors' conservative weight). It goes
        conservative when EITHER the adapter's parameter uncertainty phi'P phi is high (just after a
        detected change; p_lo..p_hi) OR the sensor noise is high (model-free estimate from second
        differences of the measurements, relative amplitude n_lo..n_hi). Adapters without a posterior
        always get w_hi.
    """
    rng = np.random.default_rng(seed)
    xref = reference(traj)
    T = len(xref) - 1
    xref_mpc = np.vstack([xref, np.repeat(xref[-1:], N + 1, axis=0)])
    noise_std = snr_noise_std(xref, snr_db)
    plant = Plant(xref[0], schedule)
    mpc = KoopMPC(model.nz, model.nu, model.C, N=N, QN=QN, Q=Q)
    x = np.zeros((T + 1, 6)); x[0] = xref[0]
    u = np.zeros((T, model.nu))
    adapt_time = np.zeros(T)
    margin_log = np.zeros(T)
    z_prev = None
    failed = False
    n_fail, last_plan, qp_fail_steps = 0, None, []
    integ = np.zeros(2)
    theta_off = 0.0
    w_log = np.zeros(T)
    hist, noise_var, f_s = [], np.zeros(6), 1.0
    sched_log = []
    w_prev = 1.0
    ctrl_time = np.zeros(T)
    ref_scale = np.std(xref, axis=0) + 1e-6
    A_nom, B_nom = model.A.copy(), model.B.copy()
    import time
    for i in range(T):
        x_meas = x[i] + noise_std * rng.standard_normal(6)
        z = model.lift(x_meas)
        t0 = time.perf_counter()
        if z_prev is not None:
            adapter.observe(z_prev, u[i - 1], z)
        adapt_time[i] = time.perf_counter() - t0
        z_fb = z
        target = xref_mpc[i + 1:i + N + 2].copy()
        # sensor noise, model-free: second differences of the raw POSITION measurements (position is
        # smooth at 100 Hz however hard the controller works)
        hist.append(x_meas)
        if len(hist) >= 3:
            dd = hist[-1] - 2 * hist[-2] + hist[-3]
            kk = len(hist) - 2
            beta = max(0.98, 1.0 - 1.0 / kk)
            noise_var = beta * noise_var + (1 - beta) * dd ** 2 / 6.0
            hist = hist[-3:]
        if att_sched is not None:
            w_lo, w_hi, (p_lo, p_hi), (n_lo, n_hi) = att_sched
            noise_lvl = float(np.max(np.sqrt(noise_var[:2]) / ref_scale[:2]))   # positions: smooth whatever the controller does
            s_par = getattr(adapter, 's_par', None)
            if s_par is None or i < 2:
                f = 1.0
            else:
                f_par = np.clip(np.log10(max(s_par, 1e-12) / p_lo) / np.log10(p_hi / p_lo), 0, 1)
                f_noise = np.clip(np.log10(max(noise_lvl, 1e-12) / n_lo) / np.log10(n_hi / n_lo), 0, 1)
                f_new = max(f_par, f_noise)
                sched_log.append((f_par, f_noise, int(np.argmax(np.sqrt(noise_var) / ref_scale))))
                f_s = f_new if f_new > f_s else 0.95 * f_s + 0.05 * f_new        # back off fast, relax slowly
                f = f_s
            w = 10 ** (np.log10(w_lo) + f * (np.log10(w_hi) - np.log10(w_lo)))
            w_log[i] = w
            if abs(w - w_prev) > 1e-6 * w_prev:           # only touch the QP when the weight really changes
                mpc.set_Q(1.5e5 * np.diag([1, 1, w, 1, 1, 0.5 * w]))
                if mppi is not None:
                    mppi.set_Q(1.5e5 * np.diag([1, 1, w, 1, 1, 0.5 * w]))
                w_prev = w
        if target_fix and i > 0:
            xf = np.dot(model.C, z_fb)
            up = u[i - 1]
            # only the learned AFFINE term (a constant force such as wind); mass/inertia changes live in B
            d = float(np.dot(model.C[3], adapter.A[:, 0] - A_nom[:, 0]))
            eps = 1e-3
            xp = xf.copy(); xp[2] += eps
            s_th = np.dot(model.C[3], np.dot(adapter.A, model.lift(xp) - z_fb)) / eps
            if abs(s_th) > 1e-6:
                theta_off = 0.95 * theta_off + 0.05 * float(np.clip(-d / s_th, -0.3, 0.3))
            target[:, 2] += theta_off
        if ki > 0:
            integ = np.clip(integ + DT * (np.dot(model.C, z_fb)[:2] - xref[i, :2]), -i_max / ki, i_max / ki)
            target[:, :2] -= ki * integ
        xlo = xhi = None
        if floor is not None:
            k, lb = floor
            xlo = np.tile(-mpc.xmax, (N + 1, 1)); xhi = np.tile(mpc.xmax, (N + 1, 1))
            xlo[1:, k] = lb
            if tighten is not None and hasattr(adapter, 'predictive_std') and mpc.prev is not None:
                Zp = mpc.prev[:(N + 1) * model.nz].reshape(N + 1, model.nz)
                Up = mpc.prev[(N + 1) * model.nz:].reshape(N, model.nu)
                std = adapter.predictive_std(Zp, Up, model.C)   # (N+1, nx)
                xlo[1:, k] = lb + tighten * std[1:, k]
                margin_log[i] = tighten * std[1, k]
            if tighten_noise is not None and i >= 2:
                xlo[1:, k] = lb + tighten_noise * np.sqrt(noise_var[k])
                margin_log[i] = tighten_noise * np.sqrt(noise_var[k])
        if mppi is not None:
            t1 = time.perf_counter()
            U_seed = None
            if mppi.seed_qp:
                for lo_, hi_ in ([(xlo, xhi)] + ([(None, None)] if xlo is not None else [])):
                    try:
                        U_seed, _ = mpc.solve(adapter.A, adapter.B, z_fb, target, lo_, hi_)
                        break
                    except RuntimeError:
                        mpc.prev = None
            u[i] = mppi.plan(adapter, z_fb, target, np.array([NOMINAL['m'] * NOMINAL['g'], 0.0]), U_seed)
            ctrl_time[i] = time.perf_counter() - t1
            z_prev = z
            x[i + 1] = plant.step(u[i])
            if not np.all(np.isfinite(x[i + 1])) or np.abs(x[i + 1]).max() > 50:
                x[i + 1:] = np.nan
                failed = True
                break
            continue
        # fallbacks: tightened floor -> plain floor -> no floor -> give up (model has diverged)
        tries = [(xlo, xhi)]
        if floor is not None:
            xlo2 = xlo.copy(); xlo2[1:, floor[0]] = floor[1]
            tries += [(xlo2, xhi), (None, None)]
        U = None
        for lo_, hi_ in tries:
            try:
                t1 = time.perf_counter()
                U, Z = mpc.solve(adapter.A, adapter.B, z_fb, target, lo_, hi_)
                ctrl_time[i] = time.perf_counter() - t1
                break
            except RuntimeError:
                mpc.prev = None
        if U is None or not np.all(np.isfinite(U)):
            # QP failed: apply the tail of the last good plan; give up after 5 consecutive failures
            n_fail += 1
            if last_plan is None or n_fail > 5:
                x[i + 1:] = np.nan
                failed = True
                break
            U = last_plan[min(n_fail, len(last_plan) - 1):]
            qp_fail_steps.append(i)
        else:
            n_fail = 0
            last_plan = U
        u[i] = U[0]
        z_prev = z      # the adapter always learns from raw measurements
        x[i + 1] = plant.step(u[i])
        if not np.all(np.isfinite(x[i + 1])) or np.abs(x[i + 1]).max() > 50:
            x[i + 1:] = np.nan
            failed = True
            break
    return dict(x=x, u=u, xref=xref, adapt_time=adapt_time, margin=margin_log, failed=failed,
                qp_fail_steps=qp_fail_steps, w_att=w_log, ctrl_time=ctrl_time,
                sched_log=sched_log)


def pos_err(res):
    return np.linalg.norm(res['x'][:, :2] - res['xref'][:, :2], axis=1)


def rmse(res, t0=None, t1=None):
    e = pos_err(res)
    i0 = 0 if t0 is None else int(round(t0 / DT))
    i1 = len(e) if t1 is None else int(round(t1 / DT))
    seg = e[i0:i1]
    if np.any(~np.isfinite(seg)):
        return np.inf
    return float(np.sqrt(np.mean(seg ** 2)))

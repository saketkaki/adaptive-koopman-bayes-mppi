# Model predictive path integral (MPPI) control on a lifted linear Koopman model.
#
# Rollouts are batched matrix products. With n_models > 1 every control sequence is evaluated under several
# models [A B] drawn from the adapter's posterior, and sequences are scored by a risk measure over those models
# (mean + CVaR). With n_models = 1 it is ordinary MPPI on the posterior-mean model.
import numpy as np


class KoopMPPI:
    def __init__(self, C, N=15, K=256, n_models=1, sigma=(2.0, 0.15), temp=0.05, iters=2,
                 risk='cvar', alpha=0.25, risk_w=1.0, floor=None, floor_w=1e8, umax=30.0,
                 Q=None, QN=None, R=None, seed=0, scale_uncertainty=True, seed_qp=True, knots=4,
                 state_samples=False, fixed_Q=False, sample_models=True):
        self.C, self.N, self.K, self.S = C, N, K, n_models
        self.nx = C.shape[0]
        self.Q = 1.5e5 * np.diag([1, 1, 1, 1, 1, 0.5]) if Q is None else Q
        self.QN = 1e3 * np.eye(self.nx) if QN is None else QN
        self.R = np.eye(2) if R is None else R
        self.sigma = np.array(sigma, float)
        self.temp = temp            # temperature as a fraction of the cost spread (scale-free)
        self.iters = iters
        self.risk, self.alpha, self.risk_w = risk, alpha, risk_w
        self.floor, self.floor_w = floor, floor_w
        self.umax = umax
        self.rng = np.random.default_rng(seed)
        self.scale_uncertainty = scale_uncertainty
        self.U = None
        self.seed_qp = seed_qp      # sample corrections around the QP plan of the posterior-mean model
        self.knots = knots          # smooth perturbations: sampled at a few knots, linearly interpolated
        self.state_samples = state_samples  # also sample the initial lifted state from the sensor-noise estimate
        self.fixed_Q = fixed_Q
        self.sample_models = sample_models   # False: all rollouts use the posterior mean (state belief only)              # keep this cost even if the runner schedules the QP's weight
        self.log = dict(spread=[], dev=[])

    def set_Q(self, Q):
        if not self.fixed_Q:
            self.Q = Q

    def _models(self, adapter):
        """(S, nz, nz), (S, nz, nu) model samples; the first one is always the posterior mean."""
        A, B = adapter.A, adapter.B
        if self.S == 1 or not hasattr(adapter, 'sample_models'):
            return A[None], B[None]
        if not self.sample_models:
            return np.repeat(A[None], self.S, axis=0), np.repeat(B[None], self.S, axis=0)
        As, Bs = adapter.sample_models(self.S - 1, self.rng, scale=self.scale_uncertainty)
        return np.concatenate([A[None], As]), np.concatenate([B[None], Bs])

    def _cost(self, A, B, z0, U, xref):
        """U: (K, N, nu) -> cost (S, K) of every control sequence under every model sample."""
        S, K = A.shape[0], U.shape[0]
        At, Bt, Ct = A.transpose(0, 2, 1), B.transpose(0, 2, 1), self.C.T
        if z0.ndim == 1:
            z = np.broadcast_to(z0, (S, K, z0.shape[0])).copy()
        else:                                                                 # one initial state per model
            z = np.repeat(z0[:, None, :], K, axis=1)
        uRu = np.einsum('kni,ij,knj->k', U, self.R, U)
        cost = np.broadcast_to(uRu, (S, K)).copy()
        for j in range(self.N):
            z = np.matmul(z, At) + np.matmul(U[None, :, j], Bt)              # (S, K, nz)
            x = np.matmul(z, Ct)
            e = x - xref[j + 1]
            W = self.Q if j + 1 < self.N else self.QN
            cost += (np.matmul(e, W) * e).sum(-1)
            if self.floor is not None:
                k, lb = self.floor
                v = np.maximum(lb - x[..., k], 0.0)
                cost += self.floor_w * (v ** 2 + 0.01 * (v > 0))
        return cost

    def _score(self, cost):
        if cost.shape[0] == 1:
            return cost[0]
        mean = cost.mean(axis=0)
        if self.risk == 'mean':
            return mean
        q = np.quantile(cost, 1 - self.alpha, axis=0)                     # CVaR over model samples
        tail = np.where(cost >= q[None], cost, np.nan)
        cvar = np.nanmean(tail, axis=0)
        return mean + self.risk_w * (cvar - mean)

    def _noise(self, K):
        N, nu = self.N, 2
        if not self.knots:
            return self.rng.standard_normal((K, N, nu)) * self.sigma
        if getattr(self, '_W', None) is None:                      # linear-interpolation matrix (N x knots)
            t_k = np.linspace(0, N - 1, self.knots)
            self._W = np.stack([np.interp(np.arange(N), t_k, np.eye(self.knots)[j]) for j in range(self.knots)], 1)
        kn = self.rng.standard_normal((K, self.knots, nu)) * self.sigma
        return np.einsum('nk,ikd->ind', self._W, kn)

    def plan(self, adapter, z0, xref, u_hover, U_seed=None):
        N, nu = self.N, 2
        if self.U is None:
            self.U = np.tile(u_hover, (N, 1))
        base = U_seed if (self.seed_qp and U_seed is not None) else self.U
        A, B = self._models(adapter)
        z_init = z0
        if self.state_samples and A.shape[0] > 1 and hasattr(adapter, 'r_noise'):
            # belief over the current lifted state: measurement noise ~ r_noise * Sigma (adapter's units)
            sd = np.sqrt(max(adapter.r_noise, 0.0) * np.diag(adapter.Sigma))
            z_init = np.repeat(z0[None], A.shape[0], axis=0)
            z_init[1:, 1:] += self.rng.standard_normal((A.shape[0] - 1, len(sd))) * sd
        for it in range(self.iters):
            eps = self._noise(self.K)
            eps[0] = 0.0                                                    # the base plan itself is a candidate
            U = np.clip(base[None] + eps, -self.umax, self.umax)
            score = self._score(self._cost(A, B, z_init, U, xref))
            c = score - score.min()
            lam = self.temp * (np.median(c) + 1e-12)
            w = np.exp(-c / lam)
            w /= w.sum()
            base = np.einsum('k,kij->ij', w, U)
        self.U = base
        if U_seed is not None:
            self.log['dev'].append(float(np.abs(self.U[0] - U_seed[0]).max()))
        if A.shape[0] > 1:
            # spread of the predicted altitude at the end of the horizon across model samples (diagnostic)
            z = np.broadcast_to(z0, (A.shape[0], z0.shape[0])).copy()
            for j in range(N):
                z = np.einsum('sij,sj->si', A, z) + np.einsum('sij,j->si', B, self.U[j])
            self.log['spread'].append(float(np.std(np.dot(z, self.C.T)[:, 1])))
        u0 = self.U[0].copy()
        self.U = np.vstack([self.U[1:], self.U[-1:]])                       # warm start: shift
        return u0

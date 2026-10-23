# Online adaptation of the Koopman matrices [A B].
#   NoAdapt      - nominal model
#   AuthorsNN    - the authors' AdaptNet_linear, called exactly as in their notebook
#   RLS          - recursive least squares with a fixed forgetting factor (standard baseline)
#   BayesKoopman - ours: matrix-normal Kalman filter on [A B] with a mean-reverting prior,
#                  innovation-based change detection and noise-level estimation, and a
#                  predictive covariance that the MPC can use
import numpy as np
from scipy.stats import chi2


class NoAdapt:
    name = 'Nominal'

    def __init__(self, A, B):
        self.A, self.B = A.copy(), B.copy()

    def observe(self, z_prev, u_prev, z_now):
        pass


class AuthorsNN:
    """Faithful port of the notebook loop: retrain a zero-initialised linear net on the last
    `window` (z_pred, u, dz) samples every step and ADD its output to the running A, B."""
    name = 'NN adapt (authors)'

    def __init__(self, A, B, window=10, lr=1.5e-4, epochs=10, l1=1e-2, l2=1e-1):
        import torch
        from core.adapt_net_linear import AdaptNet_linear
        torch.manual_seed(0)
        self.AdaptNet = AdaptNet_linear
        self.A, self.B = A.copy(), B.copy()
        self.w = window
        self.params = dict(state_dim=6, ctrl_dim=B.shape[1], lift_dim=A.shape[0] - 7, optimizer='adam', lr=lr,
                           epochs=epochs, batch_size=window, l2_reg=l2, l1_reg=l1, first_obs_const=True,
                           override_C=True, warm_start=False,
                           del_A_prev=np.zeros_like(A), del_B_prev=np.zeros_like(B))
        self.z_pred_hist, self.u_hist, self.dz_hist = [], [], []
        self.z_pred_next = None

    def observe(self, z_prev, u_prev, z_now):
        # authors: z_pred[i] is the MPC's one-step prediction made at i-1 (= A z_obs + B u with the model at i-1)
        if self.z_pred_next is None:
            self.z_pred_hist.append(z_prev.copy())          # z_pred[0] = lift(x0)
        z_pred_now = np.dot(self.A, z_prev) + np.dot(self.B, u_prev)
        self.u_hist.append(u_prev.copy())
        self.dz_hist.append(z_now - z_pred_now)             # dz[i] = z_obs[i] - z_pred[i]
        self.z_pred_hist.append(z_pred_now)
        i = len(self.z_pred_hist) - 1
        if i >= self.w + 1:
            Zold = np.array(self.z_pred_hist[i - (self.w + 1):i - 1]).T
            Uold = np.array(self.u_hist[i - (self.w + 1):i - 1]).T
            dZ = np.array(self.dz_hist[i - self.w - 1:i - 1]).T   # dz[i-w .. i-1]; dz_hist[k-1] = dz[k]
            net = self.AdaptNet(self.params)
            net.model_pipeline(Zold, Uold, dZ, print_epoch=False)
            dA, dB = net.get_del_matrices()
            self.A = self.A + dA
            self.B = self.B + dB
        self.z_pred_next = True


class _MatrixNormalBase:
    """Shared bookkeeping: Theta = [A B] rows 1..nz-1 are estimated (row 0 = constant observable is fixed).

    Parameters live in centred, normalised regressor coordinates phi_c = T [z; u],
    T = S^-1 (I - mu e0'), with mu, S the training-data mean/std of [z; u] (mu_0 = 0, S_0 = 1 for the
    constant observable). Centring matters: in closed loop the thrust sits near hover, so an offset
    (e.g. wind) is almost collinear with the thrust column. In centred coordinates the minimum-norm
    explanation of an offset is the affine column, not a change in control effectiveness."""

    def __init__(self, A, B, scale, mean=None):
        self.nz, self.nu = B.shape
        self.d = self.nz + self.nu
        mean = np.zeros(self.d) if mean is None else np.array(mean, float)
        mean[0] = 0.0
        e0 = np.zeros(self.d); e0[0] = 1.0
        self.T = np.dot(np.diag(1.0 / scale), np.eye(self.d) - np.outer(mean, e0))
        self.Tinv = np.dot(np.eye(self.d) + np.outer(mean, e0), np.diag(scale))
        self.Theta_nom = np.dot(np.hstack([A, B]), self.Tinv)
        self.Theta = self.Theta_nom.copy()
        self.scale = scale

    def _raw(self):
        return np.dot(self.Theta, self.T)

    @property
    def A(self):
        return self._raw()[:, :self.nz]

    @property
    def B(self):
        return self._raw()[:, self.nz:]

    def _phi(self, z, u):
        return np.dot(self.T, np.concatenate([z, u]))


class RLS(_MatrixNormalBase):
    name = 'RLS (forgetting)'

    def __init__(self, A, B, scale, mean=None, lam=0.99, p0=1e-2):
        super().__init__(A, B, scale, mean)
        self.lam = lam
        self.P = p0 * np.eye(self.d)

    def observe(self, z_prev, u_prev, z_now):
        phi = self._phi(z_prev, u_prev)
        e = z_now - np.dot(self.Theta, phi)
        e[0] = 0.0
        self.P = self.P / self.lam
        Pphi = np.dot(self.P, phi)
        k = Pphi / (1.0 + np.dot(phi, Pphi))
        self.Theta += np.outer(e, k)
        self.P -= np.outer(k, Pphi)
        self.P = 0.5 * (self.P + self.P.T)


class BayesKoopman:
    """Matrix-normal Kalman filter on Theta = [A B] (rows 1..nz-1; row 0 is the constant observable).

    Process model (mean-reverting random walk toward the nominal model):
        Theta_{k+1} = Theta_nom + rho (Theta_k - Theta_nom) + W,   W ~ MN(0, Sigma, P0 / tau)
    with rho^2 = 1 - 1/tau, so the stationary covariance is the prior P0 and tau (steps) is the memory
    length. Unlike a forgetting factor this cannot wind up: directions the closed loop never excites
    relax back to the nominal model instead of growing without bound.

    Measurement: z_k = Theta phi_{k-1} + v,  v ~ N(0, r Sigma),  phi = [z; u],
    Sigma = residual variances of the nominal model on its own training data (offline, no tuning).
    Noise and process covariances share Sigma, so all rows share one d x d column covariance P (an
    RLS-sized update) and the predictive law of Theta phi is N(Theta phi, (phi' P phi + r) Sigma).

    Prior shape: P0 is specified in centred, normalised regressor coordinates
        phi_c = S^-1 (phi - mu_k phi_0),  P0_c = p0 diag(1 [affine], a_frac [z cols], b_frac [u cols])
    where S = training-data stds and mu_k is a slow EWMA of the regressor, i.e. centred at the CURRENT
    operating point. In closed loop the thrust sits near hover, so an offset (wind) is almost collinear
    with the thrust column; centring at the operating point makes the minimum-norm explanation of an
    offset the affine column, and lets B be identified from variations around the operating point.
    The filter itself is coordinate-free (Theta, P are kept in raw coordinates); only the prior/process
    noise shape follows the operating point: P0_raw = T' P0_c T.

    Change detection: nu = e' Sigma^-1 e / ((phi'P phi + r) n) ~ 1 under the model. A fast EWMA of nu is
    compared with a chi-square threshold (false-alarm rate alpha); on detection P += P0_raw ("stop
    trusting what was learned"), which restores fast adaptation after an abrupt change.
    Defaults a_frac=0, b_frac=0.01: offsets are learned fast and control effectiveness slowly. In closed
    loop B is weakly excited, and a fast-moving B estimate is what destabilised the other adapters.

    Measurement noise: r is estimated from DIFFERENCED innovations (white noise survives differencing, a
    slowly varying model mismatch does not), inflated by r_safety for coloured noise / noisy regressors,
    floored at r_min, after a warm-up during which the model is not updated. Under heavy noise r grows
    and the adapter falls back toward the nominal model instead of chasing noise.
    """
    name = 'Bayesian (ours)'

    def __init__(self, A, B, scale, res_cov, mean=None, p0=10.0, a_frac=0.0, b_frac=0.01, tau=500, tau_c=100,
                 alpha=1e-3, fast=0.8, slow=0.995, detect=True, track_noise=True, cooldown=20, r_min=1.0,
                 warmup=10, r_safety=3.0, scales=((0.8, 4.5), (0.95, 2.6), (0.99, 1.85))):
        self.nz, self.nu = B.shape
        self.d = self.nz + self.nu
        n = self.nz - 1
        self.Theta_nom = np.hstack([A, B])
        self.Theta = self.Theta_nom.copy()
        self.scale = np.array(scale, float)
        self.mu = np.zeros(self.d) if mean is None else np.array(mean, float)
        self.mu[0] = 0.0
        self.beta_c = 1.0 - 1.0 / tau_c
        # The full residual covariance spans ~5 orders of magnitude (correlated encoder features), so its
        # inverse is dominated by fragile directions. Use the diagonal, floored at the median variance.
        var = np.diag(res_cov)
        self.Sigma = np.diag(np.maximum(var, np.median(var)))
        self.Sigma_inv = np.linalg.inv(self.Sigma)
        self.Sigma_full = np.zeros((self.nz, self.nz)); self.Sigma_full[1:, 1:] = self.Sigma
        w = np.full(self.d, float(a_frac)); w[0] = 1.0; w[self.nz:] = b_frac
        self.P0c = p0 * np.diag(w)
        self.tau = tau
        self.rho = np.sqrt(1.0 - 1.0 / tau)
        self.P = self._prior()
        self.r_min = r_min
        self.r = r_min
        self.fast, self.slow = fast, slow
        self.nu_fast = 1.0
        # threshold for the fast EWMA of nu (mean ~1 under the model); effective dof ~ n (1+fast)/(1-fast)
        dof = n * (1 + fast) / (1 - fast)
        self.thresh = chi2.ppf(1 - alpha, dof) / dof
        self.detect, self.track_noise = detect, track_noise
        self.cooldown, self.cool = cooldown, 0
        self.warmup, self.k = warmup, 0
        self.cal = 1.0
        # multi-scale detector: EWMAs of nu at several time scales, each with a threshold calibrated as
        # 1.5x the largest value seen in no-change runs on held-out trajectories (circle, S) at 60-30 dB.
        # (A chi-square threshold assumes white innovations; with noisy lifted regressors they are
        # correlated and heavy-tailed, which produced false alarms.) scales=None -> single chi-square test.
        self.scales = scales
        self.base = 0.3
        self.ewma = [self.base] * (len(scales) if scales else 0)
        self.r_safety = r_safety
        self.e_prev = None
        self.r_noise = r_min
        self.log = dict(nu=[], kick=[], r=[], trP=[])

    @property
    def A(self):
        return self.Theta[:, :self.nz]

    @property
    def B(self):
        return self.Theta[:, self.nz:]

    def _phi(self, z, u):
        return np.concatenate([z, u])

    def _prior(self):
        e0 = np.zeros(self.d); e0[0] = 1.0
        T = np.dot(np.diag(1.0 / self.scale), np.eye(self.d) - np.outer(self.mu, e0))
        return np.dot(T.T, np.dot(self.P0c, T))

    def observe(self, z_prev, u_prev, z_now):
        n = self.nz - 1
        phi = self._phi(z_prev, u_prev)
        self.mu[1:] = self.beta_c * self.mu[1:] + (1 - self.beta_c) * phi[1:]
        P0 = self._prior()
        # time update (mean-reverting)
        self.Theta = self.Theta_nom + self.rho * (self.Theta - self.Theta_nom)
        self.P = self.rho ** 2 * self.P + P0 / self.tau
        e = (z_now - np.dot(self.Theta, phi))[1:]
        self.k += 1
        # measurement-noise level from DIFFERENCED innovations: a parameter change gives a slowly varying
        # innovation and cancels in e_k - e_{k-1}, white measurement noise does not. So r tracks noise,
        # not model mismatch, and does not slow adaptation down right after a change.
        if self.track_noise and self.e_prev is not None:
            de = e - self.e_prev
            r_obs = 0.5 * float(np.dot(de, np.dot(self.Sigma_inv, de))) / n
            beta = max(self.slow, 1.0 - 1.0 / self.k)          # running mean at start, then slow EWMA
            self.r_noise = beta * self.r_noise + (1 - beta) * r_obs
            # safety factor: innovations are coloured and the regressor itself is noisy (errors-in-variables),
            # so the white-noise Kalman model is over-confident; r_safety inflates the noise level for that
            self.r = max(self.r_safety * self.r_noise, self.r_min)
        self.e_prev = e
        Pphi = np.dot(self.P, phi)
        s = np.dot(phi, Pphi) + self.r
        power = float(np.dot(e, np.dot(self.Sigma_inv, e))) / n
        nu = power / s
        self.nu_fast = self.fast * self.nu_fast + (1 - self.fast) * nu
        self.cal = 0.9 * self.cal + 0.1 * nu if self.k > 1 else nu
        if self.scales and self.k > self.warmup:      # the noise estimate is still settling during warm-up
            self.ewma = [b * v + (1 - b) * nu for v, (b, th) in zip(self.ewma, self.scales)]
        kick = False
        if self.k <= self.warmup:
            # warm-up: learn the noise level first, do not touch the model yet
            self.s_par = 1e6
            self.log['nu'].append(nu); self.log['kick'].append(False)
            self.log['r'].append(self.r); self.log['trP'].append(np.trace(self.P))
            return
        if self.cool > 0:
            self.cool -= 1
        elif self.detect and self._fired(nu):
            self.P = self.P + P0
            Pphi = np.dot(self.P, phi)
            s = np.dot(phi, Pphi) + self.r
            self.cool = self.cooldown
            kick = True
            self.ewma = [self.base] * len(self.ewma)
        k = Pphi / s
        self.Theta[1:] += np.outer(e, k)
        self.P -= np.outer(k, Pphi)
        self.P = 0.5 * (self.P + self.P.T)
        self.s_par = float(np.dot(phi, np.dot(self.P, phi)))   # parameter part of the predictive variance
        self.log['nu'].append(nu); self.log['kick'].append(kick)
        self.log['r'].append(self.r); self.log['trP'].append(np.trace(self.P))

    def sample_models(self, n, rng, scale=True):
        """n samples of [A B] from the matrix-normal posterior MN(Theta, Sigma, P).

        scale=True calibrates the spread: the posterior's absolute scale comes from the nominal model's
        training residuals (Sigma), which is far too small for the errors seen right after a change. The
        normalised innovation nu should average 1 when the posterior is calibrated, so its recent average
        (cal) multiplies the row covariance: wide after a change, narrow once the model has caught up.
        """
        L_S = np.sqrt(np.diag(self.Sigma))                       # rows 1..nz-1
        w, V = np.linalg.eigh(0.5 * (self.P + self.P.T))
        L_P = V * np.sqrt(np.maximum(w, 0.0))
        c = max(self.cal, 1e-6) if scale else 1.0
        Z = rng.standard_normal((n, self.nz - 1, self.d))
        dTh = np.sqrt(c) * L_S[None, :, None] * np.matmul(Z, L_P.T)
        Th = np.repeat(self.Theta[None], n, axis=0)
        Th[:, 1:] += dTh
        return Th[:, :, :self.nz], Th[:, :, self.nz:]

    def _fired(self, nu):
        if not self.scales:
            return self.nu_fast > self.thresh
        return any(v > th for v, (b, th) in zip(self.ewma, self.scales))

    def predictive_std(self, Z, U, C):
        """Std of C z_j along a planned trajectory (Z: N+1 x nz, U: N x nu) from parameter + noise
        uncertainty, propagated through the posterior-mean model. Returns (N+1, nx)."""
        A = self.A
        cov = np.zeros((self.nz, self.nz))
        out = np.zeros((len(Z), C.shape[0]))
        for j in range(len(U)):
            phi = self._phi(Z[j], U[j])
            cov = np.dot(A, np.dot(cov, A.T)) + (np.dot(phi, np.dot(self.P, phi)) + self.r) * self.Sigma_full
            out[j + 1] = np.sqrt(np.maximum(np.diag(np.dot(C, np.dot(cov, C.T))), 0))
        return out


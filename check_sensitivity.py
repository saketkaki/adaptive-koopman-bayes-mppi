# How sensitive is the authors' adapter to tiny numerical differences in the residuals?
# Perturb the observed lifted state seen by the adapter only (not the controller) by ~1e-7.
import sys
import numpy as np
from koopman_sim import KoopModel, run_closed_loop, constant_schedule, rmse
from adapters import AuthorsNN

wind = float(sys.argv[1]); seed = int(sys.argv[2]); eps = float(sys.argv[3])
m = KoopModel()


class Perturbed(AuthorsNN):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.rng = np.random.default_rng(seed)

    def observe(self, z_prev, u_prev, z_now):
        z_now = z_now + eps * self.rng.standard_normal(z_now.shape) * (np.arange(len(z_now)) > 0)
        super().observe(z_prev, u_prev, z_now)


r = run_closed_loop(m, Perturbed(m.A, m.B), constant_schedule(m=2.4, Ixx=1.2, V_w=wind))
print(f'wind={wind} seed={seed} eps={eps:g}: RMSE={rmse(r):.4f}')

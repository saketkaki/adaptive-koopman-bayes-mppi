# Check that our MPC + ported authors' adapter reproduce the authors' own pipeline numbers.
import sys, time
import numpy as np
from koopman_sim import KoopModel, run_closed_loop, constant_schedule, rmse
from adapters import NoAdapt, AuthorsNN

wind = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
m = KoopModel()
sched = constant_schedule(m=2.4, Ixx=1.2, V_w=wind)
ref = np.load(f'results/authors_lemniscate_w{wind:g}.npz')
for name, ad in [('nominal', NoAdapt(m.A, m.B)), ('nn_adapt', AuthorsNN(m.A, m.B))]:
    t = time.time()
    r = run_closed_loop(m, ad, sched)
    dx = np.nanmax(np.abs(r['x'] - ref[name]))
    print(f'wind={wind}: {name:9s} ours RMSE={rmse(r):.4f}  max |x_ours - x_authors| = {dx:.2e}  ({time.time()-t:.0f}s)')

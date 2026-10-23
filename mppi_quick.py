# Single MPPI runs: python mppi_quick.py <adapter> <scenario> [key=value ...]
#   MPPI keys: K S N sigF sigM temp iters risk alpha risk_w floor scale ; run keys: snr_db seed sched
import sys, time
import numpy as np
from koopman_sim import KoopModel, run_closed_loop, rmse
from quick import make, SCENARIOS, SCHED, FLOOR
from mppi import KoopMPPI

adapter_name, scen = sys.argv[1], sys.argv[2]
kv = {}
for a in sys.argv[3:]:
    k, v = a.split('=')
    try:
        v = float(v) if any(c in v for c in '.e') else int(v)
    except ValueError:
        pass
    kv[k] = v
m = KoopModel(); st = dict(np.load('results/train_stats.npz'))
ad = make(adapter_name, m, st)
floor = (1, kv.get('floor', FLOOR)) if scen == 'floor' else None
ctrl = KoopMPPI(m.C, N=int(kv.get('N', 15)), K=int(kv.get('K', 256)), n_models=int(kv.get('S', 1)),
                sigma=(kv.get('sigF', 2.0), kv.get('sigM', 0.15)), temp=kv.get('temp', 0.05),
                iters=int(kv.get('iters', 2)), risk=kv.get('risk', 'cvar'), alpha=kv.get('alpha', 0.25),
                risk_w=kv.get('risk_w', 1.0), floor=floor, scale_uncertainty=bool(kv.get('scale', 1)),
                seed=int(kv.get('seed', 0)), state_samples=bool(kv.get('xs', 0)),
                fixed_Q=bool(kv.get('aggr', 0)),
                Q=(1.5e5 * np.diag([1, 1, kv['aggr'], 1, 1, 0.5 * kv['aggr']]) if kv.get('aggr') else None))
run_kw = dict(snr_db=kv.get('snr_db'), seed=int(kv.get('seed', 0)), mppi=ctrl, floor=floor)
if kv.get('sched'):
    run_kw['att_sched'] = SCHED
t = time.time()
r = run_closed_loop(m, ad, SCENARIOS[scen], **run_kw)
extra = ''
if scen == 'floor':
    below = np.maximum(floor[1] - r['x'][:, 1], 0)
    extra += f"  viol depth={1e3*np.nanmax(below):.1f}mm time={0.01*np.sum(below > 0):.2f}s"
if scen == 'jumps':
    extra += '  win ' + ' '.join(f'{100*rmse(r, a, a + 1):.1f}' for a in (3, 6.5, 9.5))
if ctrl.log['spread']:
    sp = np.array(ctrl.log['spread'])
    extra += f"  spread(mm) med={1e3*np.median(sp):.2f} max={1e3*sp.max():.2f}"
print(f"MPPI {adapter_name} {scen} {sys.argv[3:]}: RMSE={100*rmse(r):.2f}cm failed={r['failed']} "
      f"ctrl_ms={1e3*np.mean(r['ctrl_time']):.1f}{extra} ({time.time()-t:.0f}s)")

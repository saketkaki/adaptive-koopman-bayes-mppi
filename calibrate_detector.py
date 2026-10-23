# Distribution of the detector statistic when NOTHING changes (detection off), per noise level and per
# EWMA time scale. Used to set thresholds that keep false alarms rare under noise.
import sys, json
import numpy as np
from koopman_sim import KoopModel, run_closed_loop
from quick import make, SCENARIOS

SCALES = [0.8, 0.95, 0.99]
m = KoopModel(); st = dict(np.load('results/train_stats.npz'))
snr = None if sys.argv[1] == 'none' else float(sys.argv[1])
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
traj = sys.argv[3] if len(sys.argv) > 3 else 'lemniscate'
ad = make('bayes', m, st, detect=False)
r = run_closed_loop(m, ad, SCENARIOS['none'], traj=traj, snr_db=snr, seed=seed)
nu = np.array(ad.log['nu'][15:])
out = {}
for b in SCALES:
    f = np.zeros_like(nu); acc = 1.0
    for i, v in enumerate(nu):
        acc = b * acc + (1 - b) * v; f[i] = acc
    out[str(b)] = dict(p99=float(np.percentile(f, 99)), p999=float(np.percentile(f, 99.9)), max=float(f.max()))
print(json.dumps(dict(snr=snr, seed=seed, traj=traj, failed=r['failed'], nu_mean=float(nu.mean()), scales=out)))

# Altitude-floor experiment: payload (+25%) picked up 0.3 s before the lowest point of the lemniscate,
# floor 1.6 cm below the reference minimum. All methods use our Bayesian adapter (except 'qp_nominal').
# Results: results/floor_runs.jsonl (resumable).
import os, json, time, itertools
from multiprocessing import Pool
import numpy as np

OUT = 'results/floor_runs.jsonl'
FLOOR = -1.14
METHODS = ['qp_nominal', 'qp', 'qp_t1', 'qp_t2', 'mppi_mean', 'mppi_post', 'mppi_belief', 'mppi_stateonly']


def jobs():
    out = []
    for meth in METHODS:
        out.append(dict(method=meth, snr=None, seed=0))
        for snr, seed in itertools.product([40, 30], range(10)):
            out.append(dict(method=meth, snr=snr, seed=seed))
    for j in out:
        j['id'] = f"{j['method']}|{j['snr']}|{j['seed']}"
    return out


def run(job):
    import warnings; warnings.filterwarnings('ignore')
    from koopman_sim import KoopModel, run_closed_loop, rmse
    from quick import make, SCENARIOS
    from mppi import KoopMPPI
    m = KoopModel(); st = dict(np.load('results/train_stats.npz'))
    meth = job['method']
    ad = make('nominal' if meth == 'qp_nominal' else 'bayes', m, st)
    kw = dict(snr_db=job['snr'], seed=job['seed'], floor=(1, FLOOR))
    if meth == 'qp_t1':
        kw['tighten_noise'] = 1.0
    if meth == 'qp_t2':
        kw['tighten_noise'] = 2.0
    if meth.startswith('mppi'):
        S = 1 if meth == 'mppi_mean' else 8
        kw['mppi'] = KoopMPPI(m.C, K=256 if S == 1 else 128, n_models=S, floor=(1, FLOOR), seed=job['seed'],
                              state_samples=(meth in ('mppi_belief', 'mppi_stateonly')),
                              sample_models=(meth != 'mppi_stateonly'))
    t = time.time()
    r = run_closed_loop(m, ad, SCENARIOS['floor'], **kw)
    below = np.maximum(FLOOR - r['x'][:, 1], 0)
    row = dict(job, rmse=rmse(r), failed=bool(r['failed']), depth_mm=float(1e3 * np.nanmax(below)),
               time_below_s=float(0.01 * np.sum(below > 0)), ctrl_ms=float(1e3 * np.mean(r['ctrl_time'])),
               wall=time.time() - t)
    for k, v in list(row.items()):
        if isinstance(v, float) and not np.isfinite(v):
            row[k] = None
    return row


if __name__ == '__main__':
    done = set()
    if os.path.exists(OUT):
        done = {json.loads(l)['id'] for l in open(OUT)}
    todo = [j for j in jobs() if j['id'] not in done]
    print(f'{len(todo)} runs', flush=True)
    t0 = time.time()
    with Pool(2) as pool, open(OUT, 'a') as f:
        for i, row in enumerate(pool.imap_unordered(run, todo)):
            f.write(json.dumps(row) + '\n'); f.flush()
            print(f"[{i+1}/{len(todo)} {time.time()-t0:.0f}s] {row['id']} rmse={row['rmse']} depth={row['depth_mm']:.1f}", flush=True)
    print('ALL DONE', flush=True)

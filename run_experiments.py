# Full experiment grid (v2). One JSON line per run in results/runs_v2.jsonl; resumable.
# Baseline runs whose code path did not change are copied from results/runs.jsonl (v1 grid).
# python run_experiments.py [group ...]   groups: main jumps wind ctrl abl sens
import os, sys, json, time, itertools
from multiprocessing import Pool
import numpy as np

OUT = 'results/runs_v2.jsonl'
OLD = 'results/runs.jsonl'
AGG = dict(Q='agg')   # fixed aggressive controller: attitude weight 0.03
SCHED = 'sched'       # confidence-scheduled controller (ours)

# name -> (adapter kind, adapter kwargs, controller)
METHODS = {
    'nominal': ('nominal', {}, None),
    'nn': ('nn', {}, None),
    'rls_fast': ('rls', dict(lam=0.95, p0=1e-4), None),
    'rls_slow': ('rls', dict(lam=0.999, p0=1e-2), None),
    'bayes_v1': ('bayes', dict(scales=None, warmup=100), None),       # first delivered version
    'bayes_c1': ('bayes', {}, None),                                  # v2 adapter, authors' controller
    'bayes': ('bayes', {}, SCHED),                                    # v2 adapter + scheduled controller
    # controller swap: fixed aggressive attitude weight for everyone
    'nominal_agg': ('nominal', {}, 'agg'),
    'nn_agg': ('nn', {}, 'agg'),
    'bayes_agg': ('bayes', {}, 'agg'),
    # component ablations of the full method
    'abl_nodetect': ('bayes', dict(detect=False), SCHED),
    'abl_chi2': ('bayes', dict(scales=None), SCHED),
    'abl_flatprior': ('bayes', dict(b_frac=1.0), SCHED),
    'abl_nonoise': ('bayes', dict(track_noise=False), SCHED),
}
BASE = ['nominal', 'nn', 'rls_fast', 'rls_slow']
OURS = ['bayes_v1', 'bayes_c1', 'bayes']
TRAJS = ['lemniscate', 'circle', 'S', 'adobe']


def J(**kw):
    kw.setdefault('seed', 0)
    return kw


def jobs(groups):
    out = []
    if 'main' in groups:
        for traj, meth in itertools.product(TRAJS, BASE + OURS):
            out.append(J(group='main', scen='mass', traj=traj, method=meth, snr=None, save=traj == 'lemniscate'))
            for seed in range(3):
                out.append(J(group='main', scen='mass', traj=traj, method=meth, snr=40, seed=seed))
    if 'jumps' in groups:
        for meth in BASE + OURS:
            out.append(J(group='jumps', scen='jumps', traj='lemniscate', method=meth, snr=None, save=True))
            for snr, seed in itertools.product([50, 40, 30], range(3)):
                out.append(J(group='jumps', scen='jumps', traj='lemniscate', method=meth, snr=snr, seed=seed,
                             save=(snr == 40 and seed == 0)))
    if 'wind' in groups:
        for meth in BASE + OURS:
            for traj in TRAJS:
                out.append(J(group='wind', scen='wind', traj=traj, method=meth, snr=None, save=traj == 'lemniscate'))
            for snr, seed in itertools.product([50, 40], range(5)):
                out.append(J(group='wind', scen='wind', traj='lemniscate', method=meth, snr=snr, seed=seed))
    if 'ctrl' in groups:
        for meth, scen in itertools.product(['nominal_agg', 'nn_agg', 'bayes_agg'], ['mass', 'jumps', 'wind']):
            out.append(J(group='ctrl', scen=scen, traj='lemniscate', method=meth, snr=None))
            for seed in range(3):
                out.append(J(group='ctrl', scen=scen, traj='lemniscate', method=meth, snr=40, seed=seed))
    if 'abl' in groups:
        for meth, scen in itertools.product(['abl_nodetect', 'abl_chi2', 'abl_flatprior', 'abl_nonoise'], ['jumps', 'wind']):
            out.append(J(group='abl', scen=scen, traj='lemniscate', method=meth, snr=None))
            for seed in range(2):
                out.append(J(group='abl', scen=scen, traj='lemniscate', method=meth, snr=40, seed=seed))
    if 'sens' in groups:
        for lr, w in itertools.product([5e-5, 1.5e-4, 5e-4], [5, 10, 20]):
            for snr in [None, 40]:
                out.append(J(group='sens', scen='jumps', traj='lemniscate', method='nn', kw=dict(lr=lr, window=w), snr=snr))
        for p0, tau in itertools.product([1.0, 10.0, 100.0], [200, 500, 2000]):
            for snr in [None, 40]:
                out.append(J(group='sens', scen='jumps', traj='lemniscate', method='bayes', kw=dict(p0=p0, tau=tau), snr=snr))
    for j in out:
        j['id'] = '|'.join(str(j.get(k)) for k in ('group', 'scen', 'traj', 'method', 'snr', 'seed')) + \
                  ('|' + json.dumps(j['kw'], sort_keys=True) if 'kw' in j else '')
    return out


def run_one(job):
    import warnings; warnings.filterwarnings('ignore')
    from koopman_sim import KoopModel, run_closed_loop, rmse
    from quick import make, SCENARIOS, SCHED as SCHED_PARAMS
    m = KoopModel(); st = dict(np.load('results/train_stats.npz'))
    kind, kw, ctrl = METHODS[job['method']]
    kw = dict(kw, **job.get('kw', {}))
    ad = make(kind, m, st, **kw)
    run_kw = {}
    if ctrl == SCHED:
        run_kw['att_sched'] = SCHED_PARAMS
    elif ctrl == 'agg':
        run_kw['Q'] = 1.5e5 * np.diag([1, 1, 0.03, 1, 1, 0.015])
    t = time.time()
    r = run_closed_loop(m, ad, SCENARIOS[job['scen']], traj=job['traj'], snr_db=job['snr'], seed=job['seed'], **run_kw)
    row = dict(job, rmse=rmse(r), failed=bool(r['failed']), qp_fail=len(r['qp_fail_steps']),
               adapt_ms=float(1e3 * np.mean(r['adapt_time'])), wall=time.time() - t,
               min_z=float(np.nanmin(r['x'][:, 1])))
    if ctrl == SCHED:
        row['w_att_median'] = float(np.median(r['w_att']))
        row['frac_aggressive'] = float(np.mean(r['w_att'] < 0.1))
    if job['scen'] == 'jumps':
        row['win'] = [rmse(r, a, a + 1) for a in (3.0, 6.5, 9.5)]
    if hasattr(ad, 'log'):
        row['kick_steps'] = [int(i) for i in np.nonzero(ad.log['kick'])[0]]
        row['r_end'] = float(ad.r)
    if job.get('save'):
        fn = 'results/traj/' + job['id'].replace('|', '_').replace(' ', '') + '.npz'
        extra = {}
        if hasattr(ad, 'log'):
            extra = dict(nu=np.array(ad.log['nu']), kick=np.array(ad.log['kick']), trP=np.array(ad.log['trP']))
        if ctrl == SCHED:
            extra['w_att'] = r['w_att']
        np.savez(fn, x=r['x'], u=r['u'], xref=r['xref'], **extra)
        row['file'] = fn
    for k, v in list(row.items()):
        if isinstance(v, float) and not np.isfinite(v):
            row[k] = None
    return row


if __name__ == '__main__':
    groups = sys.argv[1:] or ['jumps', 'wind', 'main', 'ctrl', 'abl', 'sens']
    os.makedirs('results/traj', exist_ok=True)
    all_jobs = jobs(groups)
    done = set()
    if os.path.exists(OUT):
        done = {json.loads(l)['id'] for l in open(OUT)}
    # reuse unchanged baseline runs from the v1 grid (same code path, same ids)
    if os.path.exists(OLD):
        reuse = {}
        for l in open(OLD):
            r = json.loads(l)
            if r['method'] in BASE:
                reuse[r['id']] = r
        with open(OUT, 'a') as f:
            for j in all_jobs:
                if j['id'] in reuse and j['id'] not in done:
                    f.write(json.dumps(dict(reuse[j['id']], reused=True)) + '\n'); done.add(j['id'])
    todo = [j for j in all_jobs if j['id'] not in done]
    todo.sort(key=lambda j: groups.index(j['group']))
    print(f'{len(todo)} runs to do ({len(done)} done or reused)', flush=True)
    t0 = time.time()
    with Pool(2) as pool, open(OUT, 'a') as f:
        for i, row in enumerate(pool.imap_unordered(run_one, todo)):
            f.write(json.dumps(row) + '\n'); f.flush()
            print(f"[{i+1}/{len(todo)} {time.time()-t0:.0f}s] {row['id']}: rmse={row['rmse']} failed={row['failed']}", flush=True)
    print('ALL DONE', flush=True)

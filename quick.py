# Quick single runs for tuning/debugging.
# python quick.py <method> <scenario> [key=value ...]
#   method: nominal | nn | rls | bayes      scenario: mass | wind | jumps
import sys, time
import numpy as np
from koopman_sim import KoopModel, run_closed_loop, constant_schedule, jump_schedule, rmse
from adapters import NoAdapt, AuthorsNN, RLS, BayesKoopman

SCENARIOS = {
    'none': constant_schedule(),
    'mass': constant_schedule(m=2.4, Ixx=1.2),
    'wind': constant_schedule(m=2.4, Ixx=1.2, V_w=3.0),
    # payload attached at 3 s, wind on at 6 s, payload dropped at 9 s
    # payload (+20% mass/inertia) attached at 3 s, dropped at 6.5 s, attached again at 9.5 s
    # payload (+25%) picked up 0.3 s before the lowest point of the lemniscate (altitude-floor test)
    'floor': jump_schedule([(4.85, dict(m=2.5, Ixx=1.25))]),
    'jumps': jump_schedule([(3.0, dict(m=2.4, Ixx=1.2)), (6.5, dict(m=2.0, Ixx=1.0)), (9.5, dict(m=2.4, Ixx=1.2))]),
}


FLOOR = -1.15
SCHED = (0.03, 1.0, (3.0, 30.0), (10 ** (-62 / 20), 10 ** (-56 / 20)))


def make(method, m, stats, **kw):
    if method == 'nominal':
        return NoAdapt(m.A, m.B)
    if method == 'nn':
        return AuthorsNN(m.A, m.B, **kw)
    if method == 'rls':
        return RLS(m.A, m.B, stats['scale'], stats['mean'], **kw)
    if method == 'bayes':
        return BayesKoopman(m.A, m.B, stats['scale'], stats['res_cov'], stats['mean'], **kw)
    raise ValueError(method)


if __name__ == '__main__':
    method, scen = sys.argv[1], sys.argv[2]
    kw = {}
    run_kw = {}
    for a in sys.argv[3:]:
        k, v = a.split('=')
        v = {'True': True, 'False': False}.get(v, v)
        if isinstance(v, str):
            v = float(v) if any(c in v for c in '.e') else int(v)
        (run_kw if k in ('snr_db', 'seed', 'tighten', 'ki', 'i_max', 'target_fix', 'qn_pos', 'att_w', 'sched', 'tighten_noise') else kw)[k] = v
    m = KoopModel()
    stats = dict(np.load('results/train_stats.npz'))
    ad = make(method, m, stats, **kw)
    t = time.time()
    if scen == 'floor':
        run_kw['floor'] = (1, FLOOR)
    if run_kw.pop('sched', 0):
        run_kw['att_sched'] = SCHED
    if 'att_w' in run_kw:
        w = run_kw.pop('att_w')
        run_kw['Q'] = 1.5e5 * np.diag([1, 1, w, 1, 1, 0.5 * w])
    if 'qn_pos' in run_kw:
        a = run_kw.pop('qn_pos')
        run_kw['QN'] = np.diag([a * 1.5e5, a * 1.5e5, 1e3, 1e3, 1e3, 1e3])
    r = run_closed_loop(m, ad, SCENARIOS[scen], **run_kw)
    seg = ''
    if scen == 'jumps':
        seg = '  windows: ' + ' '.join(f'{rmse(r, a, a + 1):.3f}' for a in (3, 6.5, 9.5))
    extra = ''
    if method == 'bayes':
        extra = f"  kicks={int(np.sum(ad.log['kick']))} r_end={ad.r:.2f}"
    seg += f"  min z={np.nanmin(r['x'][:, 1]):.3f}"
    if scen == 'floor':
        below = np.maximum(FLOOR - r['x'][:, 1], 0)
        seg += f"  viol: depth={1e3*np.nanmax(below):.1f}mm time={0.01*np.sum(below > 0):.2f}s margin_max={1e3*r['margin'].max():.1f}mm"
    if 'att_sched' in run_kw:
        seg += f"  w_att median={np.median(r['w_att']):.3f}"
    print(f"{method} {scen} {kw} {sys.argv[3:]}: RMSE={rmse(r):.4f}{seg} failed={r['failed']} "
          f"qpfail={len(r['qp_fail_steps'])} adapt_ms={1e3*np.mean(r['adapt_time']):.2f} ({time.time()-t:.0f}s){extra}")

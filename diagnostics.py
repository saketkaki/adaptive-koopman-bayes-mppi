# Two diagnostics for the write-up.
# 1) Wind: is tracking error explained by model accuracy? Compare each adapter's 15-step prediction error
#    along its own closed-loop trajectory with its tracking RMSE; plus an oracle model with the exact wind force.
# 2) Uncertainty-aware tightening: how large is the propagated 15-step predictive std right after a detected
#    change, compared with the altitude sag it would need to prevent?
import json
import numpy as np
from koopman_sim import KoopModel, run_closed_loop, rmse, constant_schedule, DT
from quick import make, SCENARIOS
from adapters import NoAdapt

m = KoopModel(); st = dict(np.load('results/train_stats.npz'))
out = {}


class Rec:
    def __init__(s, ad):
        s.ad, s.As, s.Bs = ad, [], []
    A = property(lambda s: s.ad.A)
    B = property(lambda s: s.ad.B)

    def observe(s, *a):
        s.ad.observe(*a); s.As.append(s.ad.A.copy()); s.Bs.append(s.ad.B.copy())

    def __getattr__(s, k):
        return getattr(s.ad, k)


def pred15(rec, r):
    errs = []
    for i in range(50, len(r['u']) - 20, 25):
        A, B = rec.As[i - 1], rec.Bs[i - 1]
        z = m.lift(r['x'][i])
        for j in range(15):
            z = np.dot(A, z) + np.dot(B, r['u'][i + j])
        errs.append(np.linalg.norm(np.dot(m.C, z)[:2] - r['x'][i + 15, :2]))
    return float(np.sqrt(np.mean(np.square(errs))))


print('--- wind (3 m/s + 20% mass), lemniscate')
for name in ['nn', 'bayes']:
    rec = Rec(make(name, m, st))
    r = run_closed_loop(m, rec, SCENARIOS['wind'])
    e = r['x'] - r['xref']
    out[f'wind_{name}'] = dict(rmse=rmse(r), pred15=pred15(rec, r), offset_y=float(np.mean(e[-500:, 0])))
    print(name, out[f'wind_{name}'])
aw = 0.05 * 9 * np.sin(np.pi / 4)
for label, sched in [('wind only', constant_schedule(V_w=3.0))]:
    for name, fix in [('nominal', 0), ('oracle', 1)]:
        ad = NoAdapt(m.A, m.B)
        ad.A[4, 0] += fix * aw * DT; ad.A[5, 0] += fix * aw * DT
        r = run_closed_loop(m, ad, sched)
        out[f'windonly_{name}'] = dict(rmse=rmse(r), offset_y=float(np.mean((r['x'] - r['xref'])[-500:, 0])))
        print(label, name, out[f'windonly_{name}'])

print('--- predictive std right after detected changes (payload jumps, noise-free)')
ad = make('bayes', m, st)
stds = []


class Probe:
    A = property(lambda s: ad.A)
    B = property(lambda s: ad.B)

    def observe(s, zp, up, zn):
        ad.observe(zp, up, zn)
        if ad.log['kick'][-1]:
            Z = np.tile(zn, (16, 1)); U = np.tile(up, (15, 1))
            stds.append(1000 * ad.predictive_std(Z, U, m.C)[-1, 1])


r = run_closed_loop(m, Probe(), SCENARIOS['jumps'])
nom = run_closed_loop(m, NoAdapt(m.A, m.B), SCENARIOS['jumps'])
sag = 1000 * float(np.max(np.abs(nom['x'][300:450, 1] - nom['xref'][300:450, 1])))
out['tighten'] = dict(std15_after_kick_mm=stds, nominal_max_alt_error_after_attach_mm=sag)
print(out['tighten'])
json.dump(out, open('results/diagnostics.json', 'w'), indent=1)

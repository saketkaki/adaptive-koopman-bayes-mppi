# Tables (results/summary.md) and figures (figures/) from results/runs_v2.jsonl
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

rows = [json.loads(l) for l in open('results/runs_v2.jsonl')]
os.makedirs('figures', exist_ok=True)

MAIN = ['nominal', 'nn', 'rls_fast', 'rls_slow', 'bayes_v1', 'bayes_c1', 'bayes']
LABEL = {'nominal': 'Nominal (no adaptation)', 'nn': 'NN adapter (authors)', 'rls_fast': 'RLS, λ=0.95',
         'rls_slow': 'RLS, λ=0.999', 'bayes_v1': 'Bayesian v1 (first version)',
         'bayes_c1': 'Bayesian v2, authors\' controller', 'bayes': 'Bayesian v2 + scheduled controller (ours, full)'}
SHORT = {'nominal': 'Nominal', 'nn': 'NN (authors)', 'rls_fast': 'RLS λ=.95', 'rls_slow': 'RLS λ=.999',
         'bayes_v1': 'Ours v1', 'bayes_c1': 'Ours v2 (C1)', 'bayes': 'Ours (full)'}
COLOR = {'nominal': '#8a8984', 'nn': '#2a78d6', 'bayes': '#eb6834', 'rls_fast': '#1baf7a', 'rls_slow': '#eda100',
         'bayes_v1': '#e87ba4', 'bayes_c1': '#008300'}
MARK = {'nominal': 's', 'nn': 'o', 'rls_fast': 'v', 'rls_slow': '^', 'bayes_v1': 'P', 'bayes_c1': 'X', 'bayes': 'D'}
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.facecolor': SURF, 'figure.facecolor': SURF, 'axes.grid': True,
                     'grid.color': GRID, 'grid.linewidth': 0.8, 'axes.spines.top': False,
                     'axes.spines.right': False, 'legend.frameon': False})


def sel(**kw):
    out = []
    for r in rows:
        ok = True
        for k, v in kw.items():
            if k == 'traj_in':
                ok &= r['traj'] in v
            else:
                ok &= r.get(k) == v
        if ok:
            out.append(r)
    return out


def stat(rs):
    vals = [r['rmse'] for r in rs if not r['failed'] and r['rmse'] is not None]
    return (np.median(vals) if vals else None), sum(r['failed'] for r in rs), len(rs)


def fmt(m, nf, n):
    if n == 0:
        return '—'
    if m is None:
        return f'fail ({nf}/{n})'
    return f'{100*m:.1f}' + (f' ({nf}/{n} fail)' if nf else '')


L = []
P = L.append
P('Position RMSE in cm. Median over runs that did not diverge; "(k/n fail)" = k of n runs diverged.\n')
P('### Main comparison\n')
cols = [('Mass +20%, noise-free, 4 traj', dict(group='main', snr=None)),
        ('Mass +20%, 40 dB, 4 traj × 3', dict(group='main', snr=40)),
        ('Payload jumps, noise-free', dict(group='jumps', snr=None)),
        ('Payload jumps, 50 dB ×3', dict(group='jumps', snr=50)),
        ('Payload jumps, 40 dB ×3', dict(group='jumps', snr=40)),
        ('Payload jumps, 30 dB ×3', dict(group='jumps', snr=30)),
        ('Wind + mass, noise-free, 4 traj', dict(group='wind', snr=None)),
        ('Wind + mass, 50 dB ×5', dict(group='wind', snr=50)),
        ('Wind + mass, 40 dB ×5', dict(group='wind', snr=40))]
P('| Method | ' + ' | '.join(c for c, _ in cols) + ' |')
P('|---|' + '---|' * len(cols))
for meth in MAIN:
    P(f'| {LABEL[meth]} | ' + ' | '.join(fmt(*stat(sel(method=meth, **q))) for _, q in cols) + ' |')

P('\n### Controller swap (lemniscate): the same adapters under three controllers\n')
P('C1 = authors\' MPC. Aggressive = attitude weight 0.03 fixed. Scheduled = attitude weight set online from '
  'our posterior + sensor-noise estimate (needs a posterior, so only ours).\n')
P('| Adapter | Controller | mass, clean | mass, 40 dB | jumps, clean | jumps, 40 dB | wind, clean | wind, 40 dB |')
P('|---|---|---|---|---|---|---|---|')
def ctrl_cells(meth_c1, meth_agg):
    cells = []
    for scen, grp in [('mass', 'main'), ('jumps', 'jumps'), ('wind', 'wind')]:
        for snr in [None, 40]:
            if meth_agg is None:
                rs = [r for r in sel(method=meth_c1, scen=scen, traj='lemniscate', snr=snr) if r['group'] == grp]
            else:
                rs = sel(method=meth_agg, scen=scen, snr=snr, group='ctrl')
            cells.append(fmt(*stat(rs)))
    return cells
for name, c1, agg in [('Nominal', 'nominal', 'nominal_agg'), ('NN (authors)', 'nn', 'nn_agg'), ('Ours v2', 'bayes_c1', 'bayes_agg')]:
    P(f'| {name} | C1 | ' + ' | '.join(ctrl_cells(c1, None)) + ' |')
    P(f'| {name} | aggressive | ' + ' | '.join(ctrl_cells(None, agg)) + ' |')
P('| Ours v2 | scheduled | ' + ' | '.join(ctrl_cells('bayes', None)) + ' |')

P('\n### Ablation of the full method (lemniscate)\n')
P('| Variant | jumps, clean | jumps, 40 dB ×2 | wind, clean | wind, 40 dB ×2 |')
P('|---|---|---|---|---|')
ABL = [('Full method', 'bayes'), ('− change detection', 'abl_nodetect'), ('single χ² detector instead of multi-scale', 'abl_chi2'),
       ('flat prior on B (b_frac = 1)', 'abl_flatprior'), ('no sensor-noise estimation', 'abl_nonoise')]
for name, meth in ABL:
    cells = []
    for scen in ['jumps', 'wind']:
        for snr in [None, 40]:
            if meth == 'bayes':
                rs = [r for r in sel(method='bayes', scen=scen, traj='lemniscate', snr=snr) if r['group'] in ('jumps', 'wind')]
                if snr == 40:
                    rs = [r for r in rs if r['seed'] < 2]
            else:
                rs = sel(method=meth, scen=scen, snr=snr, group='abl')
            cells.append(fmt(*stat(rs)))
    P(f'| {name} | ' + ' | '.join(cells) + ' |')

P('\n### Change detection, full method, payload jumps (jumps at steps 300, 650, 950)\n')
P('| Noise | seed | detections (step) | delay after each jump (ms) |')
P('|---|---|---|---|')
for s in [None, 50, 40, 30]:
    for r in sorted(sel(group='jumps', method='bayes', snr=s), key=lambda r: r['seed']):
        ks = r.get('kick_steps', [])
        delays = []
        for tj in (300, 650, 950):
            after = [k + 1 - tj for k in ks if tj <= k + 1 < tj + 150]
            delays.append(f'{10*after[0]}' if after else 'not detected')
        P(f"| {'none' if s is None else f'{s} dB'} | {r['seed']} | {ks} | {', '.join(delays)} |")
P('\n### Adaptation cost per control step (ms, median)\n')
P('| Method | ms / step |')
P('|---|---|')
cost = {}
for meth in MAIN:
    v = [r['adapt_ms'] for r in rows if r['method'] == meth and r['group'] in ('main', 'jumps', 'wind')]
    cost[meth] = float(np.median(v))
    P(f'| {LABEL[meth]} | {cost[meth]:.3f} |')
open('results/summary.md', 'w').write('\n'.join(L))
print('\n'.join(L))


def ewma(x, b, init=0.3):
    out = np.zeros_like(x); acc = init
    for i, v in enumerate(x):
        acc = b * acc + (1 - b) * v; out[i] = acc
    return out


# ------------------------------------------------------------------ fig1: payload jumps
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 7), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1, 0.8]))
for meth in ['nominal', 'nn', 'bayes_c1', 'bayes']:
    r = sel(group='jumps', method=meth, snr=None)[0]
    d = np.load(r['file'])
    t = 0.01 * np.arange(len(d['x']))
    e = 100 * np.linalg.norm(d['x'][:, :2] - d['xref'][:, :2], axis=1)
    ax1.plot(t, e, color=COLOR[meth], lw=1.7 if meth == 'bayes' else 1.2, label=SHORT[meth], zorder=3 if meth == 'bayes' else 2)
for tj, lab in [(3.0, 'payload on'), (6.5, 'payload off'), (9.5, 'payload on')]:
    for ax in (ax1, ax2, ax3):
        ax.axvline(tj, color=INK2, lw=0.8, ls=(0, (3, 3)), zorder=1)
    ax1.text(tj + 0.05, 0.97, lab, transform=ax1.get_xaxis_transform(), color=INK2, fontsize=8, va='top')
ax1.set_ylabel('position error (cm)'); ax1.set_ylim(0, None)
ax1.legend(loc='upper left', bbox_to_anchor=(0.0, 0.88), fontsize=9)
ax1.set_title('Payload attach / drop / attach (noise-free, lemniscate)', loc='left', fontsize=11, color=INK)
rb = sel(group='jumps', method='bayes', snr=None)[0]
db = np.load(rb['file'])
nu = db['nu']; tt = 0.01 * (np.arange(len(nu)) + 1)
f = ewma(nu, 0.8)
ax2.semilogy(tt, np.maximum(f, 1e-4), color=COLOR['bayes'], lw=1.2)
ax2.axhline(4.5, color=INK, lw=0.9)
ax2.text(0.1, 4.5 * 1.6, 'fast-scale threshold', color=INK, fontsize=8)
kk = np.nonzero(db['kick'])[0]
ax2.plot(tt[kk], np.maximum(f[kk], 1e-4), 'v', color=INK, ms=7, label='change detected')
ax2.set_ylabel('innovation stat.'); ax2.legend(loc='lower right', fontsize=8)
w = db['w_att']
ax3.semilogy(0.01 * np.arange(len(w)), w, color=COLOR['bayes'], lw=1.2)
ax3.set_ylabel('attitude weight'); ax3.set_xlabel('time (s)')
ax3.set_yticks([0.03, 0.1, 1.0]); ax3.set_yticklabels(['0.03\naggressive', '0.1', '1.0\nconservative'], fontsize=7)
fig.tight_layout(); fig.savefig('figures/fig1_jumps_timeseries.png', dpi=200)

# ------------------------------------------------------------------ fig2: wind
fig, ax = plt.subplots(figsize=(9, 3.4))
for meth in ['nominal', 'nn', 'bayes_c1', 'bayes']:
    r = sel(group='wind', method=meth, snr=None, traj='lemniscate')[0]
    d = np.load(r['file'])
    t = 0.01 * np.arange(len(d['x']))
    e = 100 * np.linalg.norm(d['x'][:, :2] - d['xref'][:, :2], axis=1)
    ax.plot(t, e, color=COLOR[meth], lw=1.7 if meth == 'bayes' else 1.2, label=f"{SHORT[meth]} — {100*r['rmse']:.1f} cm RMSE")
ax.set_ylabel('position error (cm)'); ax.set_xlabel('time (s)'); ax.set_ylim(0, None)
ax.set_title('Constant 3 m/s wind + 20% mass (noise-free, lemniscate)', loc='left', fontsize=11, color=INK)
ax.legend(loc='upper left', fontsize=9)
fig.tight_layout(); fig.savefig('figures/fig2_wind_timeseries.png', dpi=200)

# ------------------------------------------------------------------ fig3: summary dot plot
panels = [(c.replace(', ', '\n', 1), q) for c, q in cols]
fig, axes = plt.subplots(1, len(panels), figsize=(16, 3.9), sharey=True)
ymax = 30
for ax, (title, q) in zip(axes, panels):
    for j, meth in enumerate(MAIN):
        rs = sel(method=meth, **q)
        ok = [100 * r['rmse'] for r in rs if not r['failed']]
        nf = len(rs) - len(ok)
        jit = np.linspace(-0.15, 0.15, max(len(ok), 1))[:len(ok)]
        ax.scatter(j + jit, np.minimum(ok, ymax), s=20, color=COLOR[meth], marker=MARK[meth], edgecolor=SURF, lw=0.6, zorder=3)
        if ok:
            ax.plot([j - 0.3, j + 0.3], [np.median(ok)] * 2, color=INK, lw=1.4, zorder=4)
        if nf:
            ax.text(j, ymax * 1.15, f'{nf}✕', ha='center', va='bottom', fontsize=7.5, color='#c0392b')
    ax.set_yscale('log'); ax.set_ylim(0.5, ymax * 2)
    ax.set_xticks(range(len(MAIN))); ax.set_xticklabels([SHORT[m] for m in MAIN], rotation=60, ha='right', fontsize=7.5)
    ax.set_title(title, fontsize=8.5, color=INK); ax.grid(axis='x', visible=False)
axes[0].set_ylabel('position RMSE (cm, log)')
fig.text(0.01, 0.01, '✕ = diverged runs (count above column, capped points drawn at 30 cm). Black bar = median of completed runs.',
         fontsize=8, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig('figures/fig3_summary.png', dpi=200)

# ------------------------------------------------------------------ fig4: sensitivity
from matplotlib.colors import LogNorm
fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
specs = [('nn', None, 'lr', [5e-5, 1.5e-4, 5e-4], 'window', [5, 10, 20], 'NN adapter, noise-free'),
         ('nn', 40, 'lr', [5e-5, 1.5e-4, 5e-4], 'window', [5, 10, 20], 'NN adapter, 40 dB'),
         ('bayes', None, 'p0', [1.0, 10.0, 100.0], 'tau', [200, 500, 2000], 'Ours (full), noise-free'),
         ('bayes', 40, 'p0', [1.0, 10.0, 100.0], 'tau', [200, 500, 2000], 'Ours (full), 40 dB')]
norm = LogNorm(vmin=1, vmax=20)
for ax, (meth, snr, kx, xs, ky, ys, title) in zip(axes, specs):
    M = np.full((len(ys), len(xs)), np.nan)
    for r in sel(group='sens', method=meth, snr=snr):
        M[ys.index(r['kw'][ky]), xs.index(r['kw'][kx])] = np.inf if r['failed'] else 100 * r['rmse']
    ax.imshow(np.clip(np.where(np.isfinite(M), M, np.nan), 1, 20), cmap='Blues', norm=norm, origin='lower', aspect='auto')
    for i in range(len(ys)):
        for j in range(len(xs)):
            v = M[i, j]
            if np.isinf(v):
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor='#f3d9d6', edgecolor=SURF, lw=2, hatch='//', zorder=2))
            txt = 'fail' if np.isinf(v) else ('—' if np.isnan(v) else f'{v:.1f}')
            ax.text(j, i, txt, ha='center', va='center', fontsize=9, color='white' if (np.isfinite(v) and v > 6) else INK, zorder=3)
    ax.set_xticks(range(len(xs))); ax.set_xticklabels([f'{x:g}' for x in xs])
    ax.set_yticks(range(len(ys))); ax.set_yticklabels([str(y) for y in ys])
    ax.set_xlabel({'lr': 'learning rate', 'p0': 'prior scale p0'}[kx])
    ax.set_ylabel({'window': 'window (steps)', 'tau': 'memory τ (steps)'}[ky])
    ax.set_title(title, fontsize=9, color=INK); ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
nom = stat(sel(group='jumps', method='nominal', snr=None))[0]
fig.suptitle(f'Hyperparameter sensitivity, payload-jump task: position RMSE (cm); nominal = {100*nom:.1f} cm; hatched = diverged',
             fontsize=10, color=INK, x=0.01, ha='left')
fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig('figures/fig4_sensitivity.png', dpi=200)

# ------------------------------------------------------------------ fig5: compute
fig, ax = plt.subplots(figsize=(6.4, 2.6))
ms = ['nn', 'rls_fast', 'rls_slow', 'bayes']
ax.barh(range(len(ms)), [cost[m] for m in ms], color=[COLOR[m] for m in ms], height=0.55)
for i, m in enumerate(ms):
    ax.text(cost[m] * 1.15, i, f'{cost[m]:.2f} ms', va='center', fontsize=9, color=INK)
ax.set_xscale('log'); ax.set_xlim(0.005, 200)
ax.set_yticks(range(len(ms))); ax.set_yticklabels([SHORT[m] for m in ms])
ax.set_xlabel('adaptation time per control step (ms, log); control period = 10 ms')
ax.axvline(10, color=INK2, lw=0.8, ls=(0, (3, 3))); ax.grid(axis='y', visible=False)
fig.tight_layout(); fig.savefig('figures/fig5_compute.png', dpi=200)
print('figures written')

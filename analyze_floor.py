# Table + figure for the altitude-floor experiment (results/floor_runs.jsonl)
import json
import numpy as np
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

rows = [json.loads(l) for l in open('results/floor_runs.jsonl')]
ORDER = ['qp_nominal', 'qp', 'qp_t1', 'qp_t2', 'mppi_mean', 'mppi_post', 'mppi_stateonly', 'mppi_belief']
LABEL = {'qp_nominal': 'QP, no adaptation', 'qp': 'QP + ours (hard floor)', 'qp_t1': 'QP + ours, floor +1σ noise',
         'qp_t2': 'QP + ours, floor +2σ noise', 'mppi_mean': 'MPPI, mean model', 'mppi_post': 'MPPI, posterior models',
         'mppi_stateonly': 'MPPI, state belief only (mean model)', 'mppi_belief': 'MPPI, posterior models + state belief'}
COLOR = {'qp_nominal': '#8a8984', 'qp': '#2a78d6', 'qp_t1': '#1baf7a', 'qp_t2': '#eda100', 'mppi_mean': '#e87ba4',
         'mppi_post': '#008300', 'mppi_stateonly': '#6250d6', 'mppi_belief': '#eb6834'}
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'


def sel(meth, snr):
    return sorted([r for r in rows if r['method'] == meth and r['snr'] == snr], key=lambda r: r['seed'])


L = ['Floor 1.6 cm below the reference minimum; +25% payload picked up 0.3 s before the lowest point.',
     'Depth = deepest violation (mm), time = seconds below the floor, "viol." = runs with any violation.\n']
L.append('| Method | clean: RMSE / depth | 40 dB: viol. runs | 40 dB: mean depth | 30 dB: viol. runs | 30 dB: mean depth | 30 dB: mean time below | 30 dB: median RMSE | control ms |')
L.append('|---|---|---|---|---|---|---|---|---|')
for m in ORDER:
    c = sel(m, None)[0]
    cells = [f"{100*c['rmse']:.1f} cm / {c['depth_mm']:.1f}" if c['rmse'] else 'fail']
    for snr in [40, 30]:
        rs = [r for r in sel(m, snr) if not r['failed']]
        nf = len(sel(m, snr)) - len(rs)
        d = np.array([r['depth_mm'] for r in rs])
        cells.append(f"{np.sum(d > 0)}/{len(sel(m, snr))}" + (f' ({nf} fail)' if nf else ''))
        cells.append(f"{d.mean():.1f}")
        if snr == 30:
            cells.append(f"{np.mean([r['time_below_s'] for r in rs]):.2f}")
            cells.append(f"{100*np.median([r['rmse'] for r in rs]):.1f} cm")
    cells.append(f"{np.median([r['ctrl_ms'] for r in sel(m, 30)]):.1f}")
    L.append(f'| {LABEL[m]} | ' + ' | '.join(cells) + ' |')

L.append('\nPaired comparison over the same seeds (Wilcoxon signed-rank on violation depth):\n')
for snr in [40, 30]:
    for a, b in [('mppi_belief', 'qp'), ('mppi_belief', 'mppi_mean'), ('mppi_belief', 'mppi_post'), ('mppi_belief', 'qp_t1'), ('mppi_belief', 'mppi_stateonly')]:
        da = np.array([r['depth_mm'] for r in sel(a, snr)]); db = np.array([r['depth_mm'] for r in sel(b, snr)])
        diff = da - db
        p = wilcoxon(da, db).pvalue if np.any(diff != 0) else 1.0
        L.append(f'- {snr} dB, {LABEL[a]} vs {LABEL[b]}: mean depth {da.mean():.1f} vs {db.mean():.1f} mm, '
                 f'better in {np.sum(diff < 0)}, worse in {np.sum(diff > 0)}, tied {np.sum(diff == 0)} of {len(diff)} seeds, p = {p:.3f}')
open('results/floor_summary.md', 'w').write('\n'.join(L))
print('\n'.join(L))

plt.rcParams.update({'font.size': 10, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.facecolor': SURF, 'figure.facecolor': SURF, 'axes.grid': True,
                     'grid.color': GRID, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for ax, snr in zip(axes, [40, 30]):
    for j, m in enumerate(ORDER):
        d = [r['depth_mm'] for r in sel(m, snr) if not r['failed']]
        jit = np.linspace(-0.18, 0.18, len(d))
        ax.scatter(j + jit, d, s=22, color=COLOR[m], edgecolor=SURF, lw=0.6, zorder=3)
        ax.plot([j - 0.3, j + 0.3], [np.mean(d)] * 2, color=INK, lw=1.4, zorder=4)
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels([LABEL[m].replace(', ', ',\n', 1) for m in ORDER], rotation=50, ha='right', fontsize=7.5)
    ax.set_title(f'{snr} dB sensor noise (10 seeds)', fontsize=10, color=INK, loc='left')
    ax.grid(axis='x', visible=False)
axes[0].set_ylabel('deepest floor violation (mm)')
fig.text(0.01, 0.01, 'Black bar = mean over seeds. Points at 0 = no violation.', fontsize=8, color=INK2)
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig('figures/fig6_floor.png', dpi=200)
print('figure written')

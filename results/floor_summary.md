Floor 1.6 cm below the reference minimum; +25% payload picked up 0.3 s before the lowest point.
Depth = deepest violation (mm), time = seconds below the floor, "viol." = runs with any violation.

| Method | clean: RMSE / depth | 40 dB: viol. runs | 40 dB: mean depth | 30 dB: viol. runs | 30 dB: mean depth | 30 dB: mean time below | 30 dB: median RMSE | control ms |
|---|---|---|---|---|---|---|---|---|
| QP, no adaptation | 7.4 cm / 0.0 | 0/10 | 0.0 | 10/10 | 18.2 | 0.39 | 9.2 cm | 9.0 |
| QP + ours (hard floor) | 1.1 cm / 0.0 | 0/10 | 0.0 | 10/10 | 9.6 | 0.30 | 8.7 cm | 9.1 |
| QP + ours, floor +1σ noise | 1.1 cm / 0.0 | 0/10 | 0.0 | 10/10 | 8.9 | 0.25 | 8.5 cm | 9.8 |
| QP + ours, floor +2σ noise | 1.1 cm / 0.0 | 0/10 | 0.0 | 9/10 | 8.3 | 0.24 | 8.6 cm | 10.0 |
| MPPI, mean model | 1.2 cm / 0.0 | 0/10 | 0.0 | 10/10 | 11.5 | 0.34 | 8.7 cm | 15.6 |
| MPPI, posterior models | 1.2 cm / 0.0 | 0/10 | 0.0 | 10/10 | 11.6 | 0.32 | 8.6 cm | 19.4 |
| MPPI, state belief only (mean model) | 1.1 cm / 0.0 | 0/10 | 0.0 | 9/10 | 4.4 | 0.15 | 8.2 cm | 19.2 |
| MPPI, posterior models + state belief | 1.1 cm / 0.0 | 0/10 | 0.0 | 6/10 | 3.4 | 0.11 | 8.3 cm | 18.8 |

Paired comparison over the same seeds (Wilcoxon signed-rank on violation depth):

- 40 dB, MPPI, posterior models + state belief vs QP + ours (hard floor): mean depth 0.0 vs 0.0 mm, better in 0, worse in 0, tied 10 of 10 seeds, p = 1.000
- 40 dB, MPPI, posterior models + state belief vs MPPI, mean model: mean depth 0.0 vs 0.0 mm, better in 0, worse in 0, tied 10 of 10 seeds, p = 1.000
- 40 dB, MPPI, posterior models + state belief vs MPPI, posterior models: mean depth 0.0 vs 0.0 mm, better in 0, worse in 0, tied 10 of 10 seeds, p = 1.000
- 40 dB, MPPI, posterior models + state belief vs QP + ours, floor +1σ noise: mean depth 0.0 vs 0.0 mm, better in 0, worse in 0, tied 10 of 10 seeds, p = 1.000
- 40 dB, MPPI, posterior models + state belief vs MPPI, state belief only (mean model): mean depth 0.0 vs 0.0 mm, better in 0, worse in 0, tied 10 of 10 seeds, p = 1.000
- 30 dB, MPPI, posterior models + state belief vs QP + ours (hard floor): mean depth 3.4 vs 9.6 mm, better in 10, worse in 0, tied 0 of 10 seeds, p = 0.002
- 30 dB, MPPI, posterior models + state belief vs MPPI, mean model: mean depth 3.4 vs 11.5 mm, better in 10, worse in 0, tied 0 of 10 seeds, p = 0.002
- 30 dB, MPPI, posterior models + state belief vs MPPI, posterior models: mean depth 3.4 vs 11.6 mm, better in 10, worse in 0, tied 0 of 10 seeds, p = 0.002
- 30 dB, MPPI, posterior models + state belief vs QP + ours, floor +1σ noise: mean depth 3.4 vs 8.9 mm, better in 10, worse in 0, tied 0 of 10 seeds, p = 0.002
- 30 dB, MPPI, posterior models + state belief vs MPPI, state belief only (mean model): mean depth 3.4 vs 4.4 mm, better in 6, worse in 3, tied 1 of 10 seeds, p = 0.301
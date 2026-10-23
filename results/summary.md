Position RMSE in cm. Median over runs that did not diverge; "(k/n fail)" = k of n runs diverged.

### Main comparison

| Method | Mass +20%, noise-free, 4 traj | Mass +20%, 40 dB, 4 traj × 3 | Payload jumps, noise-free | Payload jumps, 50 dB ×3 | Payload jumps, 40 dB ×3 | Payload jumps, 30 dB ×3 | Wind + mass, noise-free, 4 traj | Wind + mass, 50 dB ×5 | Wind + mass, 40 dB ×5 |
|---|---|---|---|---|---|---|---|---|---|
| Nominal (no adaptation) | 9.5 | 9.6 | 5.3 | 5.3 | 5.4 | 6.7 | 35.9 | 37.7 | 37.7 |
| NN adapter (authors) | 1.7 (1/4 fail) | 21.2 (5/12 fail) | 9.8 | 1.7 | 3.5 (2/3 fail) | fail (3/3) | 29.0 (1/4 fail) | 22.1 (4/5 fail) | 7.3 (1/5 fail) |
| RLS, λ=0.95 | 1.7 | fail (12/12) | fail (1/1) | fail (3/3) | fail (3/3) | fail (3/3) | fail (4/4) | fail (5/5) | fail (5/5) |
| RLS, λ=0.999 | 3.3 | 387.4 (10/12 fail) | 5.1 | 79.4 (2/3 fail) | fail (3/3) | fail (3/3) | 166.4 (1/4 fail) | 132.3 (1/5 fail) | 365.3 (3/5 fail) |
| Bayesian v1 (first version) | 1.8 | 4.8 | 1.1 | 1.6 | 3.4 | 6.2 | 23.2 | 17.5 | 21.4 |
| Bayesian v2, authors' controller | 1.1 | 3.7 | 1.1 | 2.0 | 3.4 | 6.1 | 21.0 | 19.1 | 21.8 |
| Bayesian v2 + scheduled controller (ours, full) | 1.1 | 3.7 | 1.1 | 2.0 | 3.4 | 6.1 | 3.9 | 19.1 | 21.8 |

### Controller swap (lemniscate): the same adapters under three controllers

C1 = authors' MPC. Aggressive = attitude weight 0.03 fixed. Scheduled = attitude weight set online from our posterior + sensor-noise estimate (needs a posterior, so only ours).

| Adapter | Controller | mass, clean | mass, 40 dB | jumps, clean | jumps, 40 dB | wind, clean | wind, 40 dB |
|---|---|---|---|---|---|---|---|
| Nominal | C1 | 10.2 | 10.3 | 5.3 | 5.4 | 37.7 | 37.7 |
| Nominal | aggressive | 10.1 | 10.1 (2/3 fail) | 5.2 | 5.1 (2/3 fail) | 14.1 | fail (3/3) |
| NN (authors) | C1 | 1.7 | 21.2 (2/3 fail) | 9.8 | 3.5 (2/3 fail) | 5.8 | 7.3 (1/5 fail) |
| NN (authors) | aggressive | fail (1/1) | fail (3/3) | fail (1/1) | fail (3/3) | fail (1/1) | fail (3/3) |
| Ours v2 | C1 | 1.2 | 4.1 | 1.1 | 3.4 | 27.4 | 21.8 |
| Ours v2 | aggressive | 1.1 | fail (3/3) | 1.1 | fail (3/3) | 2.4 | fail (3/3) |
| Ours v2 | scheduled | 1.2 | 4.1 | 1.1 | 3.4 | 3.8 | 21.8 |

### Ablation of the full method (lemniscate)

| Variant | jumps, clean | jumps, 40 dB ×2 | wind, clean | wind, 40 dB ×2 |
|---|---|---|---|---|
| Full method | 1.1 | 3.5 | 3.8 | 17.4 |
| − change detection | 1.1 | 3.5 | 3.8 | 17.4 |
| single χ² detector instead of multi-scale | 1.1 | 3.9 | 3.8 | 31.7 |
| flat prior on B (b_frac = 1) | 10.3 | fail (2/2) | 12.6 | 804.0 (1/2 fail) |
| no sensor-noise estimation | 1.1 | fail (2/2) | 3.8 | fail (2/2) |

### Change detection, full method, payload jumps (jumps at steps 300, 650, 950)

| Noise | seed | detections (step) | delay after each jump (ms) |
|---|---|---|---|
| none | 0 | [301, 651, 951] | 20, 20, 20 |
| 50 dB | 0 | [] | not detected, not detected, not detected |
| 50 dB | 1 | [] | not detected, not detected, not detected |
| 50 dB | 2 | [] | not detected, not detected, not detected |
| 40 dB | 0 | [] | not detected, not detected, not detected |
| 40 dB | 1 | [] | not detected, not detected, not detected |
| 40 dB | 2 | [] | not detected, not detected, not detected |
| 30 dB | 0 | [] | not detected, not detected, not detected |
| 30 dB | 1 | [] | not detected, not detected, not detected |
| 30 dB | 2 | [] | not detected, not detected, not detected |

### Adaptation cost per control step (ms, median)

| Method | ms / step |
|---|---|
| Nominal (no adaptation) | 0.001 |
| NN adapter (authors) | 9.074 |
| RLS, λ=0.95 | 0.013 |
| RLS, λ=0.999 | 0.030 |
| Bayesian v1 (first version) | 0.101 |
| Bayesian v2, authors' controller | 0.106 |
| Bayesian v2 + scheduled controller (ours, full) | 0.105 |
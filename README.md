# Bayesian online adaptation for Koopman MPC

Built on Singh, Sah & Keshavan, *Adaptive Koopman embedding for robust control of nonlinear dynamical systems*,
IJRR 2025 ([code](https://github.com/Rajpal9/Adaptive-koopman)). Test system: their planar quadrotor, their trained
Koopman model (17 lifted states, 2 inputs), their MPC (N = 15, OSQP, same weights and bounds). All numbers are
position RMSE in cm. Full tables are in `results/summary.md`; 376 closed-loop runs are in `results/runs_v2.jsonl`.

## Results in one paragraph

We replace the authors' per-step neural-network adaptation with a closed-form Bayesian update of `[A B]`, and
let the controller use the resulting confidence. With +20% mass/inertia, error falls from 9.5 cm (no
adaptation) to **1.1 cm**; the NN gets 1.7 cm and diverged on 1 of 4 trajectories. With a payload attached,
dropped and re-attached mid-flight: **1.1 cm** for ours, 5.3 cm with no adaptation, 9.8 cm for the NN. With
3 m/s wind plus mass: **3.9 cm** for ours (median over 4 trajectories), 35.9 cm with no adaptation, 29.0 cm
for the NN (which diverged on 1 of 4). Across all 40 runs (3 scenarios, 4 noise levels, several seeds), ours
never diverged. The NN diverged in 17 of 40 and forgetting-factor RLS in 59 of 80. The update costs 0.11 ms
per step, against 9.1 ms for the NN (the control period is 10 ms).

Where it is **not** better: constant wind *with* sensor noise. At 40 dB the NN's median is better (7.3 vs
21.8 cm), though it diverged in 1 of 5 runs. At 50 dB the NN diverged in 4 of 5. Our method's results there
are widely spread across seeds (8.7–60 cm).

## What changed from the first version, and why

| Change | Problem it fixes | Evidence |
|---|---|---|
| **Confidence-scheduled controller** (new) | The authors' MPC penalises attitude against the *nominal* reference, so holding position in wind (which needs a sustained tilt) is "expensive". With an accurate model, the controller then tolerates a 25–35 cm offset. | Wind, noise-free: 27.4 → 3.8 cm (lemniscate). |
| **Multi-scale change detection** with calibrated thresholds | Under noise, innovations are correlated and heavy-tailed. The single χ² test (which assumes white noise) fired up to 33 times per run, and each false alarm reset the model. | Wind @ 40 dB: 31.7 → 17.4 cm in the ablation. In all 31 noisy runs, the only detections are single true ones right after warm-up when the change is present from t = 0; no repeated or spurious alarms. |
| **Shorter warm-up** (1 s → 0.1 s) | The adapter waited 1 s before learning, and early errors became offsets the controller never removed. | v1 → v2 adapter (mostly this change): mass, noise-free 1.8 → 1.1 cm; mass @ 40 dB 4.8 → 3.7 cm. |
| Solver only updated when the weight changes | Re-sending identical weights perturbed the QP numerics in chaotic regimes. | Scheduled = authors' controller exactly when conservative. |

### The confidence-scheduled controller

The MPC's attitude weight is set every step on a log scale between 0.03 (aggressive) and 1.0 (the authors'
conservative value). It goes conservative when **either** signal says it should:

- **Model uncertainty:** the adapter's parameter variance `φᵀPφ`. This is high just after a change, or before the
  model has learned anything.
- **Sensor noise:** a model-free estimate from second differences of the measured *positions*. Position is smooth
  at 100 Hz however hard the controller works, so whatever is left is noise.

It backs off immediately and relaxes slowly. The controller-swap table shows why this needs our adapter:

| Adapter | Controller | jumps, clean | jumps, 40 dB | wind, clean | wind, 40 dB |
|---|---|---|---|---|---|
| Nominal | authors' | 5.3 | 5.4 | 37.7 | 37.7 |
| Nominal | aggressive (fixed) | 5.2 | 5.1 (2/3 fail) | 14.1 | fail (3/3) |
| NN (authors) | authors' | 9.8 | 3.5 (2/3 fail) | 5.8 | 7.3 (1/5 fail) |
| NN (authors) | aggressive (fixed) | fail | fail (3/3) | fail | fail (3/3) |
| Ours | authors' | 1.1 | 3.4 | 27.4 | 21.8 |
| Ours | aggressive (fixed) | 1.1 | fail (3/3) | 2.4 | fail (3/3) |
| **Ours** | **scheduled** | **1.1** | **3.4** | **3.8** | **21.8** |

The aggressive controller only works with an accurate model and clean sensors: the NN diverges under it every
time. The schedule gives the aggressive result when that is safe and the conservative result when it is not.

## What the adapter does (`adapters.py`, class `BayesKoopman`)

1. **Closed-form Bayesian update.** `z_k = [A B][z_{k-1}; u_{k-1}] + noise` is linear in the unknowns. We keep
   `Θ = [A B] ~ MN(M, Σ, P)` with measurement noise `rΣ`. `Σ` is the nominal model's per-row residual variance on its
   own training data. Every row then shares one 19×19 covariance, so an update is a rank-1, RLS-sized step. The gain
   `K = Pφ/(φᵀPφ + r)` makes the step size depend on confidence relative to noise.
2. **Mean-reverting prior.** `Θ_{k+1} = Θ_nom + ρ(Θ_k − Θ_nom) + W`, `W ~ MN(0, Σ, P0/τ)`, `ρ² = 1 − 1/τ`.
   The stationary covariance is the prior `P0`. Unexcited directions relax back to the nominal model instead of
   winding up, which is what makes forgetting-factor RLS diverge.
3. **Structured prior, centred at the operating point.** In closed loop the thrust sits near hover, so a constant
   push and a change in thrust effectiveness are nearly indistinguishable. The prior learns offsets fast (weight 1)
   and input gains slowly (`b_frac = 0.01`), in coordinates centred at a slow average of the current regressor.
4. **Sensor-noise estimate** from *differenced* innovations. White noise survives differencing; a slow model
   mismatch mostly cancels. Under heavy noise `r` grows, gains shrink, and the adapter falls back toward nominal.
5. **Multi-scale change detection.** Three averages of the normalised innovation (≈5, 20, 100 steps), each with a
   threshold of 1.5× the largest value seen in no-change runs on held-out trajectories (circle, S;
   `calibrate_detector.py`). On detection, `P ← P + P0`.

Hyperparameters: `p0 = 10`, `τ = 500` steps, `b_frac = 0.01`, warm-up 10 steps, and the schedule bands
(model 3–30, noise −62 to −56 dB).

## Ablation (lemniscate)

| Variant | jumps, clean | jumps, 40 dB | wind, clean | wind, 40 dB |
|---|---|---|---|---|
| Full method | 1.1 | 3.5 | 3.8 | 17.4 |
| − change detection | 1.1 | 3.5 | 3.8 | 17.4 |
| single χ² detector | 1.1 | 3.9 | 3.8 | 31.7 |
| flat prior on B | 10.3 | fail (2/2) | 12.6 | 804 (1/2 fail) |
| no sensor-noise estimate | 1.1 | fail (2/2) | 3.8 | fail (2/2) |

How to read this honestly:

- **The structured prior and the noise estimate carry the method.** Remove either and runs diverge.
- **Change detection itself adds nothing measurable in these tests.** The mean-reverting prior keeps the gain
  high enough to relearn a ±20% change within about 0.1 s anyway. The detector's value is that a *bad* detector
  hurts: the single χ² test nearly doubles wind error at 40 dB through false alarms. Keep it as a safety net, but
  don't claim it as the main driver.

## Hyperparameter sensitivity (`figures/fig4_sensitivity.png`, payload-jump task)

- **Noise-free:** all 9 `(p0, τ)` settings give 1.1–4.6 cm. For the NN, 4 of 9 `(lr, window)` settings diverge.
- **40 dB:** `p0 = 100` diverges, so `p0` must stay ≤ 10 with noisy sensors. For the NN, 3 of 9 settings diverge.

## Reproduction notes (put these in the report)

- `repro_authors.py` runs the authors' notebook code. With +20% mass it gives 10.2 → 8.8 cm; with 3 m/s wind added,
  37.7 → 5.8 cm. Our MPC reproduces their controller's trajectory to 6×10⁻⁶ m.
- **The authors' adapter is numerically chaotic.** Identical code gives 8.8 cm in their pipeline and 1.7 cm in
  ours. Perturbing its residuals by 10⁻⁷ moves the result between 1.5 and 1.9 cm; by 10⁻⁵, some runs diverge. It
  scores 9.8 cm on the noise-free jump task but 1.7 cm with 50 dB noise. It retrains a zero-initialised net with
  Adam every step and *accumulates* the result, which makes it a fixed-step, sign-normalised integral law. Report
  medians over seeds and failure counts, never single runs.

## Things we tried that did not work (worth a sentence each in the report)

- **Chance-constraint tightening from the posterior.** The 15-step altitude standard deviation is about 2 mm, against
  41 mm of sag. The 0.15 s horizon is too short for parameter uncertainty to accumulate. The confidence-scheduled
  controller is the use of uncertainty that does matter.
- **Outer-loop integral action.** It reduced wind error (29 → 11 cm) but made the payload cases 3–4× worse.
- **Attitude-target correction from the learned disturbance.** It helped wind but leaked mass changes into the
  attitude target.
- **Riccati terminal cost.** Not usable: the lifted model is nearly marginally stable (|λ| ≈ 1.004), giving cost
  entries around 10¹⁵ and an extremely slow LQR.
- **Uncertainty-weighted state observer.** Diverged under noise (altitude to −30 m at 30 dB). Under heavy noise it
  relies on the model's open-loop prediction, and this Koopman model drifts quickly in open loop.
- **Adapting the lifted-state columns of A, or a longer memory (τ = 2000–5000).** Both were worse.

## Limitations

- Wind with sensor noise is unsolved. Results spread widely across seeds for every method, and in one run of ours
  (50 dB) the RMSE reached 60 cm.
- Under noise of 50 dB or more, changes are not detected. The adapter then improves only slowly over the nominal
  model, which is still better than the alternatives (which diverge).
- The schedule's bands and detector thresholds were set on simulation runs: detector thresholds on held-out
  trajectories, noise bands on the lemniscate. Treat them as design choices, not universal constants.
- One system, in simulation, with the authors' Koopman model.

## Contribution statement (relative to SOTA)

1. A closed-form Bayesian replacement for the per-step NN adaptation. It removes the authors' stated
   tuning limitation (window, epochs, regularisation), is about 85× cheaper, and never diverged in 40 runs where
   the NN diverged in 17.
2. A mean-reverting, operating-point-centred structured prior. Without it, the same filter diverges (ablation).
3. Sensor-noise estimation from differenced innovations, giving graceful fallback under noise (ablation: diverges
   without it).
4. **Confidence-scheduled MPC.** The posterior and a model-free noise estimate set the controller's aggressiveness
   online. This is what closes the wind gap (35.9 → 3.9 cm), and it only works because the adapted model is
   accurate: the NN and the nominal model diverge under the same aggressive controller.
5. A new evaluation scenario (abrupt payload attach/drop) and an evaluation finding: the original adapter is
   numerically chaotic, so single-run results are unreliable.

## Posterior-sampled MPPI (`mppi.py`, `floor_experiment.py`)

**Design that works:** MPPI samples smooth corrections (4 knots, interpolated) around the QP plan for the posterior-mean
model. Every sampled control sequence is scored under 8 *models* drawn from the adapter's posterior and 8 *initial states*
drawn from the sensor-noise belief. The score is mean + CVaR over those samples, plus a soft altitude-floor penalty. The
posterior spread is calibrated online by the recent average of the normalised innovation: about 0.3 mm in steady
state, opening to several cm right after a change.

**Test:** a +25% payload picked up 0.3 s before the lowest point of the lemniscate, floor 1.6 cm below the reference
minimum, 10 seeds per noise level (`results/floor_summary.md`, `figures/fig6_floor.png`).

| Controller (all with our adapter unless noted) | 30 dB: runs violating | mean deepest violation | mean time below floor | median RMSE |
|---|---|---|---|---|
| QP, no adaptation | 10/10 | 18.2 mm | 0.39 s | 9.2 cm |
| QP, hard floor | 10/10 | 9.6 mm | 0.30 s | 8.7 cm |
| QP, floor raised by 1σ / 2σ of estimated noise | 10/10 · 9/10 | 8.9 · 8.3 mm | 0.25 · 0.24 s | 8.5 · 8.6 cm |
| MPPI, mean model | 10/10 | 11.5 mm | 0.34 s | 8.7 cm |
| MPPI, posterior models only | 10/10 | 11.6 mm | 0.32 s | 8.6 cm |
| MPPI, state belief only (mean model) | 9/10 | 4.4 mm | 0.15 s | 8.2 cm |
| **MPPI, posterior models + state belief** | **6/10** | **3.4 mm** | **0.11 s** | **8.3 cm** |

- **Full method vs each baseline** (QP hard floor, QP +1σ, mean-model MPPI, posterior-only MPPI): violation depth
  is lower in 10 of 10 paired seeds, Wilcoxon p = 0.002. Tracking is no worse.
- **Full method vs state belief only:** 3.4 vs 4.4 mm, better in 6, worse in 3 of 10 seeds, p = 0.30. The model
  posterior adds a small, not significant extra gain.
- **At 40 dB and noise-free, nobody violates.** Tracking matches the QP (1.1–1.2 cm noise-free).
- **Cost:** about 19 ms per step in numpy, against 9 ms for the QP. It is not real-time at 100 Hz in Python, but the
  rollouts are batched matrix products and would be much faster on GPU or in compiled code.

**What did not work:**
- Plain MPPI (no QP seed) diverged: the problem is too stiff for random search.
- Uncalibrated posterior samples were too narrow (about 1 mm spread) to change any decision.
- Using posterior MPPI to replace the confidence schedule made wind worse (35.5 vs 27.4 cm): the aggressive
  controller's failure under noise is closed-loop noise amplification, which open-loop rollouts can't see.
- A higher risk weight (CVaR α = 0.1, weight 3) destabilised the controller.

**Honest framing:** the benefit comes mainly from planning over a *belief about the current state*, i.e. sensor noise.
Sampling the *model* posterior alone gives nothing here, because the adapter learns the payload within about 20 ms,
leaving little model uncertainty to plan around.

## Files

| File | What it is |
|---|---|
| `koopman_sim.py` | Authors' model + numpy lift; their quadrotor ODE with mid-flight changes; our OSQP Koopman MPC (matches theirs to 6e-6 m); closed loop with sensor noise and the confidence schedule |
| `adapters.py` | `NoAdapt`, `AuthorsNN` (their `AdaptNet_linear`, called as in their notebook), `RLS`, `BayesKoopman` (ours) |
| `quick.py` | Single runs, e.g. `python quick.py bayes wind sched=1 snr_db=40` |
| `run_experiments.py` | The full grid; resumable; reuses unchanged baseline runs from v1 |
| `analyze.py` | `results/summary.md` and `figures/` |
| `calibrate_detector.py` | Detector statistics under no change (held-out trajectories) → thresholds |
| `mppi.py`, `mppi_quick.py` | QP-seeded MPPI with posterior-sampled models, state belief and CVaR; single runs |
| `floor_experiment.py`, `analyze_floor.py` | Altitude-floor experiment (176 runs) and its table and figure |
| `diagnostics.py` | Why the authors' controller can't use an accurate model in wind; size of the uncertainty margins |
| `repro_authors.py`, `check_equivalence.py`, `check_sensitivity.py` | Reproduction and the chaos check |

Setup: `pip install torch osqp==0.6.7.post3 gym control scipy matplotlib`, with the authors' repo cloned next to
this folder as `Adaptive-koopman/`. Then run `python run_experiments.py && python analyze.py` (about 45 min on 2
cores).

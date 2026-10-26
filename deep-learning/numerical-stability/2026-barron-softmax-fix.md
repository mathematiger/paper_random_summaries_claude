---
title: "A Fix for Softmax and Cross-Entropy"
authors: ["Jonathan T. Barron", "Dan Gnanapragasam", "Michael Tschannen"]
year: 2026
venue: "arXiv preprint (cs.LG, v1 22 Oct 2026)"
link: "https://arxiv.org/abs/2610.31642"
added: 2026-10-26
category: "deep-learning/numerical-stability"
tags: ["softmax", "cross-entropy", "floating-point", "low-precision", "adam", "grokking", "distillation", "calibration"]
---

# Softmax fix — standard cross-entropy silently drops the target gradient at high confidence, and Adam turns that into loss spikes

**Main reference:** [A Fix for Softmax and Cross-Entropy — J. T. Barron, D. Gnanapragasam & M. Tschannen (2026), arXiv:2610.31642](https://arxiv.org/abs/2610.31642) — read Sections 1–3 closely: the logit margin (eq. 2), the hard-target gradients (eq. 3), the fixed log-softmax and cross-entropy (eqs. 5–7) and the fixed soft-target gradient (eq. 9), each with a short JAX snippet. Then read Section 4 (how Adam turns the bug into Slingshot spikes) and Section 5 with Table 1 slowly: the margin-equilibrium prediction `m* ≈ log(1/ε)` is the most transferable result. Skim Section 6 (precision sweeps on modular arithmetic) and Section 7 (distillation drift). The [project page](https://jonbarron.info/softmaxfix/) has an interactive demo; there is no official code repository, only the inline JAX functions. An unofficial [PyTorch port](https://github.com/PwnerHarry/barron-softmax-fix) exists (not checked).

**Further reading:**
- [Grokking at the Edge of Numerical Stability — L. Prieto, M. Barsbey, P. A. M. Mediano & T. Birdal (2025), ICLR](https://arxiv.org/abs/2501.04697) — first named "Softmax Collapse" and proposed StableMax as a replacement activation; this paper argues for fixing the arithmetic instead and uses StableMax as a baseline.
- [The Slingshot Mechanism: An Empirical Study of Adaptive Optimizers and the Grokking Phenomenon — V. Thilak et al. (2022)](https://arxiv.org/abs/2206.04817) — the periodic loss spikes in late training that Section 4 explains as a numerical artefact.
- [Grokking or Glitching? How Low-Precision Drives Slingshot Loss Spikes — Liu et al. (2026)](https://arxiv.org/abs/2605.06152) — concurrent explanation via low precision, with a zero-sum gradient projection as the fix; the second baseline here.
- [Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets — A. Power et al. (2022)](https://arxiv.org/abs/2201.02177) — the original grokking paper and the source of the modular-arithmetic benchmark.

**Why this matters** — Every framework computes softmax and cross-entropy in a way that is "numerically stable" in the textbook sense: logits are shifted by their maximum so nothing overflows. This paper shows that this standard implementation still has two silent bugs that appear exactly when a model becomes confident or matches its target well, which is late training, grokking and distillation. The fixes are drop-in replacements of a few lines. The more general lesson is about interaction effects: a tiny rounding error in the loss is harmless with SGD, but Adam's normalisation amplifies it into training instabilities, and the optimizer's `ε` ends up deciding how confident a network becomes.

**Core ideas**

*Setup and the margin.* Logits `z ∈ ℝ^K`, softmax `q = softmax(z)`. For a hard target `t` the cross-entropy gradient is `g = q − e_t`; for a soft target `p` it is `g = q − p`. The quantity that controls everything is the logit margin

```
m = z_t − log Σ_{j≠t} exp(z_j)          (eq. 2)
```

so that `q_t = 1 / (1 + e^{−m})` and the loss is `L = softplus(−m) = log(1 + e^{−m})`.

*Bug 1: Softmax Collapse (hard targets).* The standard target gradient is

```
g_t = q_t − 1 = 1/(1 + e^{−m}) − 1
```

Once `e^{−m}` is smaller than half a unit in the last place of 1, the sum `1 + e^{−m}` rounds to exactly 1, `q_t` becomes 1 and `g_t` becomes exactly 0. The non-target gradients `g_j = e^{z_j − z_t}/(1 + e^{−m})` are still positive, so the gradient no longer sums to zero: the network is pushed away from the wrong classes but gets no pull toward the right one. The paper reports the cliffs `c_d` where this happens as ≈ 5.5 (bf16), 16.6 (fp32) and 36.7 (fp64). These equal `p · ln 2` for the `p`-bit significand (8, 24, 53), which is my reading of where they come from; the paper defines them only in words. Max-shifting does not help, because the information is lost when forming `1 + e^{−m}`, not through overflow.

*Fix 1.* Compute the margin directly with a log-sum-exp over the *non-target* logits, relative to the target logit, and never form `q_t − 1`:

```
m   = −LSE_{j≠t} (z_j − z_t)
L   = softplus(−m)
g_j = exp(z_j − z_t − L)        for j ≠ t
g_t = −Σ_{j≠t} g_j              (exactly zero-sum)
```

In the JAX code the target slot is set to `−∞` inside the log-sum-exp, so autodiff produces these gradients automatically. `g_t ≈ −e^{−m}` keeps full relative precision at any margin. The same idea gives a fixed `log_softmax` (eq. 5): shift by the largest logit, and compute the largest logit's own log-probability as `−softplus(−m)` from the others, instead of subtracting a rounded log-sum-exp.

*Bug 2: Softmax Cancellation (soft targets).* With `p = softmax(z_T)` from a teacher and a student `z_S = z_T + δ`, the exact gradient is proportional to `δ`, but `q − p` subtracts two nearly equal numbers. The result is rounding noise of the order of machine precision (the paper reports residuals of about `1e−17` in fp64, `1e−8` in fp32 and `1e−3` in bf16) instead of the true, tiny gradient. A student initialised *at* the teacher therefore drifts away from it.

*Fix 2 (eq. 9).* Compute the ratio `q_k/p_k` in log space, where small differences are representable:

```
d_k = δ_k − Σ_i p_i δ_i                        (centre the perturbation)
s_k = d_k − log1p( Σ_i p_i · expm1(d_i) )      (= log(q_k / p_k))
u_k = p_k · expm1(s_k)                         (= q_k − p_k, without cancellation)
g_k = u_k − q_k · Σ_i u_i                      (restore exact zero sum)
```

`expm1` and `log1p` keep relative precision for small arguments, so `g` is accurate even when `δ` is far below machine precision. The forward loss stays the standard one; the corrected gradient is injected in the backward pass with a stop-gradient trick, `sg(g) · (z_S − sg(z_S))`.

*Why Adam turns Bug 1 into spikes (Section 4).* After memorisation, margins grow. When every example in a batch passes `c_d`, the target gradients vanish and parameter gradients become tiny, so Adam's second moment `v̂` decays toward zero while weight decay slowly shrinks the weights. Eventually some examples fall back below the cliff, their target gradient `≈ −e^{−m}` reappears, and Adam divides it by a tiny `√v̂ + ε`. The result is a huge step, a loss spike, a reset of `v̂`, and the cycle repeats. That is the Slingshot mechanism. With the fixed loss, `g_t` never jumps to zero, the first and second moments decay together, and the spikes disappear. Whether spikes hurt depends on weight-norm control: with LayerNorm and no weight decay they act like accidental learning-rate decay; with weight decay they slow grokking (94 ± 5% vs 100 ± 0% test accuracy on `a + b mod 31`); with neither, a single spike destroys the solution (61% → 23% on `a² + b² mod 97`, where the fixed loss reaches 85%).

*Margin equilibrium (Section 5).* At a stationary point of AdamW, `m̂ = g` and `v̂ = g²`, so the update vanishes when, per coordinate,

```
g_i / (|g_i| + ε) + λ θ_i = 0          (eq. 10;  η, β1, β2 cancel)
```

After memorisation `|g_i| ∝ e^{−m}`. While `|g_i| ≫ ε` the normalised step is about 1 and margins keep growing; once `|g_i| ≪ ε` the step is about `g_i/ε ∝ e^{−m}` and can be balanced by weight decay, which requires `|g_i| ≈ ε λ |θ_i|`. Dropping constants gives

```
m*(ε) ≈ log(1/ε)                        (eq. 11)
```

This is a heuristic argument, but it matches well: for `ε = 1e−8, 1e−12, 1e−16` it predicts 18.4, 27.6, 36.8 and the fixed loss reaches 17.9, 27.4, 36.8 (Table 1, `a + b mod 31`, `λ = 1`). Without weight decay nothing balances the step and margins keep rising. Two consequences: the default `ε = 1e−8` puts the equilibrium margin (≈ 18.4) just *above* the fp32 cliff (16.6), so standard fp32 training runs straight into Softmax Collapse; and a network's final confidence is set by optimizer hyperparameters, not by the data.

*Results (Section 6, Table 1).* All experiments are small transformers (1–2 layers) on modular arithmetic. On `a + b mod 31` with `ε = 1e−8`, the fixed loss reaches 100% test accuracy in bf16, fp32 and fp64; the standard loss reaches 94.3 ± 5.4% in fp32 and 93.3 ± 7.4% in bf16, with margins stuck at their precision cliffs (13.0 and 5.1). StableMax reaches 97.3 ± 4.7% with margins in the thousands; the zero-sum projection baseline reaches 49.3 ± 2.5% in the reported setting. Across four tasks at `ε = 1e−12`, fp32 with the fix matches fp64 with the standard loss. In distillation from a perfect teacher on `a + b mod 53`, the standard soft-target loss lets the student drift in bf16 and shows a KL spike in fp32, while the fixed version stays at KL = 0 and 100% accuracy in all precisions.

*Caveats.* Everything is synthetic and small-scale; the authors themselves note that such extreme margins and near-zero student–teacher gaps are uncommon in standard large-scale training. The "no meaningful slowdown" claim comes without timing measurements. Some settings use very few seeds (Table 1's caption says "55 seeds", presumably 5), and the margin equilibrium is derived heuristically rather than proved.

**A small example**

The paper's toy case, recomputed in NumPy with fp32: `z = [9, −9, −9]`, target `t = 0`.

```
margin m = 9 − log(2 e^{−9}) = 18 − log 2 = 17.31   > c_32 ≈ 16.6

standard (max-shifted softmax, g = q − e_t):
  q = [1.0, 1.52e−8, 1.52e−8]       q_0 rounds to exactly 1
  g = [0.0, 1.52e−8, 1.52e−8]       sum = 3.05e−8 ≠ 0     ← target gradient lost

fixed (eqs. 6–7):
  g = [−3.05e−8, 1.52e−8, 1.52e−8]  sum = 0
exact (fp64):
  g = [−3.05e−8, 1.52e−8, 1.52e−8]
```

The bug does not start at the cliff; the standard target gradient becomes coarse before it vanishes. For two classes with margin `m` in fp32:

```
m       exact −e^{−m}/(1+e^{−m})     standard q_t − 1
15.0    −3.06e−7                     −3.58e−7
16.0    −1.13e−7                     −1.19e−7
16.5    −6.8e−8                      −1.19e−7     (same value as at m = 16)
16.7    −5.6e−8                       0.0         (past the cliff)
```

Between `m ≈ 15` and the cliff, the standard gradient can only take a few representable values, so its size stops tracking the margin. Under Adam that matters, because the update is normalised by the gradient's own recent magnitude.

**How it connects**

- **MuZero training losses:** MuZero's policy loss is a soft-target cross-entropy against the MCTS visit distribution, and categorical value and reward heads with two-hot targets are soft-target cross-entropies too. Late in training, when the network matches its targets closely, Softmax Cancellation adds rounding noise to exactly these gradients (the paper measures residuals around `1e−3` in bf16). It is worth checking which precision the loss is computed in, and the soft-target fix is a few lines to test.
- **Calibration:** the margin equilibrium `m* ≈ log(1/ε)` says the final confidence of an AdamW-trained classifier is partly set by the optimizer's `ε` and weight decay rather than by the data. That is a concrete, optimizer-level mechanism for overconfidence, relevant to the calibration taxonomy work. It also means temperature-scaling results can depend on `ε`.
- **Grokking and loss spikes:** a reminder to check numerical causes before interpreting late-training instabilities in RL or supervised runs as learning dynamics.
- No other entries in this repo yet that it connects to directly.

**Check yourself**

1. The standard softmax already subtracts the maximum logit "for numerical stability". Why does that not prevent Softmax Collapse, and what does the fix compute differently?
2. Why does a vanishing target gradient produce a loss *spike* under Adam, while plain SGD with the same bug would not spike?
3. Using the margin-equilibrium argument, what final margin do you expect for AdamW with `ε = 1e−12` and weight decay, and what happens with `λ = 0`? Why does the default `ε = 1e−8` make fp32 training particularly prone to the bug?

<details><summary>Answers</summary>

1. Max-shifting prevents overflow in `exp`, but the information is lost elsewhere: in forming `1 + e^{−m}` for the normalisation, where a tiny `e^{−m}` is rounded away, and in `q_t − 1`, which then subtracts two equal numbers. The fix never forms `q_t` for the target. It computes the margin from the non-target logits relative to the target, `m = −LSE_{j≠t}(z_j − z_t)`, gets the loss as `softplus(−m)`, computes the non-target gradients directly as small numbers and defines the target gradient as their negative sum. Small quantities are always represented directly, never as differences of numbers close to 1.
2. Adam divides by `√v̂ + ε`, a running estimate of the gradient's recent magnitude. While all examples sit past the cliff, gradients are almost zero, so `v̂` decays toward zero. When the target gradient reappears for some examples, it is divided by this tiny denominator and produces a step of order `η` or larger, far bigger than the gradient's size justifies. SGD's step is `η · g`, so the reappearing tiny gradient gives a tiny step and nothing spikes.
3. `m* ≈ log(1/ε) = log(1e12) ≈ 27.6`; Table 1 reports 27.4 for the fixed loss. With `λ = 0` there is no weight-decay term to balance the shrinking normalised step, so margins keep growing past `log(1/ε)` until gradients underflow. With `ε = 1e−8` the equilibrium is ≈ 18.4, just above the fp32 cliff of ≈ 16.6, so the optimizer drives margins into the region where the standard target gradient is exactly zero, and training sits at the edge of collapse.

</details>

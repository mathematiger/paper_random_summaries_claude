---
title: "Theoretical Foundations of Conformal Prediction"
authors: ["Anastasios N. Angelopoulos", "Rina Foygel Barber", "Stephen Bates"]
year: 2024
venue: "Book draft, forthcoming with Cambridge University Press (arXiv:2411.11824 [math.ST], v5 March 2026)"
link: "https://arxiv.org/abs/2411.11824"
added: 2026-10-27
category: "statistics/conformal-prediction"
tags: ["conformal-prediction", "exchangeability", "permutation-tests", "distribution-free", "conditional-coverage", "calibration", "ece", "risk-control", "online-learning", "lower-bounds"]
---

# Conformal foundations — one rank argument under exchangeability gives finite-sample coverage, and a matching set of impossibility results says what no distribution-free method can do

**Main reference:** [Theoretical Foundations of Conformal Prediction — A. N. Angelopoulos, R. Foygel Barber & S. Bates (2024, v5 2026), arXiv:2411.11824, forthcoming book with Cambridge University Press](https://arxiv.org/abs/2411.11824) — a ~260-page proof-oriented textbook, not a paper. Read closely: Chapter 2 (exchangeability, permutation tests, and the order-statistics facts in 2.4, especially Fact 2.15, which every later proof uses) and Chapter 3 (full conformal, Algorithm 3.3; the three-step proof of Theorem 3.2 with the replacement Lemma 3.4; split conformal as a special case, Algorithm 3.6; the p-value view, Proposition 3.9; the overcoverage bound, Theorem 3.11). Then Chapter 4 on conditional coverage (Theorems 4.1, 4.3, 4.5, 4.7, 4.11, and the sample–resample Lemma 4.16, the book's main tool for lower bounds). For calibration work, read Chapter 12 in full: ECE vs binned ECE vs distance to calibration, the hardness results 12.5 and 12.7, and Venn–Abers. Read Chapter 8.3 (quantile tracking) and Chapter 10.1 (conformal risk control) for anything non-exchangeable or loss-based. Skim Chapter 5 (asymptotic optimality via oracle scores), Chapter 6 (cross-conformal, jackknife+), Chapter 7 (weighted conformal, covariate shift), Chapter 9 (randomization, computational shortcuts, universality), Chapter 11 (inference on the regression function) and Chapter 13 (conditional independence testing). It is a draft: the authors warn that numbering and statements may still change, so cite the version. There is no code; the authors point to their earlier *Gentle Introduction* for practice.

**Further reading:**
- [A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification — A. N. Angelopoulos & S. Bates (2021, rev. 2022), arXiv](https://arxiv.org/abs/2107.07511) — the applied companion by two of the same authors, with code and worked score functions; read it for implementation, this book for proofs.
- [Algorithmic Learning in a Random World, 2nd ed. — V. Vovk, A. Gammerman & G. Shafer (2022), Springer](https://doi.org/10.1007/978-3-031-06649-8) — the original monograph on conformal prediction, Mondrian predictors and Venn predictors, in Vovk's game-theoretic language; this book re-derives much of it in a unified statistical notation.
- [Adaptive Conformal Inference Under Distribution Shift — I. Gibbs & E. J. Candès (2021), NeurIPS 2021](https://arxiv.org/abs/2106.00170) — the online threshold update analysed in Chapter 8.3 (Theorem 8.7); the paper adds experiments and the step-size discussion.
- [A Unifying Theory of Distance from Calibration — J. Błasiok, P. Gopalan, L. Hu & P. Nakkiran (2023), STOC 2023](https://arxiv.org/abs/2211.16886) — the theory behind the distance-to-calibration (dCE) measure that Chapter 12 promotes over ECE.

**Why this matters** — Conformal prediction is the standard way to wrap any model in prediction sets with a finite-sample guarantee and no assumptions on the data distribution. Its literature is scattered across papers with different notations, and the proofs are often shorter than they are clear. This book collects the core arguments in one language and, just as important, pairs each positive result with a hardness result that shows where distribution-free guarantees stop: conditional coverage, training-conditional coverage, estimating the regression function, certifying ECE, and testing conditional independence. The transferable lesson is twofold. First, almost everything follows from one fact: under exchangeability, the rank of the test point's score is uniform. Second, many "nice" properties (coverage at every `x`, certified low ECE for a continuous predictor) are provably impossible without assumptions, so it pays to know which ones before trying.

**Core ideas**

*Exchangeability and the rank fact.* A vector `(Z_1, …, Z_n)` is exchangeable if its distribution is unchanged by any permutation of the indices. i.i.d. implies exchangeable, not conversely. A permutation test compares a statistic `T` on the observed order with its values on all permutations,

```
p = (1/n!) · Σ_{σ ∈ S_n} 1{ T(Z_σ(1), …, Z_σ(n)) ≥ T(Z_1, …, Z_n) }      (eq. 2.3)
```

and `P(p ≤ τ) ≤ τ` whenever the data are exchangeable (Theorem 2.4). The workhorse is Fact 2.15: if `Z ∈ ℝ^n` is exchangeable and `Z_(k)` is its `k`-th order statistic, then for each `i`

```
P(Z_i ≤ Z_(k)) ≥ k/n,      P(Z_i < Z_(k)) ≤ (k−1)/n,
```

with equality `P(Z_i ≤ Z_(k)) = k/n` if the entries are almost surely distinct. Conformal prediction is this fact applied to a vector of scores. The book also reads it as a permutation test of the hypothesis "the hypothesised test point is exchangeable with the data" (Section 3.5.1).

*Full conformal prediction (Algorithm 3.3).* Training data `D_n = ((X_i, Y_i))_{i≤n}`, test feature `X_{n+1}`, target level `1 − α`. For a candidate label `y`, form the augmented dataset `D^y_{n+1} = (D_n, (X_{n+1}, y))` and compute scores with a model fitted on all `n + 1` points:

```
S_i^y     = s((X_i, Y_i);      D^y_{n+1}),   i = 1, …, n
S_{n+1}^y = s((X_{n+1}, y);    D^y_{n+1})
q̂^y       = Quantile(S_1^y, …, S_n^y ; (1 − α)(1 + 1/n))
C(X_{n+1}) = { y : S_{n+1}^y ≤ q̂^y }                                       (eqs. 3.2–3.3)
```

The score `s((x, y); D)` must be *symmetric*: it may depend on the dataset `D` but not on the order of its points. Larger means "conforms less". Quantile levels above 1 are read as `+∞`, so for `n < (1 − α)/α` (e.g. `n < 9` at `α = 0.1`) the set is the whole label space.

*Why it covers (Theorem 3.2: `P(Y_{n+1} ∈ C(X_{n+1})) ≥ 1 − α` under exchangeability).* The proof has three steps, and it is worth seeing why each design choice is there. (1) The replacement Lemma 3.4: for any reals, `v_{n+1} ≤ Quantile(v_1..v_n; t(1 + 1/n))` iff `v_{n+1} ≤ Quantile(v_1..v_{n+1}; t)`. So the inflated level `(1 − α)(1 + 1/n)` on `n` scores is exactly the plain `1 − α` level on all `n + 1` scores including the test score. (2) At the true label `y = Y_{n+1}`, the scores are `S_i = s((X_i, Y_i); D_{n+1})` computed by one symmetric function of the whole exchangeable dataset, so the vector `(S_1, …, S_{n+1})` is itself exchangeable. This is why the model is refit with the hypothesised point inside: scores of training points under a model fitted without the test point would be systematically smaller (overfitting) and not exchangeable with the test score. (3) Apply Fact 2.15. Only the scores at the true label matter for the guarantee; the refits at all other `y` exist only because `Y_{n+1}` is unknown. Equivalently (Proposition 3.9), `C(X_{n+1}) = { y : p^y > α }` with the conformal p-value

```
p^y = (1 + Σ_{i≤n} 1{ S_i^y ≥ S_{n+1}^y }) / (n + 1)                        (Def. 3.8)
```

*Split conformal (Definition 3.5, Algorithm 3.6).* If the score is fitted on a separate pre-training set and does not depend on the calibration data, `S_i = s(X_i, Y_i)` no longer depends on `y`, a single quantile `q̂ = Quantile(S_1..S_n; (1 − α)(1 + 1/n))` (the `⌈(1 − α)(n + 1)⌉`-th smallest score) serves all `y`, and the method costs one model fit. The price is statistical: fewer points for fitting and for calibration (Section 3.4.2). The score fixes the shape of the set: absolute residual `|y − f̂(x)|` gives a constant-width band; scaled residual `|y − f̂(x)|/σ̂(x)` adapts the width; the CQR score `max{τ̂(x; α/2) − y, y − τ̂(x; 1 − α/2)}` inflates or shrinks a quantile-regression interval; `−π̂(y | x)` gives "all labels with estimated probability ≥ −q̂" for classification. (Small inconsistency in the draft: the text after the CQR score writes the upper endpoint as `τ̂(x; 1 − α)` once; it should be `1 − α/2`.) Proposition 3.10 rewrites split conformal as choosing the smallest threshold `λ` whose empirical miscoverage is at most `α − (1 − α)/n`, which is the bridge to risk control below.

*How conservative is it (Theorem 3.11, Proposition 3.12).* Coverage is at most `⌈(1 − α)(n + 1)⌉/(n + 1) + ε_tie ≤ 1 − α + 1/(n + 1) + ε_tie`, where `ε_tie` is the probability that the test score ties with a calibration score; with almost surely distinct scores the first inequality is an equality. This bounds overcoverage on the probability scale only; a bad model still gives wide sets, just not wider than its errors require. Randomised tie-breaking (Section 9.1) makes coverage exactly `1 − α`.

*Conditional coverage, positive and negative (Chapter 4).* The guarantee above is marginal over training data and test point. Two refinements are separated carefully.
- Training-conditional: for split conformal with i.i.d. data, the coverage given the calibration set stochastically dominates `Beta((1 − α)(n + 1), α(n + 1))`, so `P(coverage | D_n ≤ 1 − α − Δ) ≤ exp(−2nΔ²)` (Theorem 4.1). Its standard deviation is about `√(α(1 − α)/n)`, which answers "how big should the calibration set be". (My reading: when `(1 − α)(n + 1)` is not an integer, the relevant Beta has the ceiling as first parameter; the domination statement still holds.) For *full* conformal no such guarantee holds: Theorem 4.3 builds a degenerate symmetric score for which, with probability about `α`, conditional coverage is exactly zero. The same construction breaks jackknife+ (Theorem 6.10).
- Test-conditional: with finitely many feature values, running split conformal separately in each group gives `P(Y ∈ C(X) | X = x_k) ≥ 1 − α` (Theorem 4.4). With a nonatomic `P_X` (continuous features), any procedure with distribution-free conditional coverage must satisfy `P(y ∈ C(x)) ≥ 1 − α` for *every* pair `(x, y)`, so for `Y = ℝ` its intervals have infinite length with probability `≥ 1 − α` (Theorem 4.5, Corollary 4.6). Even the relaxed goal "coverage `1 − α` on every feature region of probability `≥ δ`" forces coverage on every region of the joint `(X, Y)` space of probability `≥ δ`, including regions of implausible `y`, which makes intervals as long as an oracle interval at the much higher level `1 − cαδ` (Lemma 4.15, Theorem 4.14). What can be guaranteed is coverage within a finite partition: bins of `X` (Theorem 4.7), labels (Theorem 4.10, needing roughly `1/α` examples per class or the label is always included), or any group function `g(x, y)` (Mondrian conformal, Theorem 4.11). The proofs rest on Lemma 4.8: restricted to the points that fall in a fixed set, an exchangeable sample stays exchangeable.

*The lower-bound engine.* Almost every hardness result in the book (Chapters 4, 11, 12, 13) uses the sample–resample construction (Lemma 4.16): drawing `m` points i.i.d. from `P` is within total variation `m(m − 1)/(2M)` of first drawing `M` points from `P` and then resampling `m` of them with replacement. A distribution-free method must also be valid under the discrete empirical distribution of the `M` points, where it is forced to behave conservatively, and it cannot tell the two worlds apart. This is a reusable proof technique well beyond conformal prediction.

*Model-based view (Chapter 5).* Distribution-free validity says nothing about set size. Chapter 5 adds assumptions only for efficiency: pick an oracle score `s*` whose `(1 − α)`-quantile set `C*` is optimal for the aim (smallest sets, or conditional coverage); if the fitted score `s_n → s*`, then `q̂_n → q*` and `C_n → C*` (Theorems 5.6, 5.8). Section 5.5 gives a double-robustness reading: coverage holds if exchangeability holds *or* if the quantile model is good, shown for CQR on stationary time series (Proposition 5.9).

*Cross-validation methods (Chapter 6).* Cross-conformal, CV+ and jackknife+ reuse all data for fitting and calibration. Their guarantee is `≥ 1 − 2α − O(1/√n)` (Corollaries 6.3, 6.8; jackknife+ exactly `1 − 2α`), proved via a "tournament lemma" (Lemma 6.4). Section 6.2.3 gives an example showing the factor 2 is necessary in the worst case, although empirical coverage is usually close to `1 − α`. Under algorithmic stability the plain jackknife is also covered (Theorem 6.12).

*Weighted and non-exchangeable variants (Chapter 7).* Replace the uniform empirical distribution of scores with weights. Under covariate shift with known likelihood ratio, `w_i ∝ (dQ_X/dP_X)(X_i)` normalised over all `n + 1` points restores `1 − α` coverage (Theorem 7.3); the same holds for label shift and general known shifts (Theorems 7.4, 7.5). With fixed, data-independent weights and *any* joint distribution, coverage is at least `1 − α − Σ_i w_i · d_TV(Z, Z^{(i)})`, where `Z^{(i)}` swaps point `i` with the test point (Theorem 7.13). Down-weighting old points therefore buys robustness to drift at a quantified cost.

*Online settings (Chapter 8).* With exchangeable streams, the online conformal p-values are independent and uniform on `{1/t, …, 1}`, so miscoverage indicators are i.i.d.-like and the running coverage converges to `1 − α` (Theorem 8.2, Corollary 8.3). For arbitrary, even adversarial, sequences the quantile-tracking update

```
q_{t+1} = q_t + η_t · (err_t − α),     err_t = 1{ Y_t ∉ C_t(X_t) }            (eq. 8.8)
```

with scores in `[0, B]` and non-increasing steps gives the deterministic bound `|(1/T) Σ err_t − α| ≤ (B + η_1)/(η_T T)` (Theorem 8.7). The guarantee is on long-run frequency, not per-step probability. Section 8.2 also builds an online test of exchangeability: products of the independent p-values form a supermartingale under exchangeability, and Ville's inequality turns it into an anytime-valid test (Proposition 8.5, Theorem 8.6).

*How general is the framework (Chapter 9).* Theorem 9.7: every symmetric method with distribution-free marginal coverage *is* full conformal with some symmetric score. So for symmetric methods the only lever is the score. Section 9.2 gives exact full conformal for linear regression and Lasso and a discretised approximation in general.

*Beyond coverage (Chapters 10–13).*
- Conformal risk control (Theorem 10.1): for nested sets `C_λ` and a loss in `[0, 1]` that is non-increasing and right-continuous in `λ`, choosing `λ̂ = inf{ λ : R̂(λ) ≤ α − (1 − α)/n }` gives `E[L(Y_{n+1}, C_λ̂(X_{n+1}))] ≤ α`. With the miscoverage loss it is exactly split conformal. Chapter 10 also covers FDR control for outlier detection (conformal p-values are PRDS, so Benjamini–Hochberg applies), selective coverage, and aggregation of conformal sets.
- Inference on the regression function `μ_P(x) = E[Y | X = x]` (Chapter 11): with nonatomic `X`, `Y ∈ [a, b]` and `Var(Y | X) ≥ σ*²`, any distribution-free confidence interval has expected length at least `2(1 − α)σ*²/(b − a)`, which does not shrink with `n` (Theorem 11.5). Relaxing the target (averaging `μ_P` over bins, or blurring it) makes valid, shrinking intervals possible again (Theorems 11.7, 11.8).
- Calibration (Chapter 12, binary `Y`). `ECE(f) = E|E[Y | f(X)] − f(X)|` is discontinuous: with `X ∼ U[0, 1]`, `P(Y = 1 | X) = 1{0.25 < X < 0.75}`, the constant `1/2` is perfectly calibrated, but the injective perturbation `f_ε(x) = (1 − ε)/2 + εx` has `ECE = 1/2` for every `ε > 0` (Example 12.2). Binned ECE is always `≤ ECE` (Jensen, Proposition 12.3), can be arbitrarily smaller (`ε/4` vs `1/2` here), and has a distribution-free upper confidence bound `binECE^ + √(2 log(1/δ)/n)` (Theorem 12.4). ECE has none: any valid upper bound for a nonatomic `f(X)` stays above `E|Y − f(X)|` (Theorem 12.5). Histogram binning post-hoc calibration guarantees `E[ECE(ĥ∘f)] ≤ √(K/(2n))` (Theorem 12.6), but *no injective* post-hoc map, so neither temperature scaling nor any strictly monotone recalibration, can guarantee ECE below `E[Var(Y | X)]` regardless of `n` (Theorem 12.7). The distance to calibration `dCE(f) = inf over perfectly calibrated g of E|g(X) − f(X)|` satisfies `dCE ≤ ECE`, is 1-Lipschitz in `L1(P)`, and has a distribution-free upper bound `dCE^ + 1/K + √(2 log(1/δ)/n)` (Theorem 12.10). Venn–Abers predictors run isotonic regression twice, with the test label set to 0 and to 1, and output the interval of the two fitted values at the test point; the endpoint at the true label is perfectly calibrated (Theorem 12.12).
- Conditional independence testing (Chapter 13): with a nonatomic conditioning variable, any test with valid Type I error for all null distributions has power at most `α` against every alternative (Theorem 13.3); local permutation tests within bins of the confounder work under smoothness.

*Caveats.* This is a theory book: no experiments, no benchmarks, no code, and it says so. Nearly every guarantee is marginal and needs exchangeability (or a known shift, or bounded scores for the online bound), which is exactly what fails in sequential decision-making. The hardness results are worst-case over all distributions; they say what cannot be *guaranteed*, not what typically happens. The draft status means theorem numbers above may move.

**A small example**

Split conformal at `α = 0.1`, computed with NumPy/SciPy (`toy.py`, scratchpad).

```
quantile index k = ⌈(1 − α)(n + 1)⌉ and exact coverage (distinct scores):
n =   8   k =   9 > n      → q̂ = +∞, set is all of Y
n =   9   k =   9          coverage = 9/10    = 0.9000
n =  19   k =  18          coverage = 18/20   = 0.9000
n =  20   k =  19          coverage = 19/21   = 0.9048   (bound 1 − α + 1/(n+1) = 0.9476)
n = 999   k = 900          coverage = 900/1000 = 0.9000
```

With 19 calibration residuals drawn from `|t_3|` (sorted: 0.15, 0.16, 0.20, 0.31, 0.56, 0.79, 0.98, 1.01, 1.09, 1.10, 1.23, 1.23, 1.27, 1.31, 1.37, 1.98, 2.19, **2.93**, 3.54), `q̂` is the 18th value, 2.93, so the interval is `f̂(x) ± 2.93`. The model here is deliberately useless (`f̂ ≡ 0`, heavy-tailed noise); a Monte Carlo run with `n = 20` and 200,000 repetitions gives coverage 0.9058 against the exact 0.9048. Coverage does not depend on model quality, only width does.

Training-conditional spread (Theorem 4.1): the probability that a single calibration set leaves you with coverage `≤ 0.85` is

```
n =  19   Beta(18, 2)      P = 0.1985   (Monte Carlo: 0.1976)   sd ≈ 0.066   Hoeffding bound 0.909
n =  99   Beta(90, 10)     P = 0.0595                            sd ≈ 0.030   Hoeffding bound 0.610
n = 999   Beta(900, 100)   P = 1.3e−6                            sd ≈ 0.009   Hoeffding bound 0.0068
```

So with 19 calibration points, one deployment in five has true coverage below 85% even though the marginal guarantee is exactly 90%. The Hoeffding form `exp(−2nΔ²)` is very loose at small `n`; use the Beta CDF.

The ECE discontinuity of Example 12.2, recomputed on a fine grid with `ε = 0.1`: `ECE(f_ε) = 0.500`, binned ECE with bins `[0, 0.5], (0.5, 1]` is `0.025 = ε/4`, and `E|f_ε − 1/2| = 0.025`, which upper-bounds `dCE(f_ε)`. The model is a 0.025-perturbation of a perfectly calibrated one; ECE calls it maximally miscalibrated, dCE does not.

**How it connects**

- [Softmax fix (Barron et al., 2026)](../../deep-learning/numerical-stability/2026-barron-softmax-fix.md) shows that a network's confidence is partly set by Adam's `ε`. Chapter 12 explains why fixing this post hoc with temperature scaling cannot come with a distribution-free ECE guarantee (Theorem 12.7), while binning or Venn–Abers can.
- [X008 Conformal decision theory](https://github.com/mathematiger/opf_uncertainty_gnn_claude_literature/blob/main/tracks/11-conformal-risk-control/2026-10-15-X008-conformal-decision-theory.md) in the OPF literature repo generalises the quantile-tracking update of Chapter 8.3 to decision losses; this book gives the underlying proof (Theorem 8.7) and the batch version (conformal risk control, Theorem 10.1).
- **Calibration taxonomy paper:** Chapter 12 offers a clean axis for the taxonomy, "can the measure be certified distribution-free?": binned ECE and dCE yes, ECE no for continuous predictors. Add continuity (dCE is 1-Lipschitz, ECE is not) and post-hoc method class (binning vs injective maps). Theorems 12.5 and 12.7 are citable impossibility results, and Proposition 12.3 plus Example 12.2 show that binned ECE is a lower bound that can be arbitrarily loose.
- **MuZero / model-based RL (my inference):** split conformal on value or reward predictions gives calibrated intervals only if calibration and test transitions are exchangeable, which self-play data under a changing policy are not. The book points to the right tools: quantile tracking (Theorem 8.7) for an online, assumption-free long-run guarantee, fixed-weight conformal (Theorem 7.13) to quantify the cost of drift, and Mondrian conformal (Theorem 4.11) for per-group coverage, e.g. per topology class or per grid region. Conformal risk control could calibrate a scalar safety margin for topology actions against a bounded overload loss in Grid2Op/PPtopogym.
- **RL theory:** the rank/permutation argument and the sample–resample lower-bound technique (Lemma 4.16) are general tools for finite-sample, distribution-free statements.

**Check yourself**

1. In full conformal prediction, why are the training scores computed with a model fitted on the augmented dataset that includes the hypothesised test point, and why does the proof only need the scores at `y = Y_{n+1}`?
2. You run split conformal at `α = 0.1` with `n = 19` calibration points and then measure 82% coverage on 2,000 fresh test points. Does this contradict Theorem 3.2? What would make you suspicious?
3. Why can histogram binning come with a distribution-free ECE guarantee while temperature scaling cannot, and how does switching from ECE to dCE change what can be certified?

<details><summary>Answers</summary>

1. Coverage reduces to the event "the test score is at most the `(1 − α)` quantile of all `n + 1` scores", which has probability `≥ 1 − α` if the `n + 1` scores are exchangeable. That holds when all scores come from one symmetric function of the full dataset with the test point inside. A model fitted only on the training points would fit them better than the unseen test point, so training scores would be stochastically smaller and the quantile too low. The scores at other candidate labels are computed only because `Y_{n+1}` is unknown; the coverage event is determined entirely by the scores at the true label.
2. Not necessarily. Theorem 3.2 is marginal over the calibration draw. With `n = 19`, the coverage conditional on the calibration set is distributed as `Beta(18, 2)`, which puts about 20% mass below 0.85 and has sd ≈ 0.066, so 82% with 2,000 test points is plausible for an unlucky calibration set. It would be suspicious if it happened repeatedly across fresh calibration sets, which would point to non-exchangeability (distribution shift, temporal dependence, data leakage into the score) rather than bad luck.
3. Binning makes `ĥ∘f` take finitely many values, so the calibration problem becomes estimating a few bin means, which is possible without assumptions (Theorem 12.6, `E[ECE] ≤ √(K/2n)`). Temperature scaling is injective, so `E[Y | ĥ(f(X))] = E[Y | f(X)]` and certifying low ECE would mean estimating a regression function on a continuous variable to small `L1` error, which is impossible distribution-free (Theorem 12.7, via the Chapter 11 hardness result). dCE measures the distance to the nearest calibrated predictor rather than the pointwise conditional-mean gap. It is 1-Lipschitz, so small perturbations cannot blow it up, and it has a distribution-free upper bound even for continuous `f` (Theorem 12.10).

</details>

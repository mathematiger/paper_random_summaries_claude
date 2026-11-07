---
title: "Beyond Calibration: Estimating the Grouping Loss of Modern Neural Networks"
authors: ["Alexandre Perez-Lebel", "Marine Le Morvan", "Gaël Varoquaux"]
year: 2023
venue: "ICLR 2023 (arXiv v1 Oct 2022, v3 Apr 2023)"
link: "https://arxiv.org/abs/2210.16315"
added: 2026-11-07
category: "machine-learning/calibration"
tags: ["calibration", "grouping-loss", "proper-scoring-rules", "brier-score", "recalibration", "distribution-shift", "multicalibration", "ece", "binning"]
---

# Grouping loss — a calibrated classifier can still be wrong for individuals; the part of the error that calibration cannot see has a name, a decomposition, and a debiased lower-bound estimator

**Main reference:** [Beyond Calibration: Estimating the Grouping Loss of Modern Neural Networks — A. Perez-Lebel, M. Le Morvan & G. Varoquaux (2023), ICLR 2023](https://arxiv.org/abs/2210.16315) — about 9 pages plus a long appendix, from the Soda team at Inria Saclay. Read closely: Section 2 (the definitions of `Q`, `S`, `C` and the counterexample in Fig. 1, a calibrated classifier with optimal accuracy whose scores still differ from the true posterior), Section 3.2 (the CL/GL/IL decomposition, Eq. 6) and all of Section 4 (grouping loss as a variance, the explained/residual split in Thm. 4.1, what binning does in Props. 4.1–4.2, and the debiased Brier estimator in Prop. 4.3). Read Appendix C.5 alongside Prop. 4.3: the derivation there (Eqs. 39–41) is the one to trust for signs. Skim Section 5.1 (simulations) but read Fig. 4 slowly: it shows the plug-in estimate overshooting and the corrected one converging to the bound. In Section 5.2, read Fig. 6 (grouping diagrams for ConvNeXt on ImageNet-R before and after isotonic recalibration) with the definition of the grouping diagram from Fig. 5 at hand, then Fig. 7 (stacked `GL_LB + CL` bars for 15 ImageNet models) and Fig. 8 (BART zero-shot classification, in- vs out-of-distribution). Appendix A collects textbook quotes that blur calibration and posterior correctness and is a quick, fun read. Code: [aperezlebel/beyond_calibration](https://github.com/aperezlebel/beyond_calibration).

**Further reading:**
- [Novel Decompositions of Proper Scoring Rules for Classification: Score Adjustment as Precursor to Calibration — M. Kull & P. Flach (2015), ECML PKDD](https://doi.org/10.1007/978-3-319-23528-8_5) — the origin of the calibration/grouping/irreducible decomposition this paper builds on; it names the grouping loss but gives no estimator.
- [On Calibration of Modern Neural Networks — C. Guo, G. Pleiss, Y. Sun & K. Q. Weinberger (2017), ICML](https://arxiv.org/abs/1706.04599) — the paper that made ECE plus temperature scaling the standard check; the counterpoint, since that pipeline only ever measures and fixes the calibration term.
- [Multicalibration: Calibration for the (Computationally-Identifiable) Masses — Ú. Hébert-Johnson, M. P. Kim, O. Reingold & G. N. Rothblum (2018), ICML](https://arxiv.org/abs/1711.08513) — requires calibration on every subgroup from a rich class; a model that is multicalibrated with respect to the partition used here has zero explained grouping loss on it, so this is the "fix" side of the same problem.
- [Decision from Suboptimal Classifiers: Excess Risk Pre- and Post-Calibration — A. Perez-Lebel, G. Varoquaux, S. Koyejo, M. Doutreligne & M. Le Morvan (2025), AISTATS](https://arxiv.org/abs/2503.18025) — the follow-up that turns CL and GL into decision regret and argues when recalibration is enough and when further training (or multicalibration) is needed.

**Why this matters** — Calibration is checked as "among samples scored 0.7, 70 % are positive". That is a statement about level sets of the score, averaged over everyone who received that score. It says nothing about whether a particular sample's 0.7 is close to its true probability: one subgroup in the level set can be at 0.5 and another at 0.9. Proper scoring rule theory has long split the expected loss into a calibration part, an irreducible part and a third "grouping" part that captures exactly this, but there was no way to estimate the third part, so practice ignored it. This paper characterises the grouping loss, shows why only a lower bound is estimable, gives a debiased estimator for it, and finds non-trivial grouping loss in modern vision and NLP models under distribution shift. The transferable lessons are that a low ECE certifies much less than people think, and that post-hoc recalibration cannot reduce grouping loss at all.

**Core ideas**

*Notation.* Binary case first (`Y ∈ {0,1}`), multiclass later with `Y` one-hot and scores in the simplex `Δ_K`.

```
Q = P(Y = 1 | X)      true posterior (a function of the full input X)
S = f(X)              the classifier's confidence score
C = E[Q | S] = P(Y = 1 | S)   calibrated score: the best you can do knowing only S
```

`S` is calibrated iff `S = C` almost surely. The point of the paper: calibration is `S = C`, while the goal is `S = Q`, and `C ≠ Q` whenever `Q` varies inside a level set `{S = s}`. The Fig. 1 counterexample makes `Q` take different values on two parts of the input space mapped to the same score, keeps both on the same side of 0.5 (so accuracy is optimal), and averages them to `s` (so the classifier is calibrated); the spread can be made as large as one likes.

*The decomposition (Eq. 6).* For a proper scoring rule `φ` with expected score `s_φ(S, Q) = E_{Y~Q}[φ(S, Y)]` and divergence `d_φ(S, Q) = s_φ(S, Q) − s_φ(Q, Q) ≥ 0`:

```
E[d_φ(S, Y)] =  E[d_φ(S, C)]   +   E[d_φ(C, Q)]   +   E[d_φ(Q, Y)]
                 CL (calibration)   GL (grouping)      IL (irreducible)

Brier:   d_BS(S, Q) = Σ_k (S_k − Q_k)²
Log-loss: d_LL(S, Q) = Σ_k Q_k log(Q_k / S_k)      (KL divergence)
```

`IL` is the noise in `Y` given `X`, which no classifier can remove. `CL + GL` is what the paper calls the epistemic loss: everything a better model could remove. Recalibration attacks `CL`; nothing that only looks at `S` can touch `GL`. The decomposition holds for joint and top-label calibration and, for Brier and log-loss, for classwise calibration (Appendix C.9).

*Grouping loss is a conditional variance (Lemma 4.1).* With `h(p) = −s_φ(p, p)` (negative generalised entropy, convex for proper rules) and the "h-variance" `V_h[U | V] = E[h(U) | V] − h(E[U | V])` (a Jensen gap),

```
GL(S) = E[ V_h[Q | S] ]
Brier:    V_h[Q | S] = E[ ||Q − C||² | S ]          (ordinary variance of Q in the level set)
Log-loss: V_h[Q | S] = E[ KL(Q || C) | S ]
```

So `GL` is how much the true posterior still varies among samples that the model cannot tell apart. This is why it is invisible to any reliability diagram: those only see the level-set means `C`.

*Why only a lower bound is estimable (Thm. 4.1).* `Q` is never observed, only `Y`. The way in is to partition each level set into regions using some function `R(X)` and to look at how the label rate varies across regions. By the law of total variance (Jensen-gap version),

```
GL(S) = E[ V_h[ E[Q | S, R] | S ] ]  +  E[ V_h[Q | S, R] ]
         GL_explained(S)                 GL_residual(S) ≥ 0
```

`E[Q | S, R]` is the label rate in a region and is estimable from `Y`. The residual (variation of `Q` inside a region) is not. So any partition yields `GL ≥ GL_explained`, with equality only when `Q` is constant on every region. The quality of the bound is entirely the quality of the partition: it can only find the heterogeneity that the chosen features resolve.

*Binning adds a bias of known sign (Props. 4.1–4.2).* Real scores are continuous, so level sets must be bins. Replacing `S` by its bin mean `S_B` merges level sets, which can only add grouping loss:

```
GL(S_B) = GL(S) + GL_induced(S, S_B),     GL_induced = E[ V_h[C | S_B] ] ≥ 0
⇒  GL(S) ≥ GL_explained(S_B) − GL_induced(S, S_B)  =:  GL_LB
```

The induced term is the variance of the calibration curve `C` inside a bin, which is estimable because `C` is a one-dimensional regression of `Y` on `S`. The paper fits `Ĉ` by LOWESS and averages its within-bin variance (Prop. C.1). The appendix also bounds it: for a calibrated continuous classifier with `N` equal-width bins, `GL_induced ≤ 1/(4N²)` (Lemma C.3), so about 0.0011 for the usual 15 bins. A side result worth remembering for any ECE discussion: binning biases the calibration loss downward (`CL_induced ≤ 0`, Appendix C.8) and the grouping loss upward, so binned estimates shift mass from CL to GL.

*The finite-sample estimator (Prop. 4.3, Brier).* Within bin `s` (and class `k`, dropped here), with `n` samples, regions `j` holding `n_j` samples, region label rate `μ̂_j` and bin label rate `ĉ`:

```
plug-in:   Σ_j (n_j/n) (μ̂_j − ĉ)²
debiased:  Σ_j (n_j/n) (μ̂_j − ĉ)²  −  Σ_j (n_j/n) μ̂_j(1 − μ̂_j)/(n_j − 1)  +  ĉ(1 − ĉ)/(n − 1)
total:     Σ_k Σ_s (n^(s,k)/N) · [debiased]^(s,k),     then subtract the induced term
```

The plug-in is biased upward: even with no heterogeneity, sampling noise makes the region rates differ. The second term removes the noise in the region means, the third adds back the noise in the bin mean (the derivation treats the probability of an empty region as zero). Caveat: the main text prints the last term with a minus sign; the appendix derivation (Eq. 41) has a plus. My toy below confirms that only the plus version is unbiased.

*Choosing the partition (Sec. 4.4).* Inside each bin, fit a decision tree on the labels `Y` with the loss of the scoring rule (squared loss for Brier), using the network's penultimate-layer features as input. Fit the tree on one half of the data and estimate the region rates on the other half; otherwise the tree selects splits that look heterogeneous by chance, and the bound is no longer a bound. The simulations recommend regions of a few dozen samples (the paper's examples use about 12 to 30 samples per region; regions with fewer than 2 samples break the correction). Fig. 6 uses the simplest choice, one balanced stump split per bin; Fig. 7 caps regions at about (bin size)/30. Tree depth, split criteria beyond this and the LOWESS bandwidth are not stated.

*Grouping diagram (Fig. 5).* A reliability diagram with, at each bin, the label rates of the regions inside it (with Clopper–Pearson intervals, regions consistent with the bin mean greyed out). Spread of the region dots around the calibration curve is grouping loss made visible. This is the most practically useful figure in the paper.

*Results.*
- Simulations (Sec. 5.1, Fig. 4): with known `Q` and a calibrated logistic regression, finer partitions give tighter bounds; the uncorrected plug-in overshoots the true GL; the bias and induced corrections bring it to or below the truth. Very small regions make the estimate unstable.
- Vision (Sec. 5.2, Figs. 6–7): 15 torchvision models trained on ImageNet-1K, evaluated top-label on ImageNet-R (30,000 renditions of 200 classes; top-label because each class has only about 50 images). Accuracies are low, roughly 13 % (AlexNet) to 30 % (ViT-B/16) as read from Fig. 7. Most networks show a clearly positive `GL_LB`, and it survives isotonic recalibration while `CL` drops. ConvNeXt-Tiny is the showcase: after recalibration it is calibrated on average, yet over-confident in one region of feature space and under-confident in another, and its grouping loss is larger than its calibration loss. ViT-B/16 has slightly higher accuracy but a lower estimated grouping loss than ConvNeXt. The same picture appears on ImageNet-C; on the in-distribution ImageNet-1K validation set the effect is small or absent. I could not read the per-model values of Table 1 from the available text, so no numbers for `GL_LB` and `CL` are given here.
- NLP (Fig. 8): BART-large trained on MultiNLI, used as a zero-shot topic classifier on Yahoo Answers Topics after fine-tuning on 5 of the 10 topics. No evidence of grouping loss on the seen topics; clear grouping loss on the unseen topics, before and after isotonic recalibration.

*Caveats.*
- Only a lower bound. A zero estimate does not mean zero grouping loss; it means the chosen partition found none. The authors say the bound is for rejecting models and list an upper bound as future work.
- The bound is only as good as the features. Partitioning in the network's own penultimate layer can only reveal heterogeneity that the network represents but collapses when producing the score. Variation of `Q` along directions the network discarded entirely (my inference) stays in the unobservable residual.
- The debiased estimator is derived for the Brier score only; no log-loss estimator is given in the main text.
- The headline vision results are under strong shift where accuracy is at most about 30 %. That grouping loss appears there is plausible; how much it matters for in-distribution use is less clear from these experiments, which mostly show "small or absent".
- Two cross-reference slips (Prop. 4.3 sign; "Prop. C.1" cited for the binning result in Sec. 5.1 while C.1 is the induced-loss estimator) and some unstated hyperparameters make exact replication depend on the code.
- Why recalibration cannot help, in one line (my phrasing of the argument the paper defers to Appendix C.10): if `g` is injective, `g(S)` and `S` have the same level sets, so `C` and hence `GL = E[d(C, Q)]` are unchanged; if `g` merges level sets, as isotonic regression's flat steps do, `GL` can only grow, by the same argument as binning.

**A small example**

Two score levels, `s ∈ {0.2, 0.7}` with equal probability. Inside each level set a hidden group shifts the true posterior to `Q = s − 0.15` or `Q = s + 0.15` (equal halves). So `Q ∈ {0.05, 0.35}` and `{0.55, 0.85}`: always on the same side of 0.5 as `S`, so accuracy is optimal, and `E[Q | S] = S`, so the classifier is perfectly calibrated. Computed with NumPy (`grouping_toy.py`, scratchpad; 20,000 repetitions per estimator row).

```
Population (Brier):   CL = 0      GL = 0.15² = 0.0225      IL = E[Q(1−Q)] = 0.1625
                      Brier = 0.1850   (Monte Carlo with 2·10⁶ samples: 0.18506)
```

ECE is zero, yet 12 % of the Brier score is grouping loss.

*Partition from an imperfect feature.* We observe `Z`, equal to the hidden group with probability 0.8. Splitting each level set on `Z` gives region means `s ± (2·0.8 − 1)·0.15 = s ± 0.09`, so

```
GL_explained = 0.09² = 0.0081      (lower bound, 36 % of the true 0.0225)
GL_residual  = 0.0144              (invisible from Y)
```

*Finite-sample estimators*, mean over repetitions:

```
                    n = 200                         n = 2000
true GL = 0.0225, explained 0.0081
  plug-in           0.00987                         0.00829
  debiased (+)      0.00819 ± 0.00574 (sd)          0.00812 ± 0.00170
  debiased (−)      0.00448                         0.00775
true GL = 0 (δ = 0)
  plug-in           0.00183                         0.00019
  debiased (+)      −0.00002 ± 0.00188              0.00000 ± 0.00019
  debiased (−)      −0.00372                        −0.00037
```

The plug-in invents grouping loss where there is none (0.0018 at `n = 200`, more than a fifth of the real signal), the plus-sign correction is unbiased, and the minus-sign version printed in Prop. 4.3 is biased down. Note the standard deviation at `n = 200`: comparable to the quantity being estimated, which is why the paper insists on large regions.

*Monotone miscalibration.* Report `S' = √S` instead: scores 0.447 and 0.837. Now `CL = 0.0399`, but the level sets are the same, so `GL` stays 0.0225. Isotonic recalibration maps `S'` back to `S`, removing all 0.0399 of `CL` and none of the grouping loss.

**How it connects**

- [`statistics/conformal-prediction/2024-angelopoulos-conformal-foundations.md`](../../statistics/conformal-prediction/2024-angelopoulos-conformal-foundations.md) — the same "right on average, wrong for subgroups" gap in another guise: marginal coverage versus conditional coverage. Grouping loss is the scoring-rule analogue of a conformal set that covers 90 % overall but 70 % in one regime.
- [X006, sources of uncertainty (opf_uncertainty_gnn_claude_literature)](https://github.com/mathematiger/opf_uncertainty_gnn_claude_literature/blob/main/tracks/09-uncertainty-quantification/2026-10-11-X006-sources-of-uncertainty-beyond-aleatoric-epistemic.md) — its omitted-variable argument is the same law-of-total-variance gap: what a model treats as noise given its inputs partly comes from variables it does not see, and it looks well calibrated on average while being wrong per regime.
- **MuZero recalibration (my inference):** `value_recalibration.py` with affine and isotonic maps, and the `prior_top1_ece` diagnostic, act only on the calibration term. A top-1 prior with near-zero ECE after isotonic recalibration can still be over-confident in, say, high-load or post-outage states and under-confident elsewhere. The paper's recipe transfers directly: bin the top-1 prior probability, fit a shallow tree inside each bin on the latent state from the representation network (or on interpretable grid features such as max line loading or number of disconnected lines), fit on half of the calibration set, estimate on the other half, and report `GL_LB` next to ECE. Evaluating on shifted chronics is where the paper predicts the effect will show.
- **Conformal candidate sets (my inference):** the recall-maximising candidate-action sets rest on calibrated probabilities. A positive `GL_LB` means set sizes are miscalibrated per regime even with marginal coverage intact, which argues for region- or Mondrian-conditional calibration over the partition the estimator finds.
- **Calibration taxonomy paper:** the CL/GL/IL decomposition is a natural axis for the taxonomy (what a calibration notion constrains: level-set means only, subgroup means, or the posterior itself), and the signed binning biases (`CL_induced ≤ 0`, `GL_induced ≥ 0`) plus the population versus finite-sample split of the estimator are exactly the kind of distinction worth stating explicitly there.
- **RL theory:** none obvious.

**Check yourself**

1. Why can no post-hoc recalibration map `g(S)`, however flexible, reduce the grouping loss? What is the one thing a map can do to the grouping loss, and when?
2. The estimator ends up as `GL_explained(S_B) − GL_induced`, computed from a tree fit on half the data and a debiasing correction. Name the three ways a naive estimate would be wrong, and say for each whether it would push the estimate up or down.
3. After isotonic recalibration your MuZero prior head has top-1 ECE of 0.01 on the calibration set. Your colleague concludes the prior's probabilities can be trusted state by state. What does this paper say is missing from that conclusion, how would you test it, and what would `GL_LB ≈ 0` on your test actually allow you to claim?

<details><summary>Answers</summary>

1. Grouping loss is `E[d(C, Q)]` with `C = E[Q | S]`, so it depends on `S` only through its level sets. An injective `g` keeps the level sets, hence `C` and `GL`, unchanged; recalibration only moves `S` toward `C`, removing `CL`. A non-injective map (isotonic regression's flat steps, binning) merges level sets, and merging adds the within-merge variance of `C`, so `GL` can only increase. Reducing `GL` needs information beyond `S`, i.e. a different model or features.
2. (a) Finite-sample noise in region means: the plug-in variance of region rates is positive even when there is no heterogeneity, so it pushes the estimate up; removed by the debiasing terms. (b) Binning: merging scores within a bin adds `GL_induced`, pushing the estimate up; removed by subtracting the LOWESS-based induced term. (c) Partition selection on the same data: a tree fit on the evaluation data picks splits that are heterogeneous by chance, pushing the estimate up; removed by the train/estimate split. Separately, any partition misses the residual, which pushes the estimate down relative to the true `GL`; that is why the result is a lower bound.
3. ECE measures `CL` only: among states with prior 0.7, the action is right 70 % of the time on average, but subgroups of states can be far from 0.7. Test it by binning the prior, fitting a tree within each bin on the latent state (or grid features) on one half of the calibration data, estimating debiased region rates on the other half, and plotting a grouping diagram; repeat on shifted chronics. `GL_LB ≈ 0` only says the chosen features found no heterogeneity at this sample size. It cannot certify that the grouping loss is zero, since the residual within regions is unobservable.

</details>

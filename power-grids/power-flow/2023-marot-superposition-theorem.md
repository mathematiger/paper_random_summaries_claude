---
title: "Superposition theorem for flexible grids"
authors: ["Antoine Marot", "Benjamin Donnot", "Noureddine Henka", "Sami Tazi"]
year: 2023
venue: "arXiv preprint (eess.SY), v1 Sep 2023, v2 Feb 2025; v1 titled \"Extended superposition theorem under power grid topological changes\""
link: "https://arxiv.org/abs/2310.04437"
added: 2026-10-28
category: "power-grids/power-flow"
tags: ["dc-power-flow", "distribution-factors", "lodf", "topology-control", "bus-splitting", "remedial-actions", "security-analysis", "grid2op", "lightsim2grid"]
---

# Extended superposition theorem — the power flow after several topology changes is an affine combination of the single-change power flows, with weights from a tiny linear system

**Main reference:** [Superposition theorem for flexible grids — A. Marot, B. Donnot, N. Henka & S. Tazi (2023, v2 2025), arXiv:2310.04437](https://arxiv.org/abs/2310.04437) — a short IEEE-style paper from RTE France (AI Lab and R&D). Read closely: Section II (the statement of the extended theorem, Eq. 2, and the general linear system with "topology sensitivity factors", Eqs. 3–4), Section III (the derivation for line disconnections via a cancelling-flow model, ending in Eq. 19), and Section V (mixed disconnections/reconnections in one system, Eq. 47, and bus splitting/merging as opening/closing a zero-impedance virtual coupler, Eq. 48). Read Section IV (reconnections, written with voltage-angle differences because the line carries no flow in the reference) once for the idea; Appendix A gives the recursive existence argument. Skim Section VI but read Table II, Fig. 3 (AC accuracy) and Table III (speed-ups and "action depth") slowly, keeping in mind the caveats below. Code: [marota/Topology_Superposition_Theorem](https://github.com/marota/Topology_Superposition_Theorem) (pip package `TopologySuperpositionTheorem`, built on Grid2Op and LightSim2Grid).

**Further reading:**
- [Generalized Line Outage Distribution Factors — T. Güler, G. Gross & M. Liu (2007), IEEE Transactions on Power Systems](https://documents.pserc.wisc.edu/documents/publications/papers/2006_general_publications/guler_generalized_lodf.pdf) — the classic multiple-outage LODF method; in the DC setting the paper's disconnection system is (my reading) a rescaled version of it, expressed through stored states instead of a PTDF matrix.
- [Unified algebraic deviation of distribution factors in linear power flow — J. van Dijk, N. Westerbeck, L. Schewe, A. Benigni & D. Witthaut (2024), arXiv](https://arxiv.org/abs/2412.16164) — derives LODFs, line closings, bus splits/merges and multiple simultaneous changes as low-rank (Woodbury) updates of the grid Laplacian, with an islanding criterion; the matrix-based counterpart to this state-based paper, and it cites it.
- [Voltage-sensitive distribution factors for contingency analysis and topology optimization — M. Titz, D. Witthaut, J. van Dijk, B. Petrick & N. Westerbeck (2025), arXiv](https://arxiv.org/abs/2509.19976) — moves distribution factors beyond DC to a voltage-sensitive AC linearisation, addressing exactly the limitation this paper only tests empirically.
- [Learning to run a power network challenge for training topology controllers — A. Marot et al. (2020), Electric Power Systems Research](https://arxiv.org/abs/1912.04211) — the L2RPN/Grid2Op setting whose combinatorial topology actions motivate the method.

**Why this matters** — Topology control (switching lines, splitting and merging busbars) is a cheap flexibility lever, but the action space is combinatorial: every combination of switching actions changes the network matrix, so in principle every candidate needs its own power flow. Distribution factors (PTDF, LODF) avoid that for line outages but need the network matrices and have been handled case by case for reconnections and bus splits. This paper gives one statement that covers all four change types: in DC, the flows after `n` simultaneous changes are an affine combination of the reference flows and the `n` single-change flows, with weights solved from an `n × n` system built only from values read off those states. The transferable lesson is that, once the unitary actions have been simulated, evaluating their combinations is almost free and exact in DC, and the weights themselves tell you how strongly actions interact.

**Core ideas**

*The setting.* DC power flow (lossless, flat voltage magnitudes, small angle differences): line flow `pf_l = σ_l · Δθ_l` (Eq. 5), with `σ_l` the line's susceptance (the paper calls it conductance) and `Δθ_l` the voltage-angle difference across it. `PF(P, T)` is the full power-flow state for nodal injections `P` and topology `T`. The classical superposition theorem (Eq. 1) is linearity in injections at fixed topology:

```
PF(P_ref + Σ_k ΔP_k, T_ref) = PF(P_ref, T_ref) + Σ_k PF(ΔP_k, T_ref)
```

It says nothing about changing `T`. The grid is assumed to stay one connected component.

*The extended theorem (Eq. 2).* For a target topology `T_tgt` reached from `T_ref` by unitary changes `τ_1, …, τ_n` (each a line opened or closed, or a bus split or merged), with injections unchanged,

```
PF(P_ref, T_tgt) = α · PF(P_ref, T_ref) + Σ_{i=1..n} β_i · PF(P_ref, T_{τ_i}),     α = 1 − Σ_i β_i
```

where `T_{τ_i}` is the reference topology with only change `τ_i` applied. The `β_i` solve

```
M β = 1,     M_jj = 1,     M_ji = tsf_{l_j}^{τ_i} = 1 − x_{l_j}^{τ_i} / x_{l_j}^{ref}    (j ≠ i)      (Eqs. 3, 4, 19, 47)
```

with `l_j` the asset changed by `τ_j` and `x` either the flow `pf` or the angle difference `Δθ` on that asset. (Eq. 4 as printed has inconsistent subscripts; the form above is the one that the derivation in Eq. 19 gives, and the one my toy example checks.) Only `n(n − 1)` scalars are needed to build `M`: the value on each changed asset in each other unitary state, and its reference value. No admittance matrix, PTDF matrix or grid topology is required, which is the paper's main practical selling point.

*Why it holds for disconnections (Section III).* Opening line `o` is equivalent to keeping it in the reference network and injecting an opposite pair of powers at its two ends, sized so that the injected transfer exactly matches the flow the line would carry: the line then carries net zero to the rest of the grid (the "cancelling flow" `cf_o`). Because the topology stays `T_ref`, the classical theorem applies: each cancelling transfer adds `cf_o` times a fixed response vector (the LODF column of `o`). The unitary state `PF(T_o)` is the special case with one transfer of size `pf_o^ref`, so the response vector is `(PF(T_o) − PF(T_ref)) / pf_o^ref`. Several outages are several transfers, hence

```
PF(T_tgt) = PF(T_ref) + Σ_o β_o (PF(T_o) − PF(T_ref)),     β_o = cf_o^tgt / pf_o^ref      (Eq. 11)
```

which is Eq. 2 with `α = 1 − Σ β`. The `β` follow from requiring zero flow on every opened line in the combined state: for each opened `l`, `(1 − Σβ) pf_l^ref + Σ_{o≠l} β_o pf_l^{T_o} = 0`, which divided by `pf_l^ref` is `M β = 1`. One outage gives `β = 1, α = 0` (just the unitary state). For two, solving the 2×2 system gives `β_1 = (1 − a)/(1 − ab)`, `β_2 = (1 − b)/(1 − ab)` with `a = M_12`, `b = M_21` (my algebra). With `lodf_{o,l} = (pf_l^{T_o} − pf_l^ref)/pf_o^ref`, the off-diagonal entries are `M_lo = −lodf_{o,l} · pf_o^ref / pf_l^ref`, so `M = D⁻¹ (I − Λ) D` with `D = diag(pf^ref)` and `Λ` the off-diagonal LODF block. That is the matrix of the generalised-LODF method up to diagonal scaling (my inference, not stated in the paper). So in DC the result is algebraically the known multiple-outage formula; the contribution is the state-based form and its extension below.

*Reconnections (Section IV).* Here the reference has the lines open, so their reference flows are zero and the flow ratio is undefined. The paper writes the same equations with angle differences in the opened-line reference, `tsf_l^{τ=l_i} = 1 − Δθ_l^{τ=l_i} / Δθ_l^{T_O}` (Eq. 34): `σ_l Δθ_l` is the "virtual flow" the line would carry if closed with no other effect. Existence is shown by inverting the disconnection theorem with roles of reference and target swapped (for two lines, `α_R = 1/α_O` and `β_R^{l1} = −β_O^{l2}/α_O`, `β_R^{l2} = −β_O^{l1}/α_O`; recursively in Appendix A), which needs `α_O ≠ 0`. A line with `Δθ = 0` in the reference is reconnected with zero flow, gets `β = 0`, and drops out of the system. The paper also defines a line-closing factor (LCDF) by analogy with the LODF; its printed formula (Eq. 35) is garbled in the version I read.

*Mixed changes and busbar actions (Section V).* Disconnections and reconnections go into one `(n_O + n_R) × (n_O + n_R)` system (Eq. 47), using `pf`-ratios where the reference flow is non-zero and `Δθ`-ratios otherwise. A bus split is modelled as opening a zero-impedance virtual coupler between the two busbars; the role of its reference flow is played by the sum of the flows of the elements that will move to one side (Eq. 48). A bus merge is closing that coupler, with the angle difference between the two nodes in place of the flow. How a substation reconfiguration that moves several elements maps onto couplers is not spelled out; in Grid2Op terms one "set bus" action on a substation becomes one coupler opening if it creates exactly two busbars (my reading).

*Interpretability.* Comparing `β` to the naive guess "add the single-action changes" (all `β = 1`, i.e. `α = 1 − n`): `β_i ≈ 1` means action `i`'s effect does not depend on the others (the paper says "electrically distant"), and deviations measure interaction. Table II illustrates this with two pairs of IEEE-14 line outages. Caveat: the printed table contradicts its own caption and text. The caption says lines from separate "clicks" (undefined; from context, substations or clusters) give `β ≈ 1` and same-click lines deviate, but the table shows `β = (1.02, 1.00)` for the same-click pair (l1-2, l1-3) and `β = (1.52, 1.63)` for the separate-click pair (l1-2, l11-12). Either rows or values are swapped. My toy below also shows that `β` near 1 is not simply about distance: two lines in series can make one `β` near zero.

*Results.* All timings are against LightSim2Grid (C++, KLU-based) as reference solver.
- Validation, IEEE 14 (Table I): five combined-action configurations mixing two line changes and two substation splits/merges reproduce solver flows to at least 4 decimals; speed-ups 2.2× to 6.6×, the slowest being the mixed-type configuration. The table lists configurations 0, 1, 2, 3, 5 (no 4), and the extracted layout does not allow mapping every coefficient to its column.
- Combined remedial actions (Table III), median over 100 trials per grid, two combined actions: 3.1× (IEEE 118), 6.2× (300), 24.4× (1354pegase), 28.1× (1888rte), 47.6× (3120sp), 42.7× (2848rte), 51.0× (2869pegase). "Action depth", the number of combined actions before EST stops being faster than a power flow, grows from 10 to 29 with grid size, against an operational practice of about 10 manoeuvres. The speed-up grows with grid size because the EST cost depends on `n`, not on the number of buses.
- AC (Fig. 3): applying the same formula to AC-computed unitary states, 100 random triple line outages per grid: for nearly independent combinations the error is at most about 0.1 MW; for interacting ones (`|β − 1| ≥ 0.1`) mostly below 1 MW, a few up to 10 MW. A pure DC power flow, for comparison, is often off by more than 10 MW and up to about 100 MW. Grids are not named in the text.
- Security analysis after a topology action (Fig. 4): N-1 line outages are recomputed after two random non-islanding actions per grid. EST reuses one small system per contingency, scales linearly in the number of contingencies and parallelises. It is orders of magnitude faster than pandapower, faster than compiled solvers above about 100 buses, at least 10× faster than KLU-based security analysis above 1,000 buses, and about 5× faster than an optimised C++ LODF approach, although EST is in Python.

*Caveats.*
- The theory is DC only. The AC results are an empirical observation on active-power flows; voltages, reactive power and losses are not modelled, and the authors leave larger AC discrepancies to future work.
- Timings appear to assume that the unitary states already exist (the text says EST reuses them without solving power flows again; no timing breakdown is given). That is fair when an agent or operator simulates all unitary actions anyway, as many L2RPN agents do, but the speed-up is not "combined-action power flow from scratch".
- Islanding is excluded by assumption, and conditioning of `M` is not discussed. In my toy, every pair of outages that islands a bus gives `det(M) ≈ 0`, consistent with the generalised-LODF islanding criterion; near-islanding combinations would give huge `β` and loss of accuracy.
- Division by a reference flow or angle difference that is (near) zero is handled by switching between `pf` and `Δθ` ratios or by dropping the asset, but numerical thresholds are not given.
- Not peer-reviewed as far as I can tell; the arXiv page lists no venue.

**A small example**

A 5-bus DC grid in which substation 1 has two busbars (1a, 1b) joined by a coupler of reactance `1e−6`; computed with NumPy (`est_toy.py`, scratchpad). Lines (from, to, reactance): 0-1a (0.10), 0-2 (0.20), 1a-2 (0.10), 1b-3 (0.10), 2-3 (0.25), 3-4 (0.10), 2-4 (0.30). Injections `P = (1.5, 0.2, −0.6, 0.3, −1.4)`, bus 0 slack.

*Two line outages, 1a-2 and 2-3.*

```
                     0-1a    0-2   1a-2   1b-3    2-3    3-4    2-4
reference flows    0.8561 0.6439 0.4316 0.6246 0.0772 1.0018 0.3982
only 1a-2 open     0.6583 0.8417 0      0.8583 -0.0667 1.0917 0.3083
only 2-3 open      0.8696 0.6304 0.3913 0.6783 0      0.9783 0.4217

M = [[1, 1 − 0.3913/0.4316], [1 − (−0.0667)/0.0772, 1]] = [[1, 0.0933], [1.8636, 1]]
β = (1.0976, −1.0454),   α = 1 − Σβ = 0.9479

EST:   0.625 0.875 0 0.825 0 1.125 0.275     (matches a direct double-outage power flow to 8e−12)
naive (all β = 1) is off by up to 0.144
```

Opening 1a-2 makes line 2-3 reverse its small flow, so the second outage's unitary state overstates its own effect; the negative `β` corrects that. Strong interaction, not distance, is what moves `β` away from 1.

*Line outage plus bus split.* Opening the coupler (its reference flow 0.6246 equals the flow of line 1b-3, the only element on busbar 1b, as Eq. 48 says) together with opening 0-2: `M = [[1, −0.2425], [−0.2913, 1]]`, `β = (1.337, 1.390)`, `α = −1.726`, and the combination again matches the direct power flow to `1e−11`. (This split leaves line 1b-3 alone on busbar 1b, so it ends up carrying no flow; the large `β` reflect that both changes reroute the same power.)

*Reconnection.* Starting from the double-outage state and closing 1a-2 and 2-3 again, the `Δθ`-based system gives `β_R = (1.1029, −1.1579)` and `α_R = 1.0550`, which equals `1/α_O = 1/0.9479`, as the two-line inversion formula predicts, and recovers the reference flows to `8e−12`.

*Islanding.* Opening 3-4 and 2-4 together isolates bus 4: `det(M) = −2e−15`. Opening both lines at the slack bus gives `det(M) = 9e−11`. A non-islanding pair (1a-2, 2-3) has `det(M) = 0.83`.

**How it connects**

- No closely related entry in this repo yet; this opens `power-grids/power-flow`.
- **MuZero for topology control (my inference):** in Grid2Op/L2RPN and PPtopogym, an agent often simulates every unitary action at the current step anyway. With those states cached, EST evaluates any combination of `k` of them by a `k × k` solve, exactly in DC and close in AC for active flows. That makes it a candidate for (a) a cheap, exact "true model" to benchmark or supervise MuZero's learned dynamics on combined actions, (b) expanding multi-action nodes inside `ap3_mcts` without calling the solver, and (c) action-space design: busbar splits enter as coupler openings, so substation actions fit the same machinery. The caveat is that MuZero's search unrolls in time, while EST combines actions at one time step with fixed injections; it does not replace a dynamics model across steps.
- **Action interaction as a feature (my inference):** `β` and `det(M)` give an interpretable, computed measure of how two actions interact and whether they island the grid. That could prune combinations in search (treat near-`β = 1` pairs as additive), flag dangerous combinations, or serve as input to a learned prior over action pairs.
- **Calibration / RL theory:** none obvious.

**Check yourself**

1. Why must the coefficients satisfy `α + Σ β_i = 1`, and why is the result linear in the unitary states at all, given that power flow is not linear in topology?
2. For reconnections, the paper switches from flow ratios to angle-difference ratios. What goes wrong with flow ratios, and what is the physical meaning of `σ_l Δθ_l` in the reference state?
3. You have the unitary states of 50 single actions at the current time step and want to rank all pairs. What does EST cost compared with running 1,225 power flows, what does it assume about those pairs, and what would `β_1 ≈ β_2 ≈ 1` versus `det(M) ≈ 0` tell you?

<details><summary>Answers</summary>

1. In DC, opening line `o` is equivalent to injecting a transfer between its ends in the unchanged reference network. At fixed topology, flows are linear in injections, so the combined state is the reference state plus a sum of transfer responses, and each response is proportional to `PF(T_o) − PF(T_ref)`. That gives `PF(T_ref) + Σ β_o (PF(T_o) − PF(T_ref))`, an affine combination whose coefficients sum to one by construction: all states share the same injections, so the weights on full states must add to one to keep the reference part intact. Nonlinearity in topology is absorbed into the `β`, which depend on the combination through the small system `M β = 1`.
2. A line open in the reference carries zero flow there, so `1 − x^τ / x^ref` divides by zero. The angle difference across the open line is generally non-zero, and `σ_l Δθ_l` is the flow the line would carry if closed without the grid readjusting, which plays the same role as the reference flow does for an outage. If that angle difference is zero, closing the line changes nothing, so its `β` is zero and it leaves the system.
3. Each pair needs only the four cross values (each line's value in the other's unitary state and in the reference), a 2×2 solve and a weighted sum of three stored flow vectors; no network matrix is factorised. That is a small fraction of 1,225 power flows on a large grid (the paper reports median 24–51× per evaluation above 1,000 buses), assuming the 50 unitary states are already computed, injections are fixed, the DC model or an AC approximation is acceptable, and the pair does not island the grid. `β ≈ (1, 1)` means the two actions act independently, so their single-action effects simply add. `det(M) ≈ 0` signals that the combination islands (or nearly islands) the grid, so EST is invalid there and the action pair is likely unsafe.

</details>

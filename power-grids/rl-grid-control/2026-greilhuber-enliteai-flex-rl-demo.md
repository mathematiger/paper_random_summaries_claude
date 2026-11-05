---
title: "EnliteAI FLEX RL demo: RL agent with GNN, bus-wise critic and AC-checked MCTS for medium-voltage grid control (Enlit Europe 2026)"
authors: ["Andreas Greilhuber (post author)", "enliteAI"]
year: 2026
venue: "LinkedIn post announcing a live demo at Enlit Europe 2026, Vienna, 10-12 Nov 2026 — not a paper; no preprint linked"
link: "https://www.linkedin.com/posts/andreasgreilhuber_enliteurope-powersystems-gridoperations-ugcPost-7523130642113601536-QWuo/"
added: 2026-11-06
category: "power-grids/rl-grid-control"
tags: ["reinforcement-learning", "mcts", "gnn", "distribution-grid", "volt-var-control", "oltc", "curtailment", "reactive-power", "ac-power-flow", "decision-support", "industry-demo", "no-paper-yet"]
---

# EnliteAI FLEX RL demo — an AlphaZero-style agent moves from transmission topology control to hourly volt/VAR and curtailment control on MV feeders, with every search branch checked by an exact AC power flow

> **Status (2026-11-06):** this entry describes an *announcement*, not a paper. No preprint, report or code for the demo was found (searched enliteAI's publication list and the web). A follow-up check after Enlit Europe (10–12 Nov 2026) is scheduled; if a paper appears it gets its own entry and this one gets a link, otherwise the demo news is added here.

**Main reference:** [EnliteAI FLEX RL demo — post by A. Greilhuber (enliteAI), LinkedIn, Nov 2026](https://www.linkedin.com/posts/andreasgreilhuber_enliteurope-powersystems-gridoperations-ugcPost-7523130642113601536-QWuo/) — a short marketing post (about one screen) for the enliteAI booth at Enlit Europe, Messe Wien, 10–12 Nov 2026. This entry is based on the post text; the attached video was not transcribed. Read it for: the control problem (three hourly controls per MV grid), the four architectural components (GNN encoder, bus-wise critic, three-hour MCTS with fully solved power flows in every branch, online training), and the two benchmark numbers. Everything else (losses, network sizes, search budget, reward, train/test split) is not stated. Background on the product and the underlying framework: [enliteAI technology page](https://www.enlite.ai/product/technology) and [Maze RL framework](https://github.com/enlite-ai/maze).

**Further reading:**
- [Power Grid Congestion Management via Topology Optimization with AlphaZero — M. Dorfer, A. R. Fuxjäger, K. Kozak, P. M. Blies & M. Wasserer (2022), NeurIPS 2022 RL4RealLife Workshop](https://arxiv.org/abs/2211.05612) — the direct ancestor: PUCT-MCTS with a policy prior on Grid2Op, 1st place in L2RPN WCCI 2022; same "search + physics simulator + human-reviewable proposal" pattern, but transmission topology instead of distribution volt/VAR. [Code](https://github.com/enlite-ai/maze-l2rpn-2022-submission).
- [Reinforcement Learning Based Power Grid Day-Ahead Planning and AI-Assisted Control — A. R. Fuxjäger, K. Kozak, M. Dorfer, P. M. Blies & M. Wasserer (2023), arXiv](https://arxiv.org/abs/2302.07654) — wraps the AlphaZero agent into day-ahead planning and real-time operator assistance with a redispatch/switching preference `α`; the "agent proposes, operator decides" workflow the demo repeats.
- [Towards Efficient Multi-Objective Optimisation for Real-World Power Grid Topology Control — Y. El Manyari, A. R. Fuxjäger, S. Zahlner, J. van Dijk, A. Castagna, D. Barbieri, J. Viebahn & M. Wasserer (2025), arXiv; listed by enliteAI as ACM e-Energy 2025](https://arxiv.org/abs/2502.00034) — with TenneT: PPO and AlphaZero agents plus a planner that trades max N-1 loading against number of switches on real TenneT data; the latest peer-reviewed step before this demo.
- [A Conceptual Framework for AI-based Decision Systems in Critical Infrastructures — AI4REALNET consortium (2025), IEEE SMC 2025](https://ieeexplore.ieee.org/document/11342626) — the human-in-the-loop framing ("decision support", operator keeps authority) behind the "Physical AI Harness" wording.

**Why this matters** — Most RL-for-grids work, including enliteAI's own papers and the L2RPN/Grid2Op line, is about *transmission topology*. Distribution operators face a different problem: rising PV and load flexibility cause over- and undervoltages and thermal overloads on radial MV/LV feeders, and the levers are continuous-ish set-points (transformer tap position, inverter reactive power, curtailment) rather than busbar switching. The demo claims that the same recipe — learned prior/critic, tree search over a short horizon, exact AC power flow as the model — transfers to this setting and trains in minutes on standard benchmarks. If it holds up, the transferable lesson is that the search-with-a-true-simulator part, not the action type, carries the method; and that a *per-bus* critic is a natural way to make value estimates localisable and explainable on a grid graph.

**Core ideas**

*The control problem (from the post).* An agent controls a medium-voltage grid model in hourly steps. Per hour it sets the on-load tap changer (OLTC) at the HV/MV transformer, and curtailment and reactive power per feeder. The objective is not stated; from the reported metric, it is at least the number of limit violations (voltage band and/or line loading — the post does not say which limits are counted), presumably with costs on curtailment, since the SimBench result is advertised as using no curtailment. A plausible formalisation (my inference):

```
state      s_t   = grid graph with bus/line/trafo features at hour t (loads, PV, tap, voltages, loadings)
action     a_t   = (Δtap_t, {q_f,t}_{f∈feeders}, {c_f,t}_{f∈feeders})
model      s_{t+1} = ACPF(s_t, a_t, forecast_{t+1})       # exact AC power flow, not learned
reward     r_t   = −#violations(s_{t+1}) − λ_c Σ_f c_f,t − λ_q Σ_f |q_f,t| − λ_tap |Δtap_t|
```

`ACPF` is a full AC power flow (enliteAI's stack runs pandapower, OpenDSS or PyPowSyBl per its technology page), `c_f` a curtailment fraction and `q_f` a reactive set-point for feeder `f`, `λ` unknown cost weights.

*Encoder: a GNN on the grid graph.* Buses as nodes, lines and transformers as edges, so the same network applies to grids of different size and the per-bus outputs below come for free. Architecture, depth and features are not stated.

*Bus-wise critic.* Instead of one scalar value `V(s)`, the critic outputs a value (or predicted trouble) per bus over the rest of the day. One way to read this (my inference): `V(s) = Σ_b v_b(s)`, with `v_b` trained against the future violation count at bus `b`. Why it is a good choice: the reward is a sum over buses, so per-bus heads give a denser, better-conditioned learning signal than one scalar; and the per-bus map is directly showable to an operator ("trouble expected at feeder 3 this evening"). Whether the heads are summed for search, used as auxiliary targets only, or both is not stated.

*Search: MCTS three hours ahead, each branch a solved power flow.* Like the AlphaZero agent of 2022, the dynamics are the simulator, not a learned model; the network supplies priors and leaf values. Every expansion runs an AC power flow, so all visited states are physically consistent and violations are exact rather than predicted. The depth-3 horizon matches the main temporal coupling in this problem: the OLTC moves only a step at a time, so the agent must start moving taps before PV peaks or the evening load ramp. Why search at all when the controls are few: per-feeder controls multiply. With a coarse discretisation of 3 tap moves and two levels each for Q and curtailment, one feeder has 12 actions per hour, but 8 feeders have 196,608 per hour and about 7.6·10¹⁵ three-hour sequences (computed below). Exhaustive search is out; MCTS with a prior that factorises over feeders is the obvious fix. Search budget, discretisation, and how the joint action is factorised are not stated.

*Training.* The post says the grid "visibly calms down" while the agent trains live, and gives about one minute of training for CIGRE MV. With an exact simulator and MCTS, much of the early improvement can come from search alone; the post does not separate search from learned components (no ablation).

*Results (as claimed in the post, single test day each).*
- CIGRE MV benchmark: limit violations on the test day fall from 72 to 9 after about one minute of training.
- SimBench MV rural (post: 99 buses, 8 feeders, 42 MW generation): from 95 to 0, using tap and reactive power only, no curtailment on the test day.
- Field experience claimed for HV with RTE and TenneT and for MV/LV with Austrian DSOs (Wiener Netze, Netz Burgenland, Netz Niederösterreich, Stadtwerke Kapfenberg, Kraftwerk Haim); for HV this is backed by the papers above, for MV/LV no publication was found.

*Caveats.*
- No paper, no code for this demo, no reviewers. All numbers are vendor claims.
- One test day per grid; no variance, no seeds, no out-of-distribution days (contrast the TenneT paper, which uses 31 in- and 181 out-of-distribution days).
- "Violations" is undefined: counted per bus-hour, per element-hour, voltage only or also thermal? The baselines (72, 95) are presumably the no-action case; no comparison against a standard volt/VAR controller (local Q(U) droop, OLTC with line-drop compensation, or an OPF) is given — those are the baselines a DSO would ask for.
- "One minute of training" says nothing about the search budget per decision, which is where most of the compute goes when every branch is a full AC power flow.
- Tap changer and reactive power alone reaching zero violations on SimBench MV rural is plausible for a voltage-limited rural feeder; it says little about thermally limited grids where curtailment is the only lever.

**A small example**

A toy version of the problem, computed with NumPy (`flex_toy.py`, scratchpad): a 6-bus radial MV feeder (5 segments, `z = 0.035 + 0.025j` pu each), loads at buses 1–5 with an evening peak, PV at buses 3–5 (0.10/0.25/0.35 pu peak) with a midday peak, AC power flow by backward-forward sweep, voltage band 0.95–1.05 pu. Controls per hour: tap move `{−1, 0, +1}` (1.25 % per step, position limited to ±4), reactive absorption by the PV inverters on/off (`q = −0.35·p_pv`), 30 % curtailment on/off — 12 joint actions. Costs: 1 per violation (bus-hour), 0.05 per tap move, 0.1 for Q, 0.5 for curtailment.

```
no action, 13:00  voltages 1.000 1.013 1.028 1.045 1.062 1.072   -> 2 overvoltages
no action, 19:00  voltages 1.000 0.981 0.967 0.957 0.950 0.948   -> 1 undervoltage

policy                      violations/day   curtailed hours   AC power flows
no action                         7                 0                   0
greedy, 1 hour ahead              2                 0                 288
exhaustive, 3 hours ahead         0                 0             119,232
```

Why the one-hour policy fails: it lowers the tap to −2 for the midday PV peak and is still at −2 at 16:00, because that hour has no violation. At 17:00 the load ramps up, but the tap can only climb one step, so the two buses at the feeder end fall below 0.95 pu (minimum 0.944 pu, 2 violations). The three-hour search sees the ramp coming and starts raising the tap at 16:00. This is the rate-limited coupling that makes a multi-hour horizon worth its cost, and the reason the demo's three-hour MCTS is a sensible design. Neither policy needs Q or curtailment here: the tap alone is enough once it is timed right, which matches the SimBench claim of zero violations without curtailment.

Cost of the exhaustive version: 12³ = 1,728 sequences per decision and 119,232 power flows per day on one feeder. With per-feeder Q and curtailment on 8 feeders, the per-hour action count is 3·2⁸·2⁸ = 196,608 and the three-hour count about 7.6·10¹⁵: hence MCTS with a learned prior.

**How it connects**

- [Superposition theorem for flexible grids](../power-flow/2023-marot-superposition-theorem.md) — also about making many power-flow evaluations cheap inside a search; there for combined topology actions in DC, here every branch is a full AC solve, which is exactly the cost such shortcuts target.
- **MuZero / model-based RL (my inference):** this is AlphaZero-style search with the *true* simulator, the design MuZero replaces with a learned dynamics model. The demo is a useful reference point for the PPtopogym/`ap3_mcts` work: on small MV grids, solving an AC power flow per node is affordable and gives exact violation signals, so the question "when does a learned model pay off?" becomes concrete — larger grids, longer horizons, or wider search where per-node power flows dominate cost. The bus-wise critic is a design worth copying for a MuZero value head on a GNN (per-bus value/violation heads, summed for search, shown for interpretability).
- **Power grid topology control (Grid2Op/L2RPN):** same company and search recipe as the 2022 L2RPN winner; the action space shifts from switching to set-points, so the prior network has to handle a factorised, multi-feeder action — the same issue as combining unitary topology actions.
- **Calibration (my inference):** if the bus-wise critic is meant as an operator-facing risk map of where trouble will appear later in the day, its outputs should be calibrated per bus; nothing in the post addresses that. A natural evaluation would be reliability diagrams of predicted vs realised violations per bus.
- **RL theory:** none obvious.

**Check yourself**

1. Why does a three-hour lookahead help even though each control acts immediately on the current hour's voltages? Which property of the OLTC creates the coupling between hours?
2. What is gained by a per-bus critic compared with a single scalar value, both for learning and for the operator? What would you need to check before showing its outputs as risk estimates?
3. The demo runs a full AC power flow in every search branch. When is this preferable to a learned dynamics model (MuZero-style), and when does it break down?

<details><summary>Answers</summary>

1. The tap position is a state that changes at most a step per hour. A good tap for the midday PV peak (low) is bad for the evening load peak (high), and the transition takes several hours. A one-hour policy only acts once a violation is imminent and then cannot move fast enough; the toy shows exactly that at 17:00. Q and curtailment act instantly and have no such coupling, which is why the horizon matters mainly through the tap.
2. The reward is a sum of per-bus violations, so per-bus heads get a target for each bus instead of one noisy sum, which typically speeds up and stabilises learning, and the network can share structure across similar buses via the GNN. For the operator, the per-bus output localises the problem. Before using it as a risk map you would check per-bus calibration (do buses predicted at 30 % trouble actually see trouble about 30 % of the time?), its behaviour out of distribution (new days, new feeders), and whether it is trained on the policy's own future or on the no-action future — the two answer different questions.
3. An exact simulator gives exact constraint checks and no model error, which matters for safety-critical decisions and is affordable when the grid is small, the horizon short and the search narrow. It breaks down when each power flow is expensive relative to the decision deadline (large grids, wide factorised actions, long horizons) or when the future injections are uncertain, since the simulator still needs forecasts. A learned model is cheaper per node and can absorb stochasticity, but its errors compound over the unroll and its violation predictions are approximate, so a hybrid (learned model for search, exact power flow to verify the chosen action) is a common compromise.

</details>

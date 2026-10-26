---
title: "VRPAgent: LLM-Driven Discovery of Heuristic Operators for Vehicle Routing Problems"
authors: ["Federico Berto", "André Hottung", "Chuanbo Hua", "Nayeli Gast Zepeda", "Haoran Ye", "Michael Römer", "Daniel Wetzel", "Davide Zago", "Michael Poli", "Stefano Massaroli", "Jinkyoo Park", "Kevin Tierney"]
year: 2025
venue: "arXiv preprint (under review)"
link: "https://arxiv.org/abs/2510.07073"
added: 2026-10-26
category: "combinatorial-optimization/llm-heuristic-discovery"
tags: ["vehicle-routing", "large-neighborhood-search", "llm-code-generation", "genetic-algorithm", "program-search"]
---

# VRPAgent — Let the LLM write only two small operators, and evolve them inside a fixed search

**Main reference:** [VRPAgent: LLM-Driven Discovery of Heuristic Operators for Vehicle Routing Problems — F. Berto, A. Hottung, C. Hua, N. Gast Zepeda, H. Ye, M. Römer, D. Wetzel, D. Zago, M. Poli, S. Massaroli, J. Park & K. Tierney (2025), arXiv:2510.07073](https://arxiv.org/abs/2510.07073) — read Section 4 closely (about 3 pages: 4.1 with Algorithm 1 is the LNS and the two operator slots, 4.2 with Algorithm 2 is the genetic search and the fitness, eq. 1). Then read Section 5.2 with Figure 2 (ablations) and Figure 5 (crossover bias, elite size, code-length penalty). Read Table 1 slowly: all gaps are relative to SISRs, not to an optimum. Section 6 and Appendix C show what the discovered code actually looks like; Appendix A has the prompts if you want to reproduce it. Skim Sections 2 and 3. The version read is an anonymised review version; the code release is announced but not linked in it.

**Further reading:**
- [Slack Induction by String Removals for Vehicle Routing Problems — J. Christiaens & G. Vanden Berghe (2020), Transportation Science 54(2)](https://doi.org/10.1287/trsc.2019.0914) — SISRs, the hand-designed ruin-and-recreate heuristic that is both the reference baseline and the main source of ideas the discovered operators recombine.
- [Mathematical discoveries from program search with large language models — B. Romera-Paredes et al. (2024), Nature](https://doi.org/10.1038/s41586-023-06924-6) — FunSearch: the template for "LLM writes one function, an evolutionary loop scores and keeps the best".
- [Evolution of Heuristics: Towards Efficient Automatic Algorithm Design Using Large Language Model — F. Liu et al. (2024), ICML](https://arxiv.org/abs/2401.02051) and [ReEvo: Large Language Models as Hyper-Heuristics with Reflective Evolution — H. Ye et al. (2024), NeurIPS](https://arxiv.org/abs/2402.01145) — the LLM-based heuristic-design baselines VRPAgent beats; read them to see what "the LLM writes a whole constructive heuristic" looks like.
- [Neural Deconstruction Search for Vehicle Routing Problems — A. Hottung, P. Wong-Chung & K. Tierney (2025), TMLR](https://arxiv.org/abs/2501.03715) — the same group's GPU-based learned destroy operator; the strongest competitor in Table 1 and the source of the test instances.

**Why this matters** — There are two ways to let an LLM design an algorithm: ask it to write the whole solver, or keep an expert solver and let the LLM fill in a few narrowly specified slots. Earlier LLM work on routing took the first route and stayed well behind hand-tuned solvers. This paper takes the second route and, for the first time, reports LLM-generated heuristics that beat the state of the art on large CVRP, VRPTW and prize-collecting VRP instances, on a single CPU core. The transferable lesson is the division of labour: a fixed framework guarantees correctness, the LLM only proposes the problem-specific search behaviour, and an evolutionary loop with a carefully designed selection pressure does the rest.

**Core ideas**

*The problem.* In the capacitated VRP (CVRP), a fleet starts and ends at a depot and must visit every customer exactly once, with the load of each route bounded by the vehicle capacity; the objective is total travel distance. VRPTW adds a time window per customer; PCVRP lets you skip customers at the price of losing their prize. Instances here have 500 to 2000 customers, where exact methods are hopeless and the best solvers are metaheuristics.

*The fixed skeleton: ruin and recreate.* VRPAgent's solver is a large neighbourhood search (LNS) in the Shaw / SISRs tradition (Algorithm 1):

```
s ← one route per customer
repeat until time limit:
    R  ← f_REMOVE(s)            # LLM-written: which customers to take out
    s' ← s without R
    for c in f_ORDER(R):        # LLM-written: in which order to put them back
        insert c at its cheapest feasible position in s'     # fixed
    if cost(s') < cost(s) or SA-accept(s', s):  s ← s'     # fixed
```

Everything except the two marked functions is fixed C++ code, including all feasibility handling (capacity, time windows) and the greedy cheapest insertion. The prompts ask for fast, stochastic removal of a small number of customers that are each close to at least one other removed customer. The paper does not give the simulated-annealing schedule.

*Why only two slots.* Writing a complete state-of-the-art VRP solver is far beyond what generated code reliably achieves. Writing a removal or ordering function is short, cannot break feasibility (the framework repairs everything), and is exactly where expert heuristics differ from each other. The problem description is the only problem-specific part of the prompts, so the same machinery is reused for all three variants.

*Discovery as a genetic algorithm over programs (Algorithm 2).* An individual is a pair of C++ functions `(f_REMOVE, f_ORDER)`. Its fitness, eq. 1, is

```
F(h) = (1/|D_train|) Σ_{x ∈ D_train} cost(LNS_h(x))  +  λ · LOC(h)        (lower is better)
```

with 64 training instances of 500 customers, 20 s per instance, `LOC` = lines of code and `λ = 2·10⁻⁴` by default. The loop:

- *Initialisation:* 100 LLM-generated individuals.
- *Elitism:* the best `N_E = 10` survive unchanged.
- *Biased crossover (by the LLM):* each of 30 children combines a random elite with a random non-elite, instructed to take about 80% of its ideas from the elite and keep the code length comparable.
- *Mutation on elites only:* one of four prompts at random — remove a mechanism, add one, tune constants, or refactor for speed. The mutant replaces its parent only if its fitness improves.
- 40 generations; the final operator pair is chosen on a separate validation set.

The design is deliberately exploitative: new genetic material enters mainly through the non-elite 20% of each crossover, while mutation only ever improves elites.

*Ablations that justify the GA.* On CVRP (Figure 2, 10 runs averaged), replacing the biased crossover by an even 50/50 mix, dropping mutation (with more offspring to compensate), or swapping in ReEvo's reflection-based GA all make discovery worse; ReEvo's GA is worst. Figure 5: 80% bias and 10 elites are the best of the values tried; doubling `λ` roughly halves operator length with only a marginal loss in objective and cuts LLM token use by more than half.

*Which LLM.* Gemini 2.5 Flash for all main runs, at about $19 per discovery run. Open models were also tested: gpt-oss (120B) on two A100s gets nearly the same quality for under $2 per run; smaller open models improve the operators but less.

*Results and how to read them (Table 1).* Test instances come from the NDS paper; runtime per instance is 60 s, 120 s and 240 s for 500, 1000 and 2000 customers. Operators were discovered only on 500-customer instances and then applied unchanged to the larger sizes. Gaps relative to SISRs (negative = better than SISRs):

| Problem | N = 500 | N = 1000 | N = 2000 |
|---|---|---|---|
| CVRP | −0.12% | −0.19% | −0.11% |
| VRPTW | −0.24% | −0.33% | −0.33% |
| PCVRP | −0.09% | −0.21% | −0.32% |

For comparison, HGS on CVRP is at 0.00%, 0.84% and 2.33%; LKH3 and NVIDIA cuOpt are between about 1.7% and 5.7%; PyVRP-HGS on VRPTW and PCVRP is 1.9–4.8% behind. Earlier LLM-based methods (EoH, MCTS-AHD, ReEvo variants, NCO-LLM) range from roughly 0.8% to 27% on CVRP. The GPU-based NDS stays close and wins some settings, mostly at 500 customers. Two caveats: the margins over SISRs are tenths of a percent, and the text's "around −0.30%" is optimistic compared with the table; and the PCVRP table reports lower-is-better values although the problem is stated as prize maximisation, without clarification.

*What the discovered operators look like (Section 6, Appendix C).* Three co-authors with VRP expertise inspected them (explicitly not a formal study). The operators are ensembles of up to nine sub-heuristics selected at random, full of magic numbers, nesting and redundancy. Their ingredients are mostly known ideas — SISRs-style string removal around a seed customer, greedy sorting by distance or demand — recombined in ways the experts had not seen. Interpretability without a component-wise ablation is poor.

**A small example**

One LNS iteration on a toy CVRP: depot `0 = (0,0)`, customers `A = (2,0)`, `B = (4,0)`, `C = (0,2)`, `D = (0,4)`, capacity 2 customers per route, Euclidean distances. Start from a poor solution with crossing routes:

```
s = { 0–A–D–0 , 0–C–B–0 }     cost 10.472 + 10.472 = 20.944
```

The fixed recreate step inserts each removed customer at its cheapest feasible position. What the two LLM-written functions decide:

```
f_REMOVE = {A, C}  (one customer from each route, near the depot)
  partial: {0–D–0, 0–B–0}
  any order → A joins B (Δ = 0), C joins D (Δ = 0)
  s' = {0–A–B–0, 0–C–D–0}      cost 8 + 8 = 16.000   ← optimum

f_REMOVE = {A, B}  (the two nearest customers, both on the x-axis)
  partial: {0–D–0, 0–C–0}
  f_ORDER = (A, B): A → with D (Δ = 2.472), B must join C (Δ = 6.472)   cost 20.944, no gain
  f_ORDER = (B, A): B → with D (Δ = 5.657), A joins C (Δ = 2.828)       cost 20.485, small gain
```

Choosing which customers to remove decides whether the iteration can untangle the routes at all; with a fixed removal set, the reinsertion order still changes the outcome because greedy insertion is myopic and capacity fills up. This is exactly why VRPAgent lets the LLM design both functions, and why SISRs-style removal favours customers that are close but sit on *different* routes. Whatever the operators do, the result is always a feasible solution: a bad operator only wastes iterations.

**How it connects**

- **Same philosophy as GNN branching in MILP:** [Keep the solver, learn its decisions (X009, OPF literature repo)](https://github.com/mathematiger/opf_uncertainty_gnn_claude_literature/blob/main/tracks/12-hybrid-and-evaluation/2026-10-17-X009-gnn-branching-exact-milp-solver.md). There a network makes one internal decision of an exact solver; here an LLM writes one internal component of a metaheuristic. In both, correctness is guaranteed by the framework and a bad component only costs search time.
- **MuZero for power grid topology control:** the same split could be used to generate candidate-action or action-pruning heuristics for MCTS expansion in Grid2Op/L2RPN or PPtopogym, with the grid simulator as the fixed feasibility layer and an evolutionary loop scored on episode return. An evolved CPU-only heuristic would also be a natural expert baseline to compare MuZero's learned prior against.
- **GA design choices worth reusing:** elite-biased LLM crossover and accept-only-if-better mutation are concrete, ablated choices for any LLM program search.
- **Caveat for grid applications:** the evolved code is hard to interpret, which matters if operators must understand why an action was proposed.

**Check yourself**

1. Why can a badly written `f_REMOVE` or `f_ORDER` never produce an infeasible routing plan, and what does a bad operator actually cost?
2. In the toy example, removing the two nearest customers `{A, B}` barely helps, while `{A, C}` reaches the optimum. What property should a good removal operator have, and how does SISRs encode it?
3. The fitness includes a penalty on lines of code. Give two reasons, beyond readability, why this can help discovery.

<details><summary>Answers</summary>

1. Feasibility is handled entirely by the fixed framework: removed customers are reinserted only at feasible positions, and the acceptance step only ever compares feasible solutions. The LLM operators only decide *which* customers are moved and *in which order*. A bad operator therefore produces useless neighbourhoods — iterations that do not improve the solution — and the cost is solution quality within the time limit, not correctness.
2. The removed customers should be related enough that reinserting them can create a *different* assignment to routes — close in space, but drawn from several routes, so that greedy reinsertion has room to swap and untangle. Removing `{A, B}` leaves `C` and `D` in separate routes that both have to absorb a far-away customer; removing `{A, C}` frees one slot in each route next to the right partner. SISRs encodes this by removing short strings of consecutive customers from several routes around a random seed customer and its spatial neighbours, which induces slack exactly where reinsertion can exploit it.
3. (a) Shorter code means fewer tokens in every crossover and mutation prompt, which cuts cost (the paper reports more than 50% fewer tokens) and leaves the LLM a smaller program to reason about, so edits are more targeted. (b) It acts as a regulariser against bloat and overfitting to the 64 training instances: constructions that add many special cases and magic numbers for a marginal training gain are penalised, which helps transfer to the unseen 1000- and 2000-customer instances.

</details>

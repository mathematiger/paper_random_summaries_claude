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

# VRPAgent: LLM-Driven Discovery of Heuristic Operators for Vehicle Routing Problems

**Link:** https://arxiv.org/abs/2510.07073

## TL;DR

An LLM writes only the two problem-specific operators (customer removal and reinsertion ordering) of a fixed large neighborhood search, and a genetic algorithm with LLM-performed crossover and mutation evolves these operators. The resulting single-CPU solver beats strong handcrafted (SISRs, HGS, LKH3) and neural baselines on CVRP, VRPTW and prize-collecting VRP with 500–2000 customers, which the authors present as the first time LLM-generated heuristics push the state of the art on these problems.

## Problem

Earlier LLM-based heuristic design (EoH, ReEvo, MCTS-AHD) asks the LLM to write whole constructive heuristics or plug-ins to weak frameworks, and the results stay far behind expert solvers. Letting the LLM write an entire high-performing metaheuristic is too hard and too error-prone. The question is how to constrain the LLM's job so that generated code is reliably correct but can still discover genuinely better search behaviour.

## Method

- **Fixed skeleton:** ruin-and-recreate LNS in C++. Each iteration: remove a set of customers (`f_REMOVE`, LLM-written), order them (`f_ORDER`, LLM-written), greedily reinsert at the best position (fixed), then accept via improvement or simulated annealing (fixed). Feasibility handling is part of the fixed code, so the LLM cannot break correctness.
- **Discovery GA:** individuals are (remove, order) code pairs. Population 100, 10 elites, 30 offspring per generation, 40 generations.
  - *Biased crossover by the LLM:* each child combines a random elite and a random non-elite, taking roughly 80% of the ideas from the elite.
  - *Mutation on elites only:* one of four prompts (remove a mechanic, add a mechanic, tune constants, refactor for speed); the mutant replaces its parent only if it is better.
- **Fitness:** mean objective on 64 training instances (500 customers, 20 s each) plus a small penalty proportional to lines of code. Final operator chosen on a validation set.
- **LLM:** Gemini 2.5 Flash for the main runs (about $20 per discovery run); open models (e.g. gpt-oss 120B on two A100s) get close at under $2.

## Results

- Gaps relative to SISRs (negative = better), at 500 / 1000 / 2000 customers:
  - CVRP: −0.12% / −0.19% / −0.11%
  - VRPTW: −0.24% / −0.33% / −0.33%
  - PCVRP: −0.09% / −0.21% / −0.32%
- Operators discovered on 500-customer instances transfer to 1000 and 2000 customers without re-discovery.
- Clearly ahead of HGS / PyVRP-HGS, LKH3 and NVIDIA cuOpt at large sizes; far ahead of other LLM-based methods (which sit at roughly 1–27% gap on CVRP).
- The GPU-based neural solver NDS remains competitive and wins some 500-customer settings.
- **Ablations:** both biased crossover and elite mutation matter; swapping in ReEvo's reflection GA is worst. Doubling the code-length penalty halves operator length with only a marginal loss in quality and cuts LLM token use substantially.

## Limitations

- The discovered operators are messy ensembles (up to nine sub-heuristics picked at random, magic numbers, deep nesting), hard to interpret without further ablation.
- Ingredients are mostly recombinations of known ideas (SISRs-style removal, greedy criteria); novelty lies in the combination.
- Only 10 discovery runs per problem and one main LLM; the expert assessment of the code is informal by the authors' own account.
- Margins over SISRs are small (tenths of a percent), and the LNS skeleton itself carries much of the performance — the paper does not isolate how much the framework vs. the discovered operators contributes beyond the SISRs reimplementation comparison.
- Minor reporting inconsistencies (PCVRP objective direction, text vs. table on gap magnitudes).

## Relevance to my research

- **Design pattern:** "LLM writes a narrow, pluggable operator; a fixed framework guarantees feasibility" transfers directly to power grid topology control — e.g. LLM-evolved action-pruning or candidate-generation heuristics feeding MuZero's MCTS expansion, with the grid simulator acting as the fixed correctness layer.
- **Program search as baseline/prior:** cheap, CPU-only evolved heuristics could serve as an expert-agent baseline for L2RPN/PPtopogym or as a prior policy to compare against learned priors.
- **GA insights:** elite-biased LLM crossover and accept-only-if-better mutation are concrete, ablated choices worth reusing if heuristic discovery is ever tried for grid actions.
- **Caution:** interpretability of evolved code is poor — relevant if operator-facing explainability matters for grid applications.

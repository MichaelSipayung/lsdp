# Paper 2 / dissertation chapter — preference-conditioned learning against exact fronts

Independent of src/, suite/ and paper/ (Paper 1). Reuses src/lsdp.cpp read-only via #include.

## Research questions

RQ1  How far are preference-conditioned neural solvers (Pareto set learning) from the TRUE
     Pareto front on a realistic multiobjective lot-sizing problem?
RQ2  Does an action space derived from the order-event theorem (Theorem A.10 of Paper 1)
     improve learning compared with an unstructured quantity action space?
RQ3  How well do policies trained on synthetic demand transfer to real demand (M5, UCI)
     across demand classes?

## Novelty claims (to be verified)

1. First preference-conditioned solver for multiobjective lot sizing with safety instruments.
2. Theory-guided decoding: actions are next-order periods with the covering level of
   Theorem A.10, so every generated plan satisfies a necessary condition for Pareto
   optimality by construction.
3. Evaluation against exact fronts (thousands of instances), with exact per-preference
   regret: for every weight vector the optimum of the scalarised problem is the minimum
   over the exact front, so the gap of the learned solution is exact, not relative to a
   heuristic.

## MDP (order-event form)

* First action: choose a requirement class (pointer over the non-dominated classes of
  Corollary A.8); this fixes (SS, SOT) and CSL.
* State at an order event: (tau, x', k, W, R); first decision from a virtual start chooses
  the first order period among feasible order-free prefixes.
* Action: next order period tau' in {tau+1, ..., T'+1} (and the exceptional levels of
  Theorem A.10); the level X is determined by the theorem.
* Episode ends at tau' = T'+1; outcome F = (POC, HC, -CSL, -ITO) evaluated exactly.

## Scalarisation and metrics

* Objectives normalised by instance-intrinsic scales (no optimisation needed at inference).
* Weighted Tchebycheff g(F; lambda) with reference point 0 of the normalised objectives.
* Per-preference regret: g(F_policy; lambda) - min_{y in exact front} g(y; lambda).
  Exact because a Tchebycheff optimum is attained on the Pareto front.
* Front-level: share of outputs that lie on the exact front, IGD+, coverage.

## Hypotheses and predictions (before training)

H1  The order-event action space gives lower regret than a quantity action space at equal
    model size and training budget. Prediction: clearly lower, because the quantity space
    contains dominated actions in every state.
H2  In distribution (synthetic), median per-preference regret below 1% of the normalised
    range after training; a myopic greedy baseline is far worse (prediction 5-15%).
H3  Inference takes milliseconds per (instance, preference), 2-4 orders of magnitude below
    computing the exact front.
H4  Transfer to real demand degrades most for lumpy and intermittent series.

## Stages

1. Environment, exact arc features, evaluator against exact fronts, greedy baseline. (now)
2. Policy network (pointer MLP over candidate features, preference-conditioned), REINFORCE
   with shared baseline over several rollouts, pure C++. Verified by gradient check.
3. Training on synthetic instances (laptop, hours), evaluation on held-out synthetic, E2,
   E6 (M5, UCI), ablation quantity vs order-event actions.
4. Paper 2.

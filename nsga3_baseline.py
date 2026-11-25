"""nsga3_baseline.py — NSGA-III baseline (pymoo), mirroring the setup of
Kania et al. (CIE 2022 / JIM 2024): NSGA-III, structured Das–Dennis reference
points, integer SBX crossover + polynomial mutation with rounding repair.
Solves exactly the same instance as lsdp.cpp (same demand file, same encoding).

usage:
  python nsga3_baseline.py --sku 1 --demand hasil/sku1_s42.csv.demand.csv \
         --seed 1 --time 01:00:00 --partitions 12 --out hasil/nsga3_sku1_s42_r1.csv
"""
import argparse
import csv
import time

import numpy as np
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.callback import Callback
from pymoo.core.problem import Problem
from pymoo.core.repair import Repair
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.ref_dirs import get_reference_directions

from lsmodel import DSOT, Instance, repair


class LotSizing(Problem):
    def __init__(self, inst):
        self.inst = inst
        n_var = inst.Tp + 2
        xl = np.zeros(n_var)
        xu = np.array([inst.a_max] * inst.Tp + [inst.ss_max, inst.ksot_max], dtype=float)
        n_ieq = 1 + (inst.T - inst.L) + (inst.T - inst.L)
        super().__init__(n_var=n_var, n_obj=4, n_ieq_constr=n_ieq, xl=xl, xu=xu, vtype=int)

    def _evaluate(self, x, out, *args, **kwargs):
        F, G, _ = self.inst.evaluate(np.round(x))
        out["F"] = F
        out["G"] = G


class StructuralRepair(Repair):
    """Problem-specific repair (feasibility + Lemma D + no overshoot), see lsmodel.repair."""

    def _do(self, problem, X, **kwargs):
        return repair(problem.inst, np.round(X))


class Archive(Callback):
    """Keeps every feasible non-dominated point seen during the run (fair to NSGA-III)."""

    def __init__(self, every=10):
        super().__init__()
        self.every = every
        self.X = None
        self.F = None
        self.buf_X, self.buf_F = [], []
        self.first_feasible_gen = None
        self.t0 = time.time()

    def _flush(self):
        if not self.buf_X:
            return
        X = np.vstack(self.buf_X + ([self.X] if self.X is not None else []))
        F = np.vstack(self.buf_F + ([self.F] if self.F is not None else []))
        F_r = np.round(F, 9)
        _, idx = np.unique(np.column_stack([F_r, X]), axis=0, return_index=True)
        X, F = X[idx], F[idx]
        nd = NonDominatedSorting().do(F, only_non_dominated_front=True)
        self.X, self.F = X[nd], F[nd]
        self.buf_X, self.buf_F = [], []

    def notify(self, algorithm):
        pop = algorithm.pop
        G = pop.get("G")
        feas = np.all(G <= 0, axis=1)
        if feas.any():
            if self.first_feasible_gen is None:
                self.first_feasible_gen = algorithm.n_gen
            self.buf_X.append(np.round(pop.get("X")[feas]).astype(int))
            self.buf_F.append(pop.get("F")[feas])
        if algorithm.n_gen % self.every == 0:
            self._flush()
            if algorithm.n_gen % (self.every * 10) == 0:
                n = 0 if self.X is None else len(self.X)
                print(f"gen {algorithm.n_gen:6d}  evals {algorithm.evaluator.n_eval:9d}  "
                      f"archive {n:6d}  elapsed {time.time() - self.t0:8.1f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sku", type=int, required=True)
    ap.add_argument("--demand", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--time", default="01:00:00", help="wall-clock limit HH:MM:SS")
    ap.add_argument("--partitions", type=int, default=12)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repair", type=int, default=0, help="1 = problem-specific repair operator")
    a = ap.parse_args()

    inst = Instance(a.sku, a.demand)
    problem = LotSizing(inst)
    ref_dirs = get_reference_directions("das-dennis", 4, n_partitions=a.partitions)
    algo = NSGA3(
        ref_dirs=ref_dirs,
        pop_size=len(ref_dirs),
        sampling=IntegerRandomSampling(),
        crossover=SBX(prob=0.9, eta=15, vtype=float, repair=RoundingRepair()),
        mutation=PM(eta=20, vtype=float, repair=RoundingRepair()),
        eliminate_duplicates=True,
        repair=StructuralRepair() if a.repair else None,
    )
    cb = Archive()
    t0 = time.time()
    res = minimize(problem, algo, get_termination("time", a.time), seed=a.seed,
                   callback=cb, verbose=False)
    cb._flush()
    wall = time.time() - t0

    n = 0 if cb.X is None else len(cb.X)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["SS", "SOT", "CSL", "POC", "HC", "ITO", "orders"] +
                   [f"Q{t}" for t in range(1, inst.T + 1)])
        if n:
            _, _, d = inst.evaluate(cb.X)
            for i in range(n):
                x = cb.X[i]
                Q = [int(v) * inst.r for v in x[:inst.Tp]] + [0] * inst.L
                w.writerow([int(x[inst.Tp]), round(x[inst.Tp + 1] * DSOT, 4),
                            f"{d['CSL'][i]:.10f}", f"{d['POC'][i]:.4f}", f"{d['HC'][i]:.6f}",
                            f"{d['ITO'][i]:.8f}", int(d['orders'][i])] + Q)
    print(f"SKU{a.sku} seed={a.seed} repair={a.repair} partitions={a.partitions} pop={len(ref_dirs)} "
          f"gens={res.algorithm.n_gen} evals={res.algorithm.evaluator.n_eval} "
          f"first_feasible_gen={cb.first_feasible_gen} archive={n} wall={wall:.1f}s")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

// psl_core.hpp - order-event environment, exact reference fronts and evaluation for
// preference-conditioned learning on the lot-sizing model of Paper 1.
// Reuses src/lsdp.cpp read-only (compile with -I<dir of lsdp.cpp>).
#pragma once
#define main lsdp_main_unused
#include "lsdp.cpp"
#undef main

// ------------------------------------------------------------------ instance preparation
struct ClassRep {
  vector<double> kappa;  // rounded requirement vector, index 1..T'
  double SS, SOT, B;     // representative outer point and service index
};
struct FrontPt { double POC, HC, CSL, ITO, B; };

struct Prepared {
  Instance in;
  int Tp = 0;
  vector<double> b, num;            // b_tau, D(tau+L)+sigma, tau = 1..T'
  double C0 = 0.0, ito_early = 0.0;
  vector<ClassRep> classes;         // non-dominated classes, decreasing B
  vector<FrontPt> front;            // exact Pareto front (empty if not computed)
  double ideal[4], nadir[4];        // of the normalised minimisation vector (POC, HC, -CSL, -ITO)
};

// classes of the continuous outer domain after exact memoization and class dominance
inline vector<ClassRep> build_classes(const Instance& in) {
  auto b1 = requirement(in, in.mu * in.MS, 0.0), b2 = requirement(in, 0.0, in.MS);
  double bmax = max(*max_element(b1.begin() + 1, b1.end()), *max_element(b2.begin() + 1, b2.end()));
  auto lat = reachable_values(in, max(0L, (long)ceil(bmax)) + 2 * max(in.moq, in.r));
  auto ceil_reach = [&](double v) -> long {
    long need = (long)ceil(v - 1e-7);
    if (need <= 0) return 0;
    return *lower_bound(lat.begin(), lat.end(), need);
  };
  map<vector<long>, ClassRep> g;
  for (const auto& oc : cell_candidates(in, lat)) {
    auto beta = requirement(in, oc.SS, oc.SOT);
    vector<long> key(beta.size());
    for (size_t i = 1; i < beta.size(); ++i) key[i] = ceil_reach(beta[i]);
    auto it = g.find(key);
    if (it == g.end() || oc.Bsup > it->second.B + 1e-12)
      g[key] = {vector<double>(key.begin(), key.end()), oc.SS, oc.SOT, oc.Bsup};
  }
  vector<ClassRep> all;
  for (auto& [k, c] : g) all.push_back(c);
  sort(all.begin(), all.end(), [](const ClassRep& a, const ClassRep& b) {
    if (fabs(a.B - b.B) > 1e-12) return a.B > b.B;
    return accumulate(a.kappa.begin(), a.kappa.end(), 0.0) < accumulate(b.kappa.begin(), b.kappa.end(), 0.0);
  });
  vector<ClassRep> kept;
  for (auto& c : all) {
    bool dom = false;
    for (auto& h : kept) {
      bool le = true;
      for (size_t i = 1; i < c.kappa.size() && le; ++i) le = h.kappa[i] <= c.kappa[i];
      if (le) { dom = true; break; }
    }
    if (!dom) kept.push_back(c);
  }
  return kept;
}

inline Prepared prepare(const Instance& in, bool exact) {
  Prepared P;
  P.in = in;
  P.Tp = in.T - in.L;
  P.b.assign(P.Tp + 2, 0.0);
  P.num.assign(P.Tp + 2, 0.0);
  for (int tau = 1; tau <= P.Tp; ++tau) {
    int t = tau + in.L;
    P.b[tau] = in.I0 - 0.5 * (in.cumD[t - 1] + in.cumD[t]);
    P.num[tau] = in.D[t] + in.sigma;
  }
  for (int t = 1; t <= in.T; ++t) P.C0 += in.I0 - 0.5 * (in.cumD[t - 1] + in.cumD[t]);
  for (int t = 1; t <= in.L; ++t)
    P.ito_early += (in.D[t] + in.sigma) / (in.I0 - 0.5 * (in.cumD[t - 1] + in.cumD[t]));
  // build_classes -> cell_candidates passes the arrangement through the global g_lines of lsdp.cpp
  // (written in arrangement_vertices, read right after in cell_candidates). lsdp.cpp is left
  // unchanged, so concurrent preparations of different instances are serialised here.
#pragma omp critical(psl_build_classes)
  P.classes = build_classes(in);
  if (exact) {
    vector<vector<FrontPt>> res(P.classes.size());
#pragma omp parallel for schedule(dynamic, 1)
    for (long g = 0; g < (long)P.classes.size(); ++g) {
      auto pts = solve_inner_cover(in, P.classes[g].kappa, 0);
      double csl = Phi(P.classes[g].B / in.sigma);
      for (auto& q : pts) res[g].push_back({q.POC, q.HC, csl, q.ITO, P.classes[g].B});
    }
    // 4-D filter, classes are in decreasing B (service compared through B, as in Paper 1)
    vector<FrontPt> arch;
    size_t g = 0;
    while (g < res.size()) {
      size_t h = g;
      vector<FrontPt> batch;
      while (h < res.size() && fabs(P.classes[h].B - P.classes[g].B) <= 1e-9) {
        for (auto& q : res[h]) batch.push_back(q);
        ++h;
      }
      vector<FrontPt> nd;
      sort(batch.begin(), batch.end(), [](const FrontPt& a, const FrontPt& b) {
        if (a.POC != b.POC) return a.POC < b.POC;
        if (a.HC != b.HC) return a.HC < b.HC;
        return a.ITO > b.ITO;
      });
      auto dom3 = [](const FrontPt& a, const FrontPt& q) {
        return a.POC <= q.POC + 1e-7 && a.HC <= q.HC + 1e-7 && a.ITO >= q.ITO - 1e-9;
      };
      for (auto& q : batch) {
        bool d = false;
        for (auto& a : arch) if (dom3(a, q)) { d = true; break; }
        if (!d) for (auto& a : nd) if (dom3(a, q)) { d = true; break; }
        if (!d) nd.push_back(q);
      }
      for (auto& q : nd) arch.push_back(q);
      g = h;
    }
    P.front = arch;
    for (int i = 0; i < 4; ++i) { P.ideal[i] = 1e300; P.nadir[i] = -1e300; }
    for (auto& q : P.front) {
      double f[4] = {q.POC, q.HC, -q.CSL, -q.ITO};
      for (int i = 0; i < 4; ++i) { P.ideal[i] = min(P.ideal[i], f[i]); P.nadir[i] = max(P.nadir[i], f[i]); }
    }
    for (int i = 0; i < 4; ++i) if (P.nadir[i] - P.ideal[i] < 1e-12) P.nadir[i] = P.ideal[i] + 1e-12;
  }
  return P;
}

// ------------------------------------------------------------------ order-event environment
struct Cand {
  int tn;          // next order period (T'+1 = no further order)
  long X;          // level reached by the order at the current period
  double dPOC, dW, dITO;   // increments: POC, sum of X(s) (holding), turnover
};
struct State {
  int cls = -1;    // chosen class
  int tau = 0;     // current order period (0 = virtual start, before the first order)
  long x = 0;      // pre-order cumulative quantity
  int k = 0;
  double W = 0, R = 0;   // sum of X(s) and partial turnover
  bool done = false;
  long XT = 0;
};

// candidates of the current state (class already chosen)
inline vector<Cand> candidates(const Prepared& P, const State& s) {
  const Instance& in = P.in;
  const int Tp = P.Tp;
  const auto& beta = P.classes[s.cls].kappa;
  vector<Cand> out;
  if (s.tau == 0) {  // order-free prefix: X = 0 on 1..tau1-1
    double R = 0.0;
    for (int t1 = 1; t1 <= Tp + 1; ++t1) {
      if (t1 >= 2) {
        int u = t1 - 1;
        if (beta[u] > 1e-9 || P.b[u] < kEps) break;
        R += P.num[u] / P.b[u];
      }
      out.push_back({t1, 0, 0.0, 0.0, R});
    }
    return out;
  }
  const long r = in.r, moq = in.moq;
  auto ceil_r = [&](double y) -> long { return y <= 1e-9 ? 0 : r * (long)ceil((y - 1e-9) / (double)r); };
  const int tau = s.tau;
  const long xp = s.x;
  double needB = -1e300, needE = -1e300;
  const double needT = 2.0 * (kEps - P.b[tau]) - (double)xp;
  for (int tn = tau + 1; tn <= Tp + 1; ++tn) {
    const int last = tn - 1;
    needB = max(needB, beta[last]);
    if (last >= tau + 1) needE = max(needE, kEps - P.b[last]);
    long X = max(ceil_r(max(needB, max(needE, needT))), xp + moq);
    while (true) {
      const double Ib0 = P.b[tau] + 0.5 * (double)(xp + X);
      if (Ib0 >= kEps - 1e-12) {
        double dR = P.num[tau] / Ib0;
        for (int u = tau + 1; u <= last; ++u) dR += P.num[u] / (P.b[u] + (double)X);
        Cand c{tn, X, in.p * (double)(X - xp) + in.c, (double)X * (tn - tau), dR};
        out.push_back(c);
      }
      if (tn > Tp || P.b[tn] + (double)(X + r) >= kEps) break;
      X += r;
    }
  }
  return out;
}

inline void step(const Prepared& P, State& s, const Cand& c) {
  if (s.tau == 0) {      // first order period chosen
    s.R += c.dITO;
    if (c.tn > P.Tp) { s.done = true; s.XT = 0; return; }
    s.tau = c.tn;
    s.x = 0;
    return;
  }
  s.k += 1;
  s.W += c.dW;
  s.R += c.dITO;
  if (c.tn > P.Tp) { s.done = true; s.XT = c.X; return; }
  s.tau = c.tn;
  s.x = c.X;
}

// final plan objectives (POC, HC, CSL, ITO)
inline array<double, 4> outcome(const Prepared& P, const State& s) {
  const Instance& in = P.in;
  double XT = (double)s.XT;
  return {in.p * XT + in.c * s.k, in.h * (P.C0 + s.W - 0.5 * XT), Phi(P.classes[s.cls].B / in.sigma),
          P.ito_early + s.R};
}

// normalised minimisation vector and weighted Tchebycheff value
inline array<double, 4> normalise(const Prepared& P, const array<double, 4>& F) {
  double f[4] = {F[0], F[1], -F[2], -F[3]};
  array<double, 4> z;
  // a (nearly) degenerate objective on the front (e.g. every Pareto point has CSL = 1, or a series
  // whose demand is covered by the initial stock, so that POC = 0 on the whole front) is scaled by
  // 0.1% of the objective's natural scale for the instance instead of its zero range
  const Instance& in = P.in;
  const double SX = max(1.0, in.mu * in.T);
  const double nat[4] = {in.p * SX + in.c * in.T, in.h * SX * in.T, 1.0, in.T * (in.mu + in.sigma) / max(in.mu, 1.0)};
  for (int i = 0; i < 4; ++i) {
    double range = max(P.nadir[i] - P.ideal[i], 1e-3 * nat[i]);
    z[i] = (f[i] - P.ideal[i]) / range;
  }
  return z;
}
inline double tcheby(const array<double, 4>& z, const array<double, 4>& lam) {
  double g = -1e300;
  for (int i = 0; i < 4; ++i) g = max(g, lam[i] * (z[i] + 1e-6));
  return g;
}
inline double exact_opt(const Prepared& P, const array<double, 4>& lam) {
  double best = 1e300;
  for (auto& q : P.front) best = min(best, tcheby(normalise(P, {q.POC, q.HC, q.CSL, q.ITO}), lam));
  return best;
}
inline bool on_front(const Prepared& P, const array<double, 4>& F) {
  for (auto& q : P.front)
    if (fabs(q.POC - F[0]) <= 2e-4 && fabs(q.HC - F[1]) <= 2e-6 && fabs(q.CSL - F[2]) <= 1e-9 &&
        fabs(q.ITO - F[3]) <= 2e-8) return true;
  return false;
}

// simplex lattice of weight vectors in 4 dimensions (H partitions)
inline vector<array<double, 4>> simplex_weights(int H) {
  vector<array<double, 4>> W;
  for (int a = 0; a <= H; ++a) for (int b = 0; a + b <= H; ++b) for (int c = 0; a + b + c <= H; ++c) {
    int d = H - a - b - c;
    W.push_back({a / (double)H, b / (double)H, c / (double)H, d / (double)H});
  }
  return W;
}

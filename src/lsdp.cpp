// lsdp.cpp — Exact multiobjective dynamic programming for the single-item
// lot sizing + safety strategy model of Kania et al. (CIE 2022).
//
// Objectives (as in the paper):
//   POC = p * sum Q(t) + c * #orders                       (min)
//   HC  = h * sum_t (I(t-1)+I(t))/2                          (min)
//   CSL = Phi((SS + mu*SOT)/sigma)                            (max)
//   ITO = sum_t (D(t)+sigma) / ((I(t-1)+I(t))/2)              (max)
// Constraints:
//   I(t) = I(t-1) + Q(t-L) - D(t)
//   I(t) >= SS + SOT*D(t)  for t = L+1..T (earlier periods are not influenced by decisions)
//   Q(t) in {0} U {moq + a*r : a = 0,1,2,...}
//   SOT + SS/mu <= MS,  SS >= 0, SOT >= 0
//
// Method: cumulative reformulation X(tau) = sum_{s<=tau} Q(s), tau = 1..T-L.
//   - HC and ITO are additive over consecutive pairs (X(tau-1), X(tau)).
//   - For fixed (SS,SOT) the inner problem is a 3-criteria shortest path on a
//     layered graph (states = reachable cumulative quantities); solved exactly by
//     Pareto label-setting DP.
//   - Inner objectives depend on (SS,SOT) only through the rounded requirement
//     vector -> exact memoization; for each distinct vector only the (SS,SOT)
//     with maximal SS+mu*SOT (i.e. maximal CSL) can be Pareto optimal.
//   - Lemma (upper bound): every Pareto-optimal plan has X(T') < beta_max + max(moq,r).
//
// Modes:
//   ./lsdp verify  [--seeds N]          brute-force check of exactness on small instances
//   ./lsdp front   [options]            compute exact Pareto front, write CSV
// Build: g++ -O3 -march=native -std=c++20 -fopenmp lsdp.cpp -o lsdp
//
// Author: generated with Claude for M. Sipayung's research; MIT licence.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

// ---------------------------------------------------------------- instance
struct Instance {
  int T = 48, L = 5;
  double p = 134.0, c = 200.0, h = 134.0 * 0.1 / 52.0;
  long moq = 70, r = 70;
  double mu = 163.0, sigma = 41.0, MS = 1.0;
  long I0 = 0;
  vector<long> D;     // D[1..T]
  vector<long> cumD;  // cumD[0..T]
};

static double Phi(double z) { return 0.5 * erfc(-z / sqrt(2.0)); }

// optional overrides of the preset parameters (negative = keep preset)
struct Overrides {
  double mu = -1, cv = -1, c = -1, hfrac = -1, MS = -1;
  long r = -1;  // sets moq = r
  int L = -1;
};

Instance make_instance(int sku, int T, unsigned seed, const Overrides& ov = Overrides()) {
  Instance in;
  in.T = T;
  if (sku == 1) {  // reconstructed from desdeo-dash/data/first_data.csv
    in.p = 134.0; in.c = 200.0; in.h = in.p * 0.1 / 52.0; in.L = 5;
    in.moq = 70; in.r = 70; in.mu = 163.0; in.sigma = 41.0; in.MS = 1.0;
  } else {         // reconstructed from desdeo-dash/data/second_data.csv
    in.p = 483.85; in.c = 200.0; in.h = in.p * 0.1 / 52.0; in.L = 4;
    in.moq = 3; in.r = 3; in.mu = 12.2; in.sigma = 3.6; in.MS = 1.4;
  }
  double cv = in.sigma / in.mu;
  if (ov.mu > 0) in.mu = ov.mu;
  if (ov.cv > 0) cv = ov.cv;
  in.sigma = cv * in.mu;
  if (ov.c >= 0) in.c = ov.c;
  if (ov.hfrac > 0) in.h = in.p * ov.hfrac / 52.0;
  if (ov.MS > 0) in.MS = ov.MS;
  if (ov.r > 0) { in.r = ov.r; in.moq = ov.r; }
  if (ov.L >= 0) in.L = ov.L;
  mt19937_64 g(seed);
  normal_distribution<double> nd(in.mu, in.sigma);
  in.D.assign(T + 1, 0);
  in.cumD.assign(T + 1, 0);
  for (int t = 1; t <= T; ++t) {
    in.D[t] = max(0L, lround(nd(g)));
    in.cumD[t] = in.cumD[t - 1] + in.D[t];
  }
  // opening stock covers the uninfluenceable periods 1..L plus one period
  in.I0 = in.cumD[in.L] + lround(in.mu);
  return in;
}

// ---------------------------------------------------------------- inner DP
struct Label {
  int k;         // number of orders
  double sx;     // sum of X(tau)
  double ito;    // sum of ITO terms (periods L+1..T)
  int prev_state, prev_label;  // back pointer
};

struct InnerPoint {
  double POC, HC, ITO;
  int k;
  vector<long> Q;  // Q[1..T]
};

struct Ctx {
  const Instance* in;
  vector<long> states;          // reachable cumulative values, ascending
  double C0_hc;                 // constant part of sum of average inventories
  double ito_early;             // constant ITO contribution of periods 1..L
  bool valid_order(long q) const {
    return q == 0 || (q >= in->moq && (q - in->moq) % in->r == 0);
  }
};

static inline bool dominates3(const Label& a, const Label& b) {
  // minimise k, sx ; maximise ito
  return a.k <= b.k && a.sx <= b.sx + 1e-9 && a.ito >= b.ito - 1e-12 &&
         (a.k < b.k || a.sx < b.sx - 1e-9 || a.ito > b.ito + 1e-12);
}
static inline bool equal3(const Label& a, const Label& b) {
  return a.k == b.k && fabs(a.sx - b.sx) <= 1e-9 && fabs(a.ito - b.ito) <= 1e-12;
}

static void insert_label(vector<Label>& v, const Label& L) {
  for (const auto& e : v)
    if (dominates3(e, L) || equal3(e, L)) return;
  size_t w = 0;
  for (size_t i = 0; i < v.size(); ++i)
    if (!dominates3(L, v[i])) v[w++] = v[i];
  v.resize(w);
  v.push_back(L);
}

// reachable cumulative quantities up to Xmax: 0 or sums of valid orders
vector<long> reachable_values(const Instance& in, long Xmax) {
  vector<char> ok(Xmax + 1, 0);
  ok[0] = 1;
  for (long x = 1; x <= Xmax; ++x) {
    // x reachable iff x - q reachable for some valid q>0
    for (long q = in.moq; q <= x; q += in.r)
      if (ok[x - q]) { ok[x] = 1; break; }
  }
  vector<long> s;
  for (long x = 0; x <= Xmax; ++x) if (ok[x]) s.push_back(x);
  return s;
}

// beta[tau] = requirement on X(tau), tau = 1..T'
vector<double> requirement(const Instance& in, double SS, double SOT) {
  int Tp = in.T - in.L;
  vector<double> beta(Tp + 1, 0.0);
  for (int tau = 1; tau <= Tp; ++tau) {
    int t = tau + in.L;
    beta[tau] = SS + SOT * in.D[t] + in.cumD[t] - (double)in.I0;
  }
  return beta;
}

// Fast label store for one state: the 3-D front (k, sx, ito) is decomposed by the
// integer k (number of orders, 0..T') into 2-D staircases. In each staircase labels
// are sorted by sx (ascending) and, being mutually non-dominated, have strictly
// increasing ito. A label (k, sx, ito) is dominated iff for some k' <= k the last
// entry with sx' <= sx has ito' >= ito  -> O(k log n) instead of O(n).
bool g_fast = true;  // false: reference implementation (linear scans), for verification
bool g_lemmaD = true;  // restrict ordering transitions by Lemma D (order only when needed)
constexpr double kEps = 1.0;  // positivity threshold: Ibar(t) >= kEps (domain of ITO)
bool g_probes = false;  // continuous mode: also add sector probes (not needed, see Theorem 3)
bool g_noito = false;  // drop ITO (bi-objective POC/HC sub-model, used for the MILP comparison)
double g_fixSS = -1, g_fixSOT = -1;  // evaluate a single outer point (SS, SOT) if set
using KStairs = vector<vector<Label>>;
static void stair_insert(KStairs& ks, const Label& L) {
  const int K = (int)ks.size();
  for (int kk = 0; kk <= L.k && kk < K; ++kk) {
    const auto& v = ks[kk];
    if (v.empty()) continue;
    auto it = upper_bound(v.begin(), v.end(), L.sx + 1e-9,
                          [](double val, const Label& e) { return val < e.sx; });
    if (it != v.begin() && prev(it)->ito >= L.ito - 1e-12) return;  // dominated or equal
  }
  for (int kk = L.k; kk < K; ++kk) {
    auto& v = ks[kk];
    if (v.empty()) continue;
    auto first = lower_bound(v.begin(), v.end(), L.sx - 1e-9,
                             [](const Label& e, double val) { return e.sx < val; });
    auto last = first;
    while (last != v.end() && last->ito <= L.ito + 1e-12) ++last;
    v.erase(first, last);
  }
  auto& v = ks[L.k];
  auto pos = lower_bound(v.begin(), v.end(), L.sx,
                         [](const Label& e, double val) { return e.sx < val; });
  v.insert(pos, L);
}

// Solve inner 3-criteria problem exactly. Returns non-dominated points in
// (POC, HC, ITO) for the given requirement vector.
vector<InnerPoint> solve_inner(const Instance& in, const vector<double>& beta,
                               long Xmax, size_t* label_count = nullptr) {
  const int Tp = in.T - in.L;
  Ctx cx;
  cx.in = &in;
  cx.states = reachable_values(in, Xmax);
  const int S = (int)cx.states.size();
  // constants
  double C0 = 0.0;  // sum_t [I0 - (cumD(t-1)+cumD(t))/2]
  for (int t = 1; t <= in.T; ++t) C0 += in.I0 - 0.5 * (in.cumD[t - 1] + in.cumD[t]);
  double ito_early = 0.0;
  for (int t = 1; t <= in.L; ++t) {
    double Ib = in.I0 - 0.5 * (in.cumD[t - 1] + in.cumD[t]);
    ito_early += (in.D[t] + in.sigma) / Ib;
  }
  // stage labels: lab[tau][state] -> vector<Label>
  vector<vector<vector<Label>>> lab(Tp + 1, vector<vector<Label>>(S));
  lab[0][0].push_back({0, 0.0, 0.0, -1, -1});  // X(0) = 0 (state index 0 is value 0)
  for (int tau = 1; tau <= Tp; ++tau) {
    const int t = tau + in.L;
    const double base = in.I0 - 0.5 * (in.cumD[t - 1] + in.cumD[t]);
    const double base_next =
        (tau < Tp) ? in.I0 - 0.5 * (in.cumD[t] + in.cumD[t + 1]) : 0.0;
    const double num = in.D[t] + in.sigma;
    int jmin = int(lower_bound(cx.states.begin(), cx.states.end(),
                               (long)ceil(beta[tau] - 1e-9)) - cx.states.begin());
    vector<KStairs> st;
    if (g_fast) st.assign(S, KStairs(Tp + 1));
    for (int i = 0; i < S; ++i) {
      const auto& src = lab[tau - 1][i];
      if (src.empty()) continue;
      const long xp = cx.states[i];
      for (int j = max(i, jmin); j < S; ++j) {
        const long x = cx.states[j];
        const long q = x - xp;
        if (!cx.valid_order(q)) continue;
        // Lemma D (order only when needed). Postponing the order to tau+1 (or dropping it
        // if tau = T') lowers the average stock in periods tau+L and tau+L+1 only. It is a
        // strictly dominating move iff (a) the requirement at tau holds without the order,
        // (b) Ibar(tau+L) >= eps without it and (c) Ibar(tau+L+1) >= eps after it; for (c)
        // the lower bound base_next + xp + q/2 is used. (c) is monotone in q, hence once a
        // q is prunable all larger q are prunable too.
        if (g_lemmaD && q > 0) {
          const bool a = (double)xp + 1e-9 >= beta[tau];
          const bool b = base + (double)xp >= kEps;
          const bool c = (tau == Tp) || (base_next + (double)xp + 0.5 * (double)q >= kEps);
          if (a && b && c) break;
        }
        const double Ibar = base + 0.5 * (double)(xp + x);
        if (Ibar < kEps) continue;  // ITO undefined for empty stock
        const double term = g_noito ? 0.0 : num / Ibar;
        for (int a = 0; a < (int)src.size(); ++a) {
          Label L{src[a].k + (q > 0), src[a].sx + (double)x, src[a].ito + term, i, a};
          if (g_fast) stair_insert(st[j], L);
          else insert_label(lab[tau][j], L);
        }
      }
    }
    if (g_fast)  // compact surviving labels; next stage refers to these indices
      for (int j = 0; j < S; ++j) {
        size_t n = 0;
        for (auto& v : st[j]) n += v.size();
        lab[tau][j].reserve(n);
        for (auto& v : st[j]) for (auto& L : v) lab[tau][j].push_back(L);
      }
  }
  vector<InnerPoint> out;
  size_t cnt = 0;
  for (int tau = 0; tau <= Tp; ++tau)
    for (int i = 0; i < S; ++i) cnt += lab[tau][i].size();
  if (label_count) *label_count = cnt;
  for (int j = 0; j < S; ++j) {
    for (int a = 0; a < (int)lab[Tp][j].size(); ++a) {
      const Label& L = lab[Tp][j][a];
      const long XT = cx.states[j];
      InnerPoint P;
      P.k = L.k;
      P.POC = in.p * XT + in.c * L.k;
      P.HC = in.h * (C0 + L.sx - 0.5 * XT);
      P.ITO = ito_early + L.ito;
      // reconstruct plan
      P.Q.assign(in.T + 1, 0);
      int s = j, idx = a;
      for (int tt = Tp; tt >= 1; --tt) {
        const Label& cur = lab[tt][s][idx];
        P.Q[tt] = cx.states[s] - cx.states[cur.prev_state];
        s = cur.prev_state; idx = cur.prev_label;
      }
      out.push_back(move(P));
    }
  }
  // labels at different final states may dominate each other in (POC,HC,ITO)
  vector<InnerPoint> nd;
  for (size_t i = 0; i < out.size(); ++i) {
    bool dom = false;
    for (size_t j = 0; j < out.size() && !dom; ++j) {
      if (i == j) continue;
      const auto &a = out[j], &b = out[i];
      bool weak = a.POC <= b.POC + 1e-7 && a.HC <= b.HC + 1e-7 && a.ITO >= b.ITO - 1e-9;
      bool strict = a.POC < b.POC - 1e-7 || a.HC < b.HC - 1e-7 || a.ITO > b.ITO + 1e-9;
      if (weak && (strict || j < i)) dom = true;  // drop exact duplicates too
    }
    if (!dom) nd.push_back(out[i]);
  }
  return nd;
}

// ---------------------------------------------------------------- brute force
static void brute_rec(const Instance& in, const vector<double>& beta, long Xmax,
                      int tau, long xp, int k, double sx, double ito,
                      vector<long>& Q, vector<InnerPoint>& all, double C0, double ito_early) {
  const int Tp = in.T - in.L;
  if (tau > Tp) {
    const double POC = in.p * xp + in.c * k, HC = in.h * (C0 + sx - 0.5 * xp),
                 ITO = ito_early + ito;
    // streaming non-dominated filter (keeps memory small)
    for (const auto& a : all)
      if (a.POC <= POC + 1e-7 && a.HC <= HC + 1e-7 && a.ITO >= ITO - 1e-9) return;
    size_t w = 0;
    for (size_t i = 0; i < all.size(); ++i) {
      const auto& b = all[i];
      bool dom = POC <= b.POC + 1e-7 && HC <= b.HC + 1e-7 && ITO >= b.ITO - 1e-9;
      if (!dom) all[w++] = all[i];
    }
    all.resize(w);
    InnerPoint P;
    P.k = k; P.POC = POC; P.HC = HC; P.ITO = ITO; P.Q = Q;
    all.push_back(move(P));
    return;
  }
  const int t = tau + in.L;
  const double base = in.I0 - 0.5 * (in.cumD[t - 1] + in.cumD[t]);
  for (long q = 0; xp + q <= Xmax; q = (q == 0 ? in.moq : q + in.r)) {
    const long x = xp + q;
    if (x + 1e-9 < beta[tau]) continue;
    const double Ibar = base + 0.5 * (double)(xp + x);
    if (Ibar < kEps) continue;
    Q[tau] = q;
    brute_rec(in, beta, Xmax, tau + 1, x, k + (q > 0), sx + x,
              ito + (in.D[t] + in.sigma) / Ibar, Q, all, C0, ito_early);
    Q[tau] = 0;
  }
}

vector<InnerPoint> brute_inner(const Instance& in, const vector<double>& beta, long Xmax) {
  double C0 = 0.0;
  for (int t = 1; t <= in.T; ++t) C0 += in.I0 - 0.5 * (in.cumD[t - 1] + in.cumD[t]);
  double ito_early = 0.0;
  for (int t = 1; t <= in.L; ++t)
    ito_early += (in.D[t] + in.sigma) / (in.I0 - 0.5 * (in.cumD[t - 1] + in.cumD[t]));
  vector<InnerPoint> all;
  vector<long> Q(in.T + 1, 0);
  brute_rec(in, beta, Xmax, 1, 0, 0, 0.0, 0.0, Q, all, C0, ito_early);
  vector<InnerPoint> nd;
  for (size_t i = 0; i < all.size(); ++i) {
    bool dom = false;
    for (size_t j = 0; j < all.size() && !dom; ++j) {
      if (i == j) continue;
      const auto &a = all[j], &b = all[i];
      bool weak = a.POC <= b.POC + 1e-7 && a.HC <= b.HC + 1e-7 && a.ITO >= b.ITO - 1e-9;
      bool strict = a.POC < b.POC - 1e-7 || a.HC < b.HC - 1e-7 || a.ITO > b.ITO + 1e-9;
      if (weak && (strict || j < i)) dom = true;
    }
    if (!dom) nd.push_back(all[i]);
  }
  return nd;
}

static bool same_front(vector<InnerPoint> a, vector<InnerPoint> b) {
  auto key = [](const InnerPoint& x, const InnerPoint& y) {
    if (fabs(x.POC - y.POC) > 1e-6) return x.POC < y.POC;
    if (fabs(x.HC - y.HC) > 1e-6) return x.HC < y.HC;
    return x.ITO > y.ITO;
  };
  sort(a.begin(), a.end(), key); sort(b.begin(), b.end(), key);
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i)
    if (fabs(a[i].POC - b[i].POC) > 1e-6 || fabs(a[i].HC - b[i].HC) > 1e-6 ||
        fabs(a[i].ITO - b[i].ITO) > 1e-9) return false;
  return true;
}

int run_verify(int nseeds) {
  int pass = 0, total = 0;
  for (int seed = 1; seed <= nseeds; ++seed) {
    for (int sku = 1; sku <= 2; ++sku) {
      // small horizon so that brute force is feasible
      Instance in = make_instance(sku, (sku == 1 ? 11 : 9), 1000u + seed);
      mt19937 g(seed * 7 + sku);
      uniform_real_distribution<double> u(0.0, 1.0);
      double SOT = in.MS * u(g) * 0.8;
      double SS = floor(in.mu * (in.MS - SOT) * u(g));
      vector<double> beta = requirement(in, SS, SOT);
      double bmax = *max_element(beta.begin() + 1, beta.end());
      long bmaxc = (long)ceil(bmax);
      long Xmax_dp = max(0L, bmaxc) + max(in.moq, in.r) + (long)ceil(2 * kEps);  // Lemma 2
      long Xmax_bf = max(0L, bmaxc) + 3 * max(in.moq, in.r);   // deliberately larger
      g_fast = true;  auto A = solve_inner(in, beta, Xmax_dp);
      g_fast = false; g_lemmaD = false; auto A2 = solve_inner(in, beta, Xmax_dp);
      g_fast = true; g_lemmaD = true;
      if (!same_front(A, A2)) printf("  ! fast and reference DP differ\n");
      auto B = brute_inner(in, beta, Xmax_bf);
      bool ok = same_front(A, B);
      ++total; pass += ok;
      printf("seed %2d sku %d  T=%d  SS=%6.1f SOT=%.3f  |front| DP=%3zu BF=%3zu  %s\n",
             seed, sku, in.T, SS, SOT, A.size(), B.size(), ok ? "OK" : "MISMATCH");
    }
  }
  printf("verify: %d / %d instances identical (DP with lemma bound vs brute force with 3x bound)\n",
         pass, total);
  return pass == total ? 0 : 1;
}

// ---------------------------------------------------------------- full front
struct FrontPoint {
  double SS, SOT, CSL, POC, HC, ITO;
  int k;
  bool attained;
  vector<long> Q;
};

// Candidate outer points for the exact treatment of continuous (SS, SOT).
// Every requirement beta_tau = SS + SOT*D(tau+L) + c_tau is rounded up to the lattice;
// the inner problem is constant on each cell of the arrangement of the lines
//   SS + D(tau+L)*SOT = lambda - c_tau   (lambda in lattice)
// and CSL increases with B = SS + mu*SOT, whose maximum over a cell (closed on its
// sides) has its supremum at a vertex of the cell closure. That vertex need NOT
// belong to the cell (it may lie on an open side), so the supremum may not be
// attained: the continuous Pareto front is in general not closed. See
// cell_candidates() for the exact enumeration of cells and their suprema.
struct Line { double a, b, rhs; };  // a*SS + b*SOT = rhs
vector<Line> g_lines;               // lines of the last arrangement (used by cell probes)
vector<pair<double, double>> arrangement_vertices(const Instance& in,
                                                  const vector<long>& lattice) {
  const int Tp = in.T - in.L;
  vector<Line> lines;
  lines.push_back({1.0, 0.0, 0.0});                    // SS = 0
  lines.push_back({0.0, 1.0, 0.0});                    // SOT = 0
  lines.push_back({1.0, in.mu, in.mu * in.MS});        // SS + mu*SOT = mu*MS
  map<pair<long, long>, int> seen;  // (D, rhs*1e6) dedup of identical lines
  for (int tau = 1; tau <= Tp; ++tau) {
    const int t = tau + in.L;
    const double ct = (double)in.cumD[t] - (double)in.I0;
    const double lo = ct, hi = ct + max(in.mu * in.MS, in.MS * (double)in.D[t]);
    for (long lam : lattice) {
      if (lam <= lo + 1e-12 || lam > hi + 1e-9) continue;  // line must cross the domain
      const double rhs = (double)lam - ct;
      auto key = make_pair((long)in.D[t], llround(rhs * 1e6));
      if (seen.count(key)) continue;
      seen[key] = 1;
      lines.push_back({1.0, (double)in.D[t], rhs});
    }
  }
  vector<pair<double, double>> V;
  V.reserve(lines.size() * lines.size() / 2);
  const double eps = 1e-9;
  for (size_t i = 0; i < lines.size(); ++i)
    for (size_t j = i + 1; j < lines.size(); ++j) {
      const Line &A = lines[i], &B = lines[j];
      const double det = A.a * B.b - A.b * B.a;
      if (fabs(det) < 1e-12) continue;
      const double SS = (A.rhs * B.b - A.b * B.rhs) / det;
      const double SOT = (A.a * B.rhs - A.rhs * B.a) / det;
      if (SS < -eps || SOT < -eps || SS + in.mu * SOT > in.mu * in.MS + 1e-7) continue;
      V.push_back({max(0.0, SS), max(0.0, SOT)});
    }
  sort(V.begin(), V.end());
  V.erase(unique(V.begin(), V.end(), [](auto& x, auto& y) {
            return fabs(x.first - y.first) < 1e-9 && fabs(x.second - y.second) < 1e-12;
          }), V.end());
  g_lines = lines;
  return V;
}

// Cells of the arrangement and the supremum of B = SS + mu*SOT on each cell.
// A cell is half-open (closed on the upper side of every strip), so the supremum
// of B is attained only if the maximising vertex belongs to the cell. For every
// vertex we probe each angular sector between the lines through it; every cell
// incident to the vertex is reached by exactly one probe. Each candidate carries
// the cell supremum Bsup (= B at the vertex) and whether it is attained.
struct OuterCand { double SS, SOT, Bsup; bool attained; };
vector<OuterCand> cell_candidates(const Instance& in, const vector<long>& lattice) {
  auto V = arrangement_vertices(in, lattice);
  const auto& lines = g_lines;
  vector<OuterCand> C;
  const double rho = 1e-5;  // probe radius (well below any lattice-line spacing)
  for (auto [SS, SOT] : V) {
    const double B = SS + in.mu * SOT;
    C.push_back({SS, SOT, B, true});
    vector<double> ang;
    for (const auto& Ln : lines) {
      double scale = max(1.0, fabs(Ln.rhs));
      if (fabs(Ln.a * SS + Ln.b * SOT - Ln.rhs) <= 1e-9 * scale) {
        // work in normalised coordinates u = SS/mu, v = SOT: line (a*mu) u + b v = rhs,
        // direction vector (b, -a*mu)
        double th = atan2(-Ln.a * in.mu, Ln.b);
        ang.push_back(th);
        ang.push_back(th > 0 ? th - M_PI : th + M_PI);
      }
    }
    if (!g_probes || ang.empty()) continue;  // vertices suffice (Theorem 3)
    sort(ang.begin(), ang.end());
    ang.erase(unique(ang.begin(), ang.end(), [](double x, double y) { return fabs(x - y) < 1e-12; }),
              ang.end());
    for (size_t k = 0; k < ang.size(); ++k) {
      double a0 = ang[k], a1 = (k + 1 < ang.size()) ? ang[k + 1] : ang[0] + 2 * M_PI;
      double mid = 0.5 * (a0 + a1);
      // probe in normalised coordinates (u, v) = (SS/mu, SOT), mapped back
      double pSS = SS + rho * in.mu * cos(mid), pSOT = SOT + rho * sin(mid);
      if (pSS < 0 || pSOT < 0 || pSS + in.mu * pSOT > in.mu * in.MS) continue;
      C.push_back({pSS, pSOT, B, false});
    }
  }
  return C;
}

int run_front(int sku, int T, unsigned seed, double dSS, double dSOT, const string& out,
              int threads, const Overrides& ov, bool continuous) {
#ifdef _OPENMP
  if (threads > 0) omp_set_num_threads(threads);
#endif
  Instance in = make_instance(sku, T, seed, ov);
  auto t0 = chrono::steady_clock::now();
  // 1) enumerate outer grid and group by rounded requirement vector
  struct Rep { double SS, SOT, B; bool attained; };  // B = (supremum of) SS + mu*SOT
  map<vector<long>, Rep> groups;
  long ncomb = 0;
  long bmax_all = 0;
  vector<long> reach_probe;  // for rounding: reachable values up to a generous bound
  {
    // beta is affine in (SS, SOT): its maximum over the domain is at a corner
    auto b1 = requirement(in, in.mu * in.MS, 0.0), b2 = requirement(in, 0.0, in.MS);
    double bmax = max(*max_element(b1.begin() + 1, b1.end()), *max_element(b2.begin() + 1, b2.end()));
    if (g_fixSS >= 0) {  // a fixed outer point may lie outside the domain corners
      auto b3 = requirement(in, g_fixSS, max(0.0, g_fixSOT));
      bmax = max(bmax, *max_element(b3.begin() + 1, b3.end()));
    }
    bmax_all = (long)ceil(bmax);
    reach_probe = reachable_values(in, max(0L, bmax_all) + 2 * max(in.moq, in.r));
  }
  // tolerance: a point lying (numerically) on a lattice line belongs to the cell
  // that is closed on that side, i.e. beta = lambda rounds to lambda
  auto ceil_reach = [&](double b) -> long {
    long need = (long)ceil(b - 1e-7);
    if (need <= 0) return 0;
    auto it = lower_bound(reach_probe.begin(), reach_probe.end(), need);
    if (it == reach_probe.end()) {  // never clamp silently: this would relax a constraint
      fprintf(stderr, "internal error: requirement %.3f beyond lattice range %ld\n", b,
              reach_probe.back());
      exit(3);
    }
    return *it;
  };
  vector<OuterCand> cand;
  if (g_fixSS >= 0) {
    cand.push_back({g_fixSS, max(0.0, g_fixSOT), g_fixSS + in.mu * max(0.0, g_fixSOT), true});
  } else if (continuous) {
    cand = cell_candidates(in, reach_probe);
  } else {
    for (double SOT = 0; SOT <= in.MS + 1e-12; SOT += dSOT) {
      double SSmax = in.mu * (in.MS - SOT);
      for (double SS = 0; SS <= SSmax + 1e-9; SS += dSS)
        cand.push_back({SS, SOT, SS + in.mu * SOT, true});
    }
  }
  for (const auto& oc : cand) {
    ++ncomb;
    auto beta = requirement(in, oc.SS, oc.SOT);
    vector<long> key(beta.size());
    for (size_t i = 1; i < beta.size(); ++i) key[i] = ceil_reach(beta[i]);
    auto it = groups.find(key);
    if (it == groups.end() || oc.Bsup > it->second.B + 1e-12 ||
        (fabs(oc.Bsup - it->second.B) <= 1e-12 && oc.attained && !it->second.attained))
      groups[key] = {oc.SS, oc.SOT, oc.Bsup, oc.attained};
  }
  vector<pair<vector<long>, Rep>> G(groups.begin(), groups.end());
  auto t1 = chrono::steady_clock::now();
  // 2) solve inner problems in parallel
  vector<vector<FrontPoint>> res(G.size());
  vector<size_t> labs(G.size(), 0);
#pragma omp parallel for schedule(dynamic, 1)
  for (long gi = 0; gi < (long)G.size(); ++gi) {
    const Rep& R = G[gi].second;
    // use the rounded key itself as requirement vector: identical rounding
    // decisions in grouping and DP (robust at arrangement vertices)
    const vector<long>& key = G[gi].first;
    vector<double> beta(key.begin(), key.end());
    double bmax = *max_element(beta.begin() + 1, beta.end());
    long Xmax = max(0L, (long)ceil(bmax)) + max(in.moq, in.r) + (long)ceil(2 * kEps);  // Lemma 2
    size_t lc = 0;
    auto pts = solve_inner(in, beta, Xmax, &lc);
    labs[gi] = lc;
    double CSL = Phi(R.B / in.sigma);  // supremum over the cell (attained flag below)
    for (auto& P : pts)
      res[gi].push_back({R.SS, R.SOT, CSL, P.POC, P.HC, P.ITO, P.k, R.attained, move(P.Q)});
  }
  auto t2 = chrono::steady_clock::now();
  // 3) global 4-objective non-dominated filter, exploiting that CSL is constant
  //    within a group: process CSL levels in decreasing order; a candidate at
  //    level c can only be dominated by points with CSL >= c. Archive keeps the
  //    3-D (POC,HC,ITO) non-dominated set of all accepted points so far.
  size_t ncand = 0;
  for (auto& v : res) ncand += v.size();
  vector<long> order(G.size());
  iota(order.begin(), order.end(), 0L);
  auto cslOf = [&](long g) { return Phi(G[g].second.B / in.sigma); };
  sort(order.begin(), order.end(), [&](long a, long b) { return cslOf(a) > cslOf(b); });
  auto dom3 = [](const FrontPoint& a, const FrontPoint& b) {  // a weakly dominates b in 3-D
    return a.POC <= b.POC + 1e-7 && a.HC <= b.HC + 1e-7 && a.ITO >= b.ITO - 1e-9;
  };
  vector<FrontPoint> archive;  // sorted by POC
  vector<FrontPoint> front;
  size_t oi = 0;
  while (oi < order.size()) {
    // batch of groups with identical CSL
    double c = cslOf(order[oi]);
    vector<FrontPoint> batch;
    while (oi < order.size() && fabs(cslOf(order[oi]) - c) <= 1e-15) {
      for (auto& p : res[order[oi]]) batch.push_back(move(p));
      ++oi;
    }
    // (a) remove candidates dominated by the archive (higher CSL) — parallel
    vector<char> keep(batch.size(), 1);
#pragma omp parallel for schedule(dynamic, 256)
    for (long i = 0; i < (long)batch.size(); ++i) {
      const auto& b = batch[i];
      auto end = upper_bound(archive.begin(), archive.end(), b.POC + 1e-7,
                             [](double v, const FrontPoint& a) { return v < a.POC; });
      for (auto it = archive.begin(); it != end; ++it)
        if (dom3(*it, b)) { keep[i] = 0; break; }
    }
    vector<FrontPoint> surv;
    for (size_t i = 0; i < batch.size(); ++i) if (keep[i]) surv.push_back(move(batch[i]));
    // (b) 3-D non-dominated filter within the batch (same CSL)
    sort(surv.begin(), surv.end(), [](const FrontPoint& a, const FrontPoint& b) {
      if (a.POC != b.POC) return a.POC < b.POC;
      if (a.HC != b.HC) return a.HC < b.HC;
      return a.ITO > b.ITO;
    });
    vector<FrontPoint> nd;
    for (auto& b : surv) {
      bool d = false;
      for (auto& a : nd) if (dom3(a, b)) { d = true; break; }
      if (!d) nd.push_back(b);
    }
    for (auto& p : nd) front.push_back(p);
    // (c) merge accepted points into the archive (kept sorted by POC). The
    //     archive need not be 3-D non-dominated: a superset gives the same answer.
    vector<FrontPoint> merged;
    merged.reserve(archive.size() + nd.size());
    merge(archive.begin(), archive.end(), nd.begin(), nd.end(), back_inserter(merged),
          [](const FrontPoint& a, const FrontPoint& b) { return a.POC < b.POC; });
    archive.swap(merged);
  }
  auto t3 = chrono::steady_clock::now();
  auto sec = [](auto a, auto b) { return chrono::duration<double>(b - a).count(); };
  size_t totlab = accumulate(labs.begin(), labs.end(), (size_t)0);
  printf("instance: SKU%d T=%d L=%d moq=r=%ld mu=%.2f sigma=%.2f c=%.2f h=%.5f p=%.2f MS=%.2f seed=%u I0=%ld\n",
         sku, in.T, in.L, in.moq, in.mu, in.sigma, in.c, in.h, in.p, in.MS, seed, in.I0);
  if (continuous)
    printf("outer: continuous (arrangement) -> %ld candidates (vertices + sector probes), %zu distinct requirement vectors\n",
           ncomb, G.size());
  else
    printf("outer grid: dSS=%.3f dSOT=%.3f -> %ld combinations, %zu distinct requirement vectors\n",
           dSS, dSOT, ncomb, G.size());
  printf("labels generated (total): %zu   candidate points: %zu   Pareto front: %zu\n",
         totlab, ncand, front.size());
  printf("time: grouping %.3fs  inner DPs %.3fs  final filter %.3fs  TOTAL %.3fs\n",
         sec(t0, t1), sec(t1, t2), sec(t2, t3), sec(t0, t3));
  FILE* f = fopen(out.c_str(), "w");
  if (!f) { perror("fopen"); return 1; }
  fprintf(f, "SS,SOT,CSL,POC,HC,ITO,orders,attained");
  for (int t = 1; t <= in.T; ++t) fprintf(f, ",Q%d", t);
  fprintf(f, "\n");
  for (auto& P : front) {
    fprintf(f, "%.6f,%.8f,%.10f,%.4f,%.6f,%.8f,%d,%d", P.SS, P.SOT, P.CSL, P.POC, P.HC, P.ITO, P.k, (int)P.attained);
    for (int t = 1; t <= in.T; ++t) fprintf(f, ",%ld", P.Q[t]);
    fprintf(f, "\n");
  }
  fclose(f);
  // demand file for reproducibility
  string dfile = out + ".demand.csv";
  FILE* g = fopen(dfile.c_str(), "w");
  fprintf(g, "t,D\n");
  for (int t = 1; t <= in.T; ++t) fprintf(g, "%d,%ld\n", t, in.D[t]);
  fclose(g);
  printf("wrote %s and %s\n", out.c_str(), dfile.c_str());
  return 0;
}

// Numerical check of the arrangement theorem: for random (SS,SOT) in the domain,
// (i) the rounded requirement vector must be among those produced by the vertices,
// (ii) its B = SS + mu*SOT must not exceed the best B recorded for that vector.
int run_verify_cont(int sku, int T, unsigned seed, long nsamples, const Overrides& ov) {
  // Numerical check of Theorem 3: for every (SS,SOT) in the domain there is an
  // arrangement vertex v whose rounded requirement vector is componentwise <= that of
  // (SS,SOT) (hence a superset of feasible plans) and whose B = SS + mu*SOT is >=.
  Instance in = make_instance(sku, T, seed, ov);
  auto b1 = requirement(in, in.mu * in.MS, 0.0), b2 = requirement(in, 0.0, in.MS);
  double bmax = max(*max_element(b1.begin() + 1, b1.end()), *max_element(b2.begin() + 1, b2.end()));
  auto lat = reachable_values(in, max(0L, (long)ceil(bmax)) + 2 * max(in.moq, in.r));
  auto ceil_reach = [&](double b) -> long {
    long need = (long)ceil(b - 1e-7);
    if (need <= 0) return 0;
    auto it = lower_bound(lat.begin(), lat.end(), need);
    if (it == lat.end()) { fprintf(stderr, "lattice range\n"); exit(3); }
    return *it;
  };
  auto keyOf = [&](double SS, double SOT) {
    auto beta = requirement(in, SS, SOT);
    vector<long> k(beta.size());
    for (size_t i = 1; i < beta.size(); ++i) k[i] = ceil_reach(beta[i]);
    return k;
  };
  bool save = g_probes; g_probes = false;
  auto C = cell_candidates(in, lat);  // vertices only
  g_probes = save;
  map<vector<long>, double> best;
  for (const auto& oc : C) {
    auto k = keyOf(oc.SS, oc.SOT);
    auto it = best.find(k);
    if (it == best.end() || oc.Bsup > it->second) best[k] = oc.Bsup;
  }
  vector<pair<double, vector<long>>> V;  // sorted by B descending
  for (auto& [k, B] : best) V.push_back({B, k});
  sort(V.begin(), V.end(), [](auto& a, auto& b) { return a.first > b.first; });
  mt19937_64 g(seed + 99);
  uniform_real_distribution<double> u(0.0, 1.0);
  long fail = 0;
  for (long s = 0; s < nsamples; ++s) {
    double SOT = in.MS * u(g), SS = in.mu * (in.MS - SOT) * u(g);
    if (s % 4 == 0) SS = in.mu * (in.MS - SOT);  // probe the upper boundary as well
    auto k = keyOf(SS, SOT);
    const double B = SS + in.mu * SOT;
    bool ok = false;
    for (auto& [Bv, kv] : V) {
      if (Bv < B - 1e-9) break;  // remaining vertices have smaller B
      bool le = true;
      for (size_t i = 1; i < k.size() && le; ++i) le = kv[i] <= k[i];
      if (le) { ok = true; break; }
    }
    fail += !ok;
  }
  printf("verify-cont SKU%d T=%d seed=%u: %zu vertices, %zu distinct vectors; %ld random points -> "
         "points without a dominating vertex: %ld  => %s\n",
         sku, T, seed, C.size(), best.size(), nsamples, fail, fail == 0 ? "OK" : "FAIL");
  return fail == 0 ? 0 : 1;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr,
            "usage:\n  %s verify [--seeds N]\n"
            "  %s verify-cont [--sku 1|2] [--T 48] [--seed S] [--samples N]\n"
            "  %s front [--sku 1|2] [--T 48] [--seed S] [--outer grid|continuous] [--dSS 1] [--dSOT 0.05]\n"
            "          [--mu m] [--cv v] [--c c] [--hfrac f] [--MS w] [--r lot] [--L lead]\n"
            "          [--out f.csv] [--threads n]\n",
            argv[0], argv[0], argv[0]);
    return 2;
  }
  string mode = argv[1];
  int sku = 1, T = 48, seeds = 20, threads = 0;
  unsigned seed = 42;
  long samples = 200000;
  double dSS = 1.0, dSOT = 0.05;
  string out = "front.csv", outer = "grid";
  Overrides ov;
  for (int i = 2; i + 1 < argc; i += 2) {
    string a = argv[i], v = argv[i + 1];
    if (a == "--sku") sku = stoi(v);
    else if (a == "--T") T = stoi(v);
    else if (a == "--seed") seed = (unsigned)stoul(v);
    else if (a == "--seeds") seeds = stoi(v);
    else if (a == "--samples") samples = stol(v);
    else if (a == "--dSS") dSS = stod(v);
    else if (a == "--dSOT") dSOT = stod(v);
    else if (a == "--out") out = v;
    else if (a == "--threads") threads = stoi(v);
    else if (a == "--outer") outer = v;
    else if (a == "--mu") ov.mu = stod(v);
    else if (a == "--cv") ov.cv = stod(v);
    else if (a == "--c") ov.c = stod(v);
    else if (a == "--hfrac") ov.hfrac = stod(v);
    else if (a == "--MS") ov.MS = stod(v);
    else if (a == "--r") ov.r = stol(v);
    else if (a == "--L") ov.L = stoi(v);
    else if (a == "--ref") { g_fast = (stoi(v) == 0); g_lemmaD = g_fast; }
    else if (a == "--noito") g_noito = (stoi(v) != 0);
    else if (a == "--SS") g_fixSS = stod(v);
    else if (a == "--SOT") g_fixSOT = stod(v);
    else if (a == "--probes") g_probes = (stoi(v) != 0);
    else if (a == "--lemmaD") g_lemmaD = (stoi(v) != 0);
    else { fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
  }
  if (mode == "verify") return run_verify(seeds);
  if (mode == "verify-cont") return run_verify_cont(sku, T, seed, samples, ov);
  if (mode == "front") return run_front(sku, T, seed, dSS, dSOT, out, threads, ov, outer == "continuous");
  fprintf(stderr, "unknown mode\n");
  return 2;
}

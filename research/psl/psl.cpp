// psl.cpp - preference-conditioned policy for multiobjective lot sizing, trained with REINFORCE
// and evaluated against exact Pareto fronts. Pure C++ (STL + OpenMP), reuses src/lsdp.cpp.
//
//   psl gradcheck
//   psl pool  --n 2000 --T 48 --seed0 1 --out pool_train.txt      (exact fronts -> ideal/nadir)
//   psl train --pool pool_train.txt --iters 20000 --batch 32 --K 8 --out model
//   psl eval  --pool pool_test.txt --model model --H 6 [--samples 16]
//   psl evalreal ../../bench/instances/*/manifest.csv --model model   (E6 data, fronts from Paper 1)
#include "psl_core.hpp"
#include "psl_nn.hpp"
#include <fstream>
#include <sstream>

// ------------------------------------------------------------------ synthetic instances
struct GenSpec { unsigned seed; int T; };
static bool gen_instance(unsigned seed, int T, Instance& in, string* kind = nullptr) {
  mt19937_64 g(seed * 2654435761ULL + 12345);
  uniform_real_distribution<double> U(0.0, 1.0);
  auto logu = [&](double a, double b) { return exp(log(a) + (log(b) - log(a)) * U(g)); };
  in = Instance();
  in.T = T;
  in.mu = logu(1.0, 200.0);
  double cv = 0.1 + 0.9 * U(g);
  in.sigma = cv * in.mu;
  in.r = max(1L, lround(in.mu / (1.0 + 3.0 * U(g))));
  in.moq = in.r * ((U(g) < 0.75) ? 1 : 2);
  in.L = 1 + (int)(5 * U(g));
  in.p = logu(1.0, 500.0);
  in.h = 0.1 * in.p / 52.0;
  in.c = 0.1 * in.p * 16.0 * in.mu / 104.0 * logu(0.25, 4.0);
  in.MS = 0.5 + U(g);
  in.D.assign(T + 1, 0); in.cumD.assign(T + 1, 0);
  double u = U(g);
  normal_distribution<double> nd(0.0, 1.0);
  string k;
  if (u < 0.6) {            // stationary
    k = "stationary";
    for (int t = 1; t <= T; ++t) in.D[t] = max(0L, lround(in.mu + in.sigma * nd(g)));
  } else if (u < 0.8) {     // intermittent: zero with probability 1-q, scaled to mean mu
    k = "intermittent";
    double q = 0.2 + 0.6 * U(g);
    for (int t = 1; t <= T; ++t)
      in.D[t] = (U(g) < q) ? max(0L, lround(in.mu / q + in.sigma / sqrt(q) * nd(g))) : 0;
  } else {                  // seasonal
    k = "seasonal";
    double A = 0.2 + 0.6 * U(g), per = 8 + 44 * U(g), ph = 6.283 * U(g);
    for (int t = 1; t <= T; ++t)
      in.D[t] = max(0L, lround(in.mu * (1 + A * sin(6.283 * t / per + ph)) + in.sigma * nd(g)));
  }
  if (kind) *kind = k;
  for (int t = 1; t <= T; ++t) in.cumD[t] = in.cumD[t - 1] + in.D[t];
  in.I0 = in.cumD[in.L] + max(lround(in.mu), 1L);
  for (int t = 1; t <= in.L; ++t)
    if (in.I0 - 0.5 * (in.cumD[t - 1] + in.cumD[t]) < kEps) return false;
  return true;
}

// pool line: seed T kind mu sigma r moq L p h c MS I0 | D1..DT | ideal0..3 nadir0..3 frontsize
struct PoolItem { Instance in; string kind; double ideal[4], nadir[4]; long fsize; };
static void write_item(FILE* f, unsigned seed, const string& kind, const Prepared& P) {
  const Instance& in = P.in;
  fprintf(f, "%u %d %s %.10g %.10g %ld %ld %d %.10g %.10g %.10g %.10g %ld", seed, in.T, kind.c_str(), in.mu,
          in.sigma, in.r, in.moq, in.L, in.p, in.h, in.c, in.MS, in.I0);
  for (int t = 1; t <= in.T; ++t) fprintf(f, " %ld", in.D[t]);
  for (int i = 0; i < 4; ++i) fprintf(f, " %.12g", P.ideal[i]);
  for (int i = 0; i < 4; ++i) fprintf(f, " %.12g", P.nadir[i]);
  fprintf(f, " %zu\n", P.front.size());
}
static vector<PoolItem> read_pool(const string& fn) {
  vector<PoolItem> V;
  ifstream f(fn);
  string line;
  while (getline(f, line)) {
    istringstream is(line);
    PoolItem it; unsigned seed;
    Instance& in = it.in;
    is >> seed >> in.T >> it.kind >> in.mu >> in.sigma >> in.r >> in.moq >> in.L >> in.p >> in.h >> in.c >> in.MS >> in.I0;
    in.D.assign(in.T + 1, 0); in.cumD.assign(in.T + 1, 0);
    for (int t = 1; t <= in.T; ++t) { is >> in.D[t]; in.cumD[t] = in.cumD[t - 1] + in.D[t]; }
    for (int i = 0; i < 4; ++i) is >> it.ideal[i];
    for (int i = 0; i < 4; ++i) is >> it.nadir[i];
    is >> it.fsize;
    if (is) V.push_back(it);
  }
  return V;
}

// ------------------------------------------------------------------ policy features
constexpr int NCF = 9;    // class features
constexpr int NEF = 23;   // order-event features
struct Scales { double SX, SP, SH, SI; };
static Scales scales(const Instance& in) {
  double SX = max(1.0, in.mu * in.T);
  return {SX, in.p * SX + in.c * in.T, in.h * SX * in.T, in.T * (in.mu + in.sigma) / max(in.mu, 1.0)};
}
static void class_features(const Prepared& P, int g, const array<double, 4>& lam, double* f) {
  const auto& c = P.classes[g];
  Scales S = scales(P.in);
  double mean = 0, mx = 0, pos = 0;
  for (int t = 1; t <= P.Tp; ++t) { mean += c.kappa[t]; mx = max(mx, c.kappa[t]); pos += c.kappa[t] > 0; }
  f[0] = c.B / max(1e-9, P.in.mu * P.in.MS);
  f[1] = Phi(c.B / P.in.sigma);
  f[2] = mean / P.Tp / S.SX * P.in.T;
  f[3] = mx / S.SX;
  f[4] = pos / P.Tp;
  for (int i = 0; i < 4; ++i) f[5 + i] = lam[i];
}
static void event_features(const Prepared& P, const State& s, const Cand& c, const array<double, 4>& lam, double* f) {
  Scales S = scales(P.in);
  const int Tp = P.Tp;
  const auto& cl = P.classes[s.cls];
  int tau = max(1, s.tau);
  f[0] = s.tau / (double)Tp;
  f[1] = s.x / S.SX;
  f[2] = s.k / (double)Tp;
  f[3] = s.W / (S.SX * Tp);
  f[4] = s.R / S.SI;
  f[5] = (Tp - s.tau) / (double)Tp;
  f[6] = (c.tn - tau) / (double)Tp;
  f[7] = c.X / S.SX;
  f[8] = (c.X - s.x) / S.SX;
  f[9] = c.dPOC / S.SP;
  f[10] = P.in.h * c.dW / S.SH;
  f[11] = c.dITO / S.SI;
  f[12] = (c.tn <= Tp) ? (cl.kappa[c.tn] - c.X) / S.SX : 0.0;
  f[13] = (Tp + 1 - c.tn) / (double)Tp;
  f[14] = (c.tn > Tp) ? 1.0 : 0.0;
  f[15] = (c.tn <= Tp) ? (P.b[c.tn] + c.X) / S.SX : 0.0;
  f[16] = (s.tau == 0) ? 1.0 : 0.0;
  f[17] = cl.B / max(1e-9, P.in.mu * P.in.MS);
  f[18] = Phi(cl.B / P.in.sigma);
  for (int i = 0; i < 4; ++i) f[19 + i] = lam[i];
}

struct Decision { vector<double> feats; int n, a; bool is_class; };
struct Policy {
  MLP cls, ev;
  Policy(unsigned seed = 1) : cls({NCF, 64, 64, 1}, seed), ev({NEF, 96, 96, 1}, seed + 7) {}
};

// one episode; returns outcome; records decisions if traj != nullptr
static array<double, 4> rollout_event(const Prepared& P, const Policy& pi, const array<double, 4>& lam, bool sample,
                                      mt19937_64& g, vector<Decision>* traj) {
  uniform_real_distribution<double> U(0.0, 1.0);
  auto choose = [&](const vector<double>& feats, int n, int dim, const MLP& net, bool is_class) {
    vector<double> sc(n);
    double mx = -1e300;
    for (int i = 0; i < n; ++i) { sc[i] = net.forward(&feats[(size_t)i * dim]); mx = max(mx, sc[i]); }
    int a = 0;
    if (sample) {
      double z = 0; for (auto& v : sc) { v = exp(v - mx); z += v; }
      double u = U(g) * z, acc = 0;
      for (a = 0; a < n - 1; ++a) { acc += sc[a]; if (acc >= u) break; }
    } else {
      a = int(max_element(sc.begin(), sc.end()) - sc.begin());
    }
    if (traj) traj->push_back({feats, n, a, is_class});
    return a;
  };
  State s;
  {
    const int n = (int)P.classes.size();
    vector<double> F((size_t)n * NCF);
    for (int i = 0; i < n; ++i) class_features(P, i, lam, &F[(size_t)i * NCF]);
    s.cls = choose(F, n, NCF, pi.cls, true);
  }
  while (!s.done) {
    auto C = candidates(P, s);
    const int n = (int)C.size();
    vector<double> F((size_t)n * NEF);
    for (int i = 0; i < n; ++i) event_features(P, s, C[i], lam, &F[(size_t)i * NEF]);
    int a = choose(F, n, NEF, pi.ev, false);
    step(P, s, C[a]);
  }
  return outcome(P, s);
}

// ------------------------------------------------------------------ ablation: per-period quantity actions
// The standard RL formulation of lot sizing: in every period u the agent either orders nothing or
// raises the cumulative quantity to one of g_qlevels levels X' >= X+moq on the r-lattice, starting at
// the least feasible one. Infeasible actions are masked, so every episode yields a feasible plan; the
// plan is mapped onto the same order events (and therefore the same objectives) as the main policy.
static int g_action = 0;       // 0 = order events (Theorem A.10), 1 = per-period quantities
static int g_qlevels = 24;
static array<double, 4> rollout_q(const Prepared& P, const Policy& pi, const array<double, 4>& lam, bool sample,
                                  mt19937_64& g, vector<Decision>* traj) {
  uniform_real_distribution<double> U(0.0, 1.0);
  const Instance& in = P.in;
  const int Tp = P.Tp;
  const long r = in.r, moq = in.moq;
  Scales S = scales(in);
  auto choose = [&](const vector<double>& feats, int n, int dim, const MLP& net, bool is_class) {
    vector<double> sc(n);
    double mx = -1e300;
    for (int i = 0; i < n; ++i) { sc[i] = net.forward(&feats[(size_t)i * dim]); mx = max(mx, sc[i]); }
    int a = 0;
    if (sample) {
      double z = 0; for (auto& v : sc) { v = exp(v - mx); z += v; }
      double u = U(g) * z, acc = 0;
      for (a = 0; a < n - 1; ++a) { acc += sc[a]; if (acc >= u) break; }
    } else a = int(max_element(sc.begin(), sc.end()) - sc.begin());
    if (traj) traj->push_back({feats, n, a, is_class});
    return a;
  };
  State s;
  {
    const int n = (int)P.classes.size();
    vector<double> F((size_t)n * NCF);
    for (int i = 0; i < n; ++i) class_features(P, i, lam, &F[(size_t)i * NCF]);
    s.cls = choose(F, n, NCF, pi.cls, true);
  }
  const auto& beta = P.classes[s.cls].kappa;
  auto ceil_r = [&](double y) -> long { return y <= 1e-9 ? 0 : r * (long)ceil((y - 1e-9) / (double)r); };
  // open order event: (tau, xp, X), accumulated turnover of the event
  bool open = false; int tau = 0; long xp = 0, X = 0; double evR = 0, preR = 0;
  for (int u = 1; u <= Tp + 1; ++u) {
    // candidate actions at period u: index 0 = no order (if feasible), then order levels
    vector<long> lev;
    bool can_skip;
    if (u > Tp) can_skip = true;
    else if (!open) can_skip = beta[u] <= 1e-9 && P.b[u] >= kEps;
    else can_skip = (double)X >= beta[u] - 1e-9 && P.b[u] + (double)X >= kEps;
    if (u <= Tp) {
      long cur = open ? X : 0;
      long lo = max(ceil_r(max(beta[u], 2.0 * (kEps - P.b[u]) - (double)cur)), cur + moq);
      for (int j = 0; j < g_qlevels; ++j) lev.push_back(lo + j * r);
    }
    if (u > Tp) {  // horizon end: close
      if (!open) { step(P, s, Cand{Tp + 1, 0, 0.0, 0.0, preR}); }
      else step(P, s, Cand{Tp + 1, X, in.p * (double)(X - xp) + in.c, (double)X * (Tp + 1 - tau), evR});
      break;
    }
    const int n = (int)lev.size() + (can_skip ? 1 : 0);
    vector<double> F((size_t)n * NEF, 0.0);
    long cur = open ? X : 0;
    for (int i = 0; i < n; ++i) {
      double* f = &F[(size_t)i * NEF];
      bool skip = can_skip && i == 0;
      long Xn = skip ? cur : lev[i - (can_skip ? 1 : 0)];
      f[0] = u / (double)Tp; f[1] = cur / S.SX; f[2] = s.k / (double)Tp; f[3] = s.W / (S.SX * Tp);
      f[4] = (s.R + (open ? evR : preR)) / S.SI; f[5] = (Tp - u) / (double)Tp;
      f[6] = open ? (u - tau) / (double)Tp : 0.0;
      f[7] = Xn / S.SX; f[8] = (Xn - cur) / S.SX;
      f[9] = skip ? 0.0 : (in.p * (double)(Xn - cur) + in.c) / S.SP;
      f[10] = in.h * (double)Xn / S.SH;
      double Ib = skip ? P.b[u] + (double)cur : P.b[u] + 0.5 * (double)(cur + Xn);
      f[11] = P.num[u] / max(Ib, kEps) / S.SI;
      f[12] = (beta[u] - Xn) / S.SX;
      f[13] = (Tp + 1 - u) / (double)Tp;
      f[14] = skip ? 1.0 : 0.0;
      f[15] = (P.b[u] + Xn) / S.SX;
      f[16] = open ? 0.0 : 1.0;
      f[17] = P.classes[s.cls].B / max(1e-9, in.mu * in.MS);
      f[18] = Phi(P.classes[s.cls].B / in.sigma);
      for (int t = 0; t < 4; ++t) f[19 + t] = lam[t];
    }
    int a = choose(F, n, NEF, pi.ev, false);
    if (can_skip && a == 0) {
      if (open) evR += P.num[u] / (P.b[u] + (double)X); else preR += P.num[u] / P.b[u];
      continue;
    }
    long Xn = lev[a - (can_skip ? 1 : 0)];
    if (!open) step(P, s, Cand{u, 0, 0.0, 0.0, preR});
    else step(P, s, Cand{u, X, in.p * (double)(X - xp) + in.c, (double)X * (u - tau), evR});
    open = true; tau = u; xp = cur; X = Xn;
    evR = P.num[u] / (P.b[u] + 0.5 * (double)(xp + X));
  }
  return outcome(P, s);
}
static array<double, 4> rollout(const Prepared& P, const Policy& pi, const array<double, 4>& lam, bool sample,
                                mt19937_64& g, vector<Decision>* traj) {
  return g_action == 0 ? rollout_event(P, pi, lam, sample, g, traj) : rollout_q(P, pi, lam, sample, g, traj);
}

// gradient of sum_t log pi(a_t) * coef, accumulated into (gW, gb) of each net
struct Grad { vector<vector<double>> cW, cb, eW, eb; };
static void accumulate_grad(const Policy& pi, const vector<Decision>& traj, double coef, Grad& G) {
  vector<vector<double>> acts;
  for (const auto& d : traj) {
    const MLP& net = d.is_class ? pi.cls : pi.ev;
    const int dim = d.is_class ? NCF : NEF;
    vector<double> sc(d.n);
    double mx = -1e300;
    for (int i = 0; i < d.n; ++i) { sc[i] = net.forward(&d.feats[(size_t)i * dim]); mx = max(mx, sc[i]); }
    double z = 0; for (auto& v : sc) { v = exp(v - mx); z += v; }
    for (int i = 0; i < d.n; ++i) {
      double gl = ((i == d.a) ? 1.0 : 0.0) - sc[i] / z;   // d log pi(a) / d score_i
      if (fabs(gl) < 1e-12) continue;
      net.forward(&d.feats[(size_t)i * dim], &acts);
      if (d.is_class) net.backward(acts, coef * gl, G.cW, G.cb);
      else net.backward(acts, coef * gl, G.eW, G.eb);
    }
  }
}

static Prepared prep_from_pool(const PoolItem& it) {
  Prepared P = prepare(it.in, false);
  for (int i = 0; i < 4; ++i) { P.ideal[i] = it.ideal[i]; P.nadir[i] = it.nadir[i]; }
  return P;
}

static array<double, 4> random_pref(mt19937_64& g) {
  exponential_distribution<double> E(1.0);
  array<double, 4> l; double s = 0;
  for (auto& v : l) { v = E(g); s += v; }
  for (auto& v : l) v /= s;
  return l;
}

// ------------------------------------------------------------------ modes
static int mode_gradcheck() {
  MLP net({5, 7, 6, 1}, 3);
  double x[5] = {0.3, -1.2, 0.5, 2.0, -0.7};
  vector<vector<double>> acts, gW, gb;
  net.zero_like(gW, gb);
  net.forward(x, &acts);
  net.backward(acts, 1.0, gW, gb);
  double maxerr = 0;
  for (size_t l = 0; l < net.W.size(); ++l)
    for (size_t i = 0; i < net.W[l].size(); i += 3) {
      double w = net.W[l][i], h = 1e-6;
      net.W[l][i] = w + h; double fp = net.forward(x);
      net.W[l][i] = w - h; double fm = net.forward(x);
      net.W[l][i] = w;
      maxerr = max(maxerr, fabs((fp - fm) / (2 * h) - gW[l][i]));
    }
  printf("gradient check: max |numerical - analytical| = %.3e => %s\n", maxerr, maxerr < 1e-6 ? "OK" : "FAIL");
  return maxerr < 1e-6 ? 0 : 1;
}

static int mode_pool(int n, int T, unsigned seed0, const string& out) {
  FILE* f = fopen(out.c_str(), "w");
  if (!f) { perror("pool"); return 1; }
  int done = 0;
  for (unsigned s = seed0; done < n; ++s) {
    Instance in; string kind;
    if (!gen_instance(s, T, in, &kind)) continue;
    auto t0 = chrono::steady_clock::now();
    Prepared P = prepare(in, true);
    double tt = chrono::duration<double>(chrono::steady_clock::now() - t0).count();
    write_item(f, s, kind, P);
    fflush(f);
    ++done;
    printf("[%d/%d] seed %u %-12s mu=%.1f r=%ld classes=%zu front=%zu  %.2fs\n", done, n, s, kind.c_str(), in.mu,
           in.r, P.classes.size(), P.front.size(), tt);
    fflush(stdout);
  }
  fclose(f);
  return 0;
}

static int mode_train(const string& poolf, int iters, int batch, int K, const string& out, double lr,
                      unsigned seed) {
  auto pool = read_pool(poolf);
  if (pool.empty()) { fprintf(stderr, "empty pool\n"); return 1; }
  printf("pool: %zu instances; preparing classes...\n", pool.size());
  vector<Prepared> PP(pool.size());
#pragma omp parallel for schedule(dynamic, 1)
  for (long i = 0; i < (long)pool.size(); ++i) PP[i] = prep_from_pool(pool[i]);
  Policy pi(seed);
  if (!out.empty()) { pi.cls.load(out + ".cls"); pi.ev.load(out + ".ev"); }   // resume if present
  Adam oc(pi.cls, lr), oe(pi.ev, lr);
  mt19937_64 rg(seed);
  FILE* lg = fopen((out + ".log").c_str(), "a");
  auto t0 = chrono::steady_clock::now();
  double ema = 0;
  for (int it = 1; it <= iters; ++it) {
    vector<pair<int, array<double, 4>>> jobs(batch);
    for (auto& j : jobs) { j.first = (int)(rg() % PP.size()); j.second = random_pref(rg); }
    vector<unsigned long long> seeds(batch);
    for (auto& s : seeds) s = rg();
    int nth = 1;
#ifdef _OPENMP
    nth = omp_get_max_threads();
#endif
    vector<Grad> TG(nth);
    for (auto& G : TG) { pi.cls.zero_like(G.cW, G.cb); pi.ev.zero_like(G.eW, G.eb); }
    vector<double> meanR(batch);
#pragma omp parallel for schedule(dynamic, 1)
    for (int bi = 0; bi < batch; ++bi) {
      int tid = 0;
#ifdef _OPENMP
      tid = omp_get_thread_num();
#endif
      const Prepared& P = PP[jobs[bi].first];
      const auto& lam = jobs[bi].second;
      mt19937_64 g(seeds[bi]);
      vector<vector<Decision>> tr(K);
      vector<double> R(K);
      for (int k = 0; k < K; ++k) R[k] = -tcheby(normalise(P, rollout(P, pi, lam, true, g, &tr[k])), lam);
      double m = accumulate(R.begin(), R.end(), 0.0) / K, sd = 0;
      for (double r : R) sd += (r - m) * (r - m);
      sd = sqrt(sd / K) + 1e-8;
      meanR[bi] = m;
      for (int k = 0; k < K; ++k) {
        double adv = (R[k] - m) / sd;
        // loss = -adv * log pi  ->  gradient of loss = -adv * grad log pi
        accumulate_grad(pi, tr[k], -adv / (batch * K), TG[tid]);
      }
    }
    Grad G = TG[0];
    for (int t = 1; t < nth; ++t)
      for (size_t l = 0; l < G.cW.size(); ++l) {
        for (size_t i = 0; i < G.cW[l].size(); ++i) G.cW[l][i] += TG[t].cW[l][i];
        for (size_t i = 0; i < G.cb[l].size(); ++i) G.cb[l][i] += TG[t].cb[l][i];
      }
    for (int t = 1; t < nth; ++t)
      for (size_t l = 0; l < G.eW.size(); ++l) {
        for (size_t i = 0; i < G.eW[l].size(); ++i) G.eW[l][i] += TG[t].eW[l][i];
        for (size_t i = 0; i < G.eb[l].size(); ++i) G.eb[l][i] += TG[t].eb[l][i];
      }
    oc.step(pi.cls, G.cW, G.cb);
    oe.step(pi.ev, G.eW, G.eb);
    double mr = accumulate(meanR.begin(), meanR.end(), 0.0) / batch;
    ema = (it == 1) ? mr : 0.98 * ema + 0.02 * mr;
    if (it % 50 == 0 || it == iters) {
      double el = chrono::duration<double>(chrono::steady_clock::now() - t0).count();
      printf("iter %6d  mean Tchebycheff %.4f (ema %.4f)  %.0fs\n", it, -mr, -ema, el);
      fflush(stdout);
      if (lg) { fprintf(lg, "%d %.6f %.6f %.1f\n", it, -mr, -ema, el); fflush(lg); }
    }
    if (it % 500 == 0 || it == iters) {
      pi.cls.save(out + ".cls"); pi.ev.save(out + ".ev");
      // numbered checkpoints, so that runs can also be compared at equal wall-clock time
      string ck = out + ".it" + to_string(it);
      pi.cls.save(ck + ".cls"); pi.ev.save(ck + ".ev");
    }
  }
  if (lg) fclose(lg);
  return 0;
}

// evaluation against exact fronts: per-preference regret and share on the front
struct EvalRes { vector<double> reg_pol, reg_best, reg_greedy; int on_pol = 0, on_best = 0, on_greedy = 0, n = 0;
                 double t_pol = 0, t_exact = 0, t_dec = 0; };
static array<double, 4> greedy_baseline(const Prepared& P, const array<double, 4>& lam) {
  double range[4];
  for (int i = 0; i < 4; ++i) range[i] = P.nadir[i] - P.ideal[i];
  double bestg = 1e300; array<double, 4> bestF{};
  for (int g = 0; g < (int)P.classes.size(); ++g) {
    State s; s.cls = g;
    while (!s.done) {
      auto C = candidates(P, s);
      int bi = 0; double bv = 1e300;
      for (int i = 0; i < (int)C.size(); ++i) {
        int len = max(1, C[i].tn - max(1, s.tau));
        double v = (lam[0] * C[i].dPOC / range[0] + lam[1] * P.in.h * C[i].dW / range[1] - lam[3] * C[i].dITO / range[3]) / len;
        if (v < bv) { bv = v; bi = i; }
      }
      step(P, s, C[bi]);
    }
    auto F = outcome(P, s);
    double gv = tcheby(normalise(P, F), lam);
    if (gv < bestg) { bestg = gv; bestF = F; }
  }
  return bestF;
}
// evaluates one prepared instance (exact front already in P); lines appended to *det
static void eval_prepared(const Prepared& P, const Policy& pi, int H, int samples, EvalRes& E, vector<string>& det,
                          const string& name) {
  auto W = simplex_weights(H);
  mt19937_64 g(12345);
  char buf[512];
  for (auto& lam : W) {
    double opt = exact_opt(P, lam);
    auto t1 = chrono::steady_clock::now();
    auto Fd = rollout(P, pi, lam, false, g, nullptr);
    E.t_dec += chrono::duration<double>(chrono::steady_clock::now() - t1).count();
    double gbest = tcheby(normalise(P, Fd), lam); auto Fb = Fd;
    for (int s = 0; s < samples; ++s) {
      auto F = rollout(P, pi, lam, true, g, nullptr);
      double gv = tcheby(normalise(P, F), lam);
      if (gv < gbest) { gbest = gv; Fb = F; }
    }
    E.t_pol += chrono::duration<double>(chrono::steady_clock::now() - t1).count();
    auto Fg = greedy_baseline(P, lam);
    double rp = tcheby(normalise(P, Fd), lam) - opt, rb = gbest - opt, rg = tcheby(normalise(P, Fg), lam) - opt;
    E.reg_pol.push_back(rp); E.reg_best.push_back(rb); E.reg_greedy.push_back(rg);
    E.on_pol += on_front(P, Fd); E.on_best += on_front(P, Fb); E.on_greedy += on_front(P, Fg); E.n++;
    snprintf(buf, sizeof buf, "%s %.4f %.4f %.4f %.4f %.6f %.6f %.6f\n", name.c_str(), lam[0], lam[1], lam[2], lam[3], rp, rb, rg);
    det.push_back(buf);
  }
}
static void merge(EvalRes& A, const EvalRes& B) {
  A.reg_pol.insert(A.reg_pol.end(), B.reg_pol.begin(), B.reg_pol.end());
  A.reg_best.insert(A.reg_best.end(), B.reg_best.begin(), B.reg_best.end());
  A.reg_greedy.insert(A.reg_greedy.end(), B.reg_greedy.begin(), B.reg_greedy.end());
  A.on_pol += B.on_pol; A.on_best += B.on_best; A.on_greedy += B.on_greedy; A.n += B.n;
  A.t_pol += B.t_pol; A.t_exact += B.t_exact; A.t_dec += B.t_dec;
}
// exact front from a Paper 1 front CSV (SS,SOT,CSL,POC,HC,ITO,...); printed precision is far
// below the comparison tolerances of on_front
static bool load_front_csv(const string& fn, Prepared& P) {
  ifstream f(fn);
  if (!f) return false;
  string line; getline(f, line);
  P.front.clear();
  while (getline(f, line)) {
    double v[6]; char* p = line.data();
    for (int i = 0; i < 6; ++i) { v[i] = strtod(p, &p); if (*p == ',') ++p; }
    P.front.push_back({v[3], v[4], v[2], v[5], 0.0});
  }
  if (P.front.empty()) return false;
  for (int i = 0; i < 4; ++i) { P.ideal[i] = 1e300; P.nadir[i] = -1e300; }
  for (auto& q : P.front) {
    double z[4] = {q.POC, q.HC, -q.CSL, -q.ITO};
    for (int i = 0; i < 4; ++i) { P.ideal[i] = min(P.ideal[i], z[i]); P.nadir[i] = max(P.nadir[i], z[i]); }
  }
  for (int i = 0; i < 4; ++i) if (P.nadir[i] - P.ideal[i] < 1e-12) P.nadir[i] = P.ideal[i] + 1e-12;
  return true;
}
static double log_total(const string& fn) {   // "TOTAL x s" from a Paper 1 log
  ifstream f(fn); string line; double t = -1;
  while (getline(f, line)) { auto k = line.find("TOTAL "); if (k != string::npos) t = atof(line.c_str() + k + 6); }
  return t;
}
static vector<map<string, string>> read_csv(const string& fn) {
  vector<map<string, string>> R;
  ifstream f(fn); string line;
  if (!getline(f, line)) return R;
  // Python's csv module ends rows with \r\n: strip \r and surrounding blanks from every field
  auto split = [](string s) {
    while (!s.empty() && (s.back() == '\r' || s.back() == '\n')) s.pop_back();
    vector<string> v; string c; istringstream is(s);
    while (getline(is, c, ',')) {
      size_t a = c.find_first_not_of(" \t\""), b = c.find_last_not_of(" \t\"\r");
      v.push_back(a == string::npos ? "" : c.substr(a, b - a + 1));
    }
    return v;
  };
  auto H = split(line);
  while (getline(f, line)) {
    auto v = split(line);
    map<string, string> m;
    for (size_t i = 0; i < H.size() && i < v.size(); ++i) m[H[i]] = v[i];
    R.push_back(m);
  }
  return R;
}
static void report(const EvalRes& E, int samples) {
  auto stats = [](vector<double> v) {
    sort(v.begin(), v.end());
    double m = accumulate(v.begin(), v.end(), 0.0) / v.size();
    return array<double, 4>{v[v.size() / 2], m, v[(size_t)(0.9 * (v.size() - 1))], v.back()};
  };
  auto a = stats(E.reg_pol), b = stats(E.reg_best), c = stats(E.reg_greedy);
  printf("\n%d (instance, preference) pairs; regret = Tchebycheff gap to the exact optimum (normalised)\n", E.n);
  printf("%-28s %8s %8s %8s %8s %10s\n", "method", "median", "mean", "p90", "max", "on front");
  printf("%-28s %8.4f %8.4f %8.4f %8.4f %9.1f%%\n", "policy (greedy decoding)", a[0], a[1], a[2], a[3], 100.0 * E.on_pol / E.n);
  printf("%-28s %8.4f %8.4f %8.4f %8.4f %9.1f%%\n", ("policy (best of 1+" + to_string(samples) + ")").c_str(), b[0], b[1], b[2], b[3], 100.0 * E.on_best / E.n);
  printf("%-28s %8.4f %8.4f %8.4f %8.4f %9.1f%%\n", "greedy + class enumeration", c[0], c[1], c[2], c[3], 100.0 * E.on_greedy / E.n);
  printf("time per preference (one thread): greedy decoding %.2f ms; decoding + %d samples %.2f ms\n",
         1e3 * E.t_dec / E.n, samples, 1e3 * E.t_pol / E.n);
}

int main(int argc, char** argv) {
  string mode = argc > 1 ? argv[1] : "gradcheck";
  map<string, string> A;
  vector<string> positional;
  for (int i = 2; i < argc; ++i) {
    string a = argv[i];
    if (a.rfind("--", 0) == 0 && i + 1 < argc) A[a] = argv[++i];
    else positional.push_back(a);
  }
  auto gi = [&](const string& k, int d) { return A.count(k) ? stoi(A[k]) : d; };
  auto gd = [&](const string& k, double d) { return A.count(k) ? stod(A[k]) : d; };
  auto gs = [&](const string& k, string d) { return A.count(k) ? A[k] : d; };
  g_action = gs("--action", "event") == "quantity" ? 1 : 0;
  g_qlevels = gi("--qlevels", 24);
  if (mode == "gradcheck") return mode_gradcheck();
  if (mode == "qcheck") {   // ablation environment: every random episode is a feasible plan on or behind the front
    g_action = 1;
    auto pool = read_pool(gs("--pool", "pool_test.txt"));
    Policy pi; mt19937_64 g(7); long bad = 0, tot = 0, onf = 0;
    for (int i = 0; i < min(gi("--limit", 5), (int)pool.size()); ++i) {
      Prepared P = prepare(pool[i].in, true);
      for (int e = 0; e < 300; ++e) {
        auto F = rollout(P, pi, random_pref(g), true, g, nullptr); ++tot; onf += on_front(P, F);
        for (auto& q : P.front)   // a feasible plan can never strictly dominate a Pareto point
          if (F[0] < q.POC - 2e-4 && F[1] < q.HC - 2e-6 && F[2] > q.CSL + 1e-9 && F[3] > q.ITO + 2e-8) { ++bad; break; }
      }
    }
    printf("quantity-action env: %ld episodes, %ld dominate a Pareto point (must be 0), %ld on the front\n", tot, bad, onf);
    return bad ? 1 : 0;
  }
  if (mode == "pool") return mode_pool(gi("--n", 100), gi("--T", 48), gi("--seed0", 1), gs("--out", "pool.txt"));
  if (mode == "train")
    return mode_train(gs("--pool", "pool.txt"), gi("--iters", 1000), gi("--batch", 32), gi("--K", 8),
                      gs("--out", "model"), gd("--lr", 1e-3), gi("--seed", 1));
  if (mode == "eval" || mode == "evalreal") {
    Policy pi;
    string mf = gs("--model", "model");
    if (mf != "none" && (!pi.cls.load(mf + ".cls") || !pi.ev.load(mf + ".ev"))) { fprintf(stderr, "cannot load model %s\n", mf.c_str()); return 1; }
    int H = gi("--H", 6), S = gi("--samples", 16);
    // jobs: name + instance + (optional) front CSV and Paper 1 log
    struct Job { string name; Instance in; string front, log; };
    vector<Job> jobs;
    if (mode == "eval") {
      auto pool = read_pool(gs("--pool", "pool_test.txt"));
      int lim = min(gi("--limit", (int)pool.size()), (int)pool.size());
      for (int i = 0; i < lim; ++i) jobs.push_back({"pool" + to_string(i) + "_" + pool[i].kind, pool[i].in, "", ""});
    } else {
      // evalreal: E6 manifests; exact fronts are taken from the Paper 1 runs in <suite>/E6/<source>/
      string root = gs("--root", "../.."), suite = gs("--suite", root + "/suite");
      for (const string& a : positional) {
        for (auto& r : read_csv(a)) {
          for (const char* k : {"source", "series", "window", "T", "demand_file", "mu", "sigma", "p", "r", "moq", "L", "c", "MS", "hfrac"})
            if (r[k].empty()) { fprintf(stderr, "manifest %s: column '%s' missing or empty\n", a.c_str(), k); exit(3); }
          Overrides ov;
          string df = r["demand_file"];
          ov.demand_file = (df.size() && df[0] == '/') ? df : root + "/" + df;
          ov.mu = stod(r["mu"]); ov.sigma = stod(r["sigma"]); ov.p = stod(r["p"]); ov.r = stol(r["r"]);
          ov.moq = stol(r["moq"]); ov.L = stoi(r["L"]); ov.c = stod(r["c"]); ov.MS = stod(r["MS"]); ov.hfrac = stod(r["hfrac"]);
          string base = suite + "/E6/" + r["source"] + "/" + r["series"] + "_w" + r["window"];
          jobs.push_back({r["source"] + "/" + r["series"] + "_w" + r["window"], make_instance(1, stoi(r["T"]), 1, ov),
                          gi("--recompute", 0) ? "" : base + ".csv", base + ".log"});
        }
      }
      int lim = min(gi("--limit", (int)jobs.size()), (int)jobs.size());
      jobs.resize(lim);
    }
    const int n = (int)jobs.size();
    vector<EvalRes> ER(n);
    vector<vector<string>> DET(n);
    vector<double> fsz(n, 0), texact(n, -1);
    int missing = 0, done = 0;
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : missing)
    for (int i = 0; i < n; ++i) {
      Prepared P;
      auto t0 = chrono::steady_clock::now();
      if (!jobs[i].front.empty()) {
        P = prepare(jobs[i].in, false);
        if (!load_front_csv(jobs[i].front, P)) { ++missing; continue; }
        texact[i] = log_total(jobs[i].log);
      } else {
        P = prepare(jobs[i].in, true);
        texact[i] = chrono::duration<double>(chrono::steady_clock::now() - t0).count();
      }
      fsz[i] = (double)P.front.size();
      eval_prepared(P, pi, H, S, ER[i], DET[i], jobs[i].name);
#pragma omp critical
      { ++done; fprintf(stderr, "evaluated %d/%d\r", done, n); }
    }
    EvalRes E;
    FILE* out = fopen(gs("--out", "eval_detail.txt").c_str(), "w");
    vector<double> te;
    for (int i = 0; i < n; ++i) {
      merge(E, ER[i]);
      if (out) for (auto& s : DET[i]) fputs(s.c_str(), out);
      if (texact[i] >= 0 && !DET[i].empty()) te.push_back(texact[i]);
    }
    if (out) fclose(out);
    if (missing) printf("WARNING: %d instances without a front CSV were not evaluated\n", missing);
    report(E, S);
    if (!te.empty()) {
      sort(te.begin(), te.end());
      printf("exact front per instance (%s): median %.2f s, p90 %.2f s, max %.2f s over %zu instances\n",
             mode == "eval" ? "recomputed here, threads shared" : "Paper 1 runs", te[te.size() / 2],
             te[(size_t)(0.9 * (te.size() - 1))], te.back(), te.size());
    }
    return 0;
  }
  fprintf(stderr, "unknown mode\n");
  return 2;
}

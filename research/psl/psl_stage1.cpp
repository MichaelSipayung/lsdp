// psl_stage1.cpp - checks of the order-event environment and a greedy baseline.
//   psl_stage1 check  [--sku 1 --T 12 --seed 1]   environment reproduces the exact front
//   psl_stage1 greedy [--sku 1 --T 48 --seed 1 --H 6]  per-preference regret of greedy
#include "psl_core.hpp"

static void dfs(const Prepared& P, State s, vector<array<double, 4>>& all) {
  if (s.done) { all.push_back(outcome(P, s)); return; }
  for (auto& c : candidates(P, s)) { State t = s; step(P, t, c); dfs(P, t, all); }
}

static array<double, 4> greedy(const Prepared& P, const array<double, 4>& lam, int& best_cls) {
  double range[4];
  for (int i = 0; i < 4; ++i) range[i] = P.nadir[i] - P.ideal[i];
  double bestg = 1e300;
  array<double, 4> bestF{};
  for (int g = 0; g < (int)P.classes.size(); ++g) {
    State s; s.cls = g;
    while (!s.done) {
      auto C = candidates(P, s);
      int bi = 0; double bv = 1e300;
      for (int i = 0; i < (int)C.size(); ++i) {
        int len = max(1, C[i].tn - max(1, s.tau));
        double v = (lam[0] * C[i].dPOC / range[0] + lam[1] * P.in.h * C[i].dW / range[1] -
                    lam[3] * C[i].dITO / range[3]) / len;
        if (v < bv) { bv = v; bi = i; }
      }
      step(P, s, C[bi]);
    }
    auto F = outcome(P, s);
    double gv = tcheby(normalise(P, F), lam);
    if (gv < bestg) { bestg = gv; bestF = F; best_cls = g; }
  }
  return bestF;
}

int main(int argc, char** argv) {
  string mode = argc > 1 ? argv[1] : "check";
  int sku = 1, T = mode == "check" ? 12 : 48, H = 6; unsigned seed = 1;
  for (int i = 2; i + 1 < argc; i += 2) {
    string a = argv[i], v = argv[i + 1];
    if (a == "--sku") sku = stoi(v); else if (a == "--T") T = stoi(v);
    else if (a == "--seed") seed = stoul(v); else if (a == "--H") H = stoi(v);
  }
  Instance in = make_instance(sku, T, seed);
  auto t0 = chrono::steady_clock::now();
  Prepared P = prepare(in, true);
  double tprep = chrono::duration<double>(chrono::steady_clock::now() - t0).count();
  printf("instance SKU%d T=%d seed=%u: %zu classes, exact front %zu points (%.3f s)\n", sku, T, seed,
         P.classes.size(), P.front.size(), tprep);
  if (mode == "check") {
    vector<array<double, 4>> all;
    for (int g = 0; g < (int)P.classes.size(); ++g) { State s; s.cls = g; dfs(P, s, all); }
    // non-dominated filter (minimise POC, HC; maximise CSL, ITO)
    vector<array<double, 4>> nd;
    for (size_t i = 0; i < all.size(); ++i) {
      bool d = false;
      for (size_t j = 0; j < all.size() && !d; ++j) {
        if (i == j) continue;
        auto &a = all[j], &q = all[i];
        bool w = a[0] <= q[0] + 1e-7 && a[1] <= q[1] + 1e-7 && a[2] >= q[2] - 1e-12 && a[3] >= q[3] - 1e-9;
        bool st = a[0] < q[0] - 1e-7 || a[1] < q[1] - 1e-7 || a[2] > q[2] + 1e-12 || a[3] > q[3] + 1e-9;
        if (w && (st || j < i)) d = true;
      }
      if (!d) nd.push_back(all[i]);
    }
    size_t match = 0;
    for (auto& F : nd) match += on_front(P, F);
    printf("environment: %zu complete plans enumerated, %zu non-dominated, %zu of them on the exact front "
           "(exact front %zu) => %s\n", all.size(), nd.size(), match, P.front.size(),
           (match == nd.size() && nd.size() == P.front.size()) ? "OK" : "MISMATCH");
    return 0;
  }
  auto W = simplex_weights(H);
  vector<double> reg; int onf = 0;
  auto t1 = chrono::steady_clock::now();
  for (auto& lam : W) {
    int cls = -1;
    auto F = greedy(P, lam, cls);
    double r = tcheby(normalise(P, F), lam) - exact_opt(P, lam);
    reg.push_back(r); onf += on_front(P, F);
  }
  double tg = chrono::duration<double>(chrono::steady_clock::now() - t1).count();
  sort(reg.begin(), reg.end());
  printf("greedy + class enumeration over %zu preferences: regret median %.4f, mean %.4f, max %.4f; "
         "on exact front %d/%zu; %.3f s per preference\n", W.size(), reg[reg.size() / 2],
         accumulate(reg.begin(), reg.end(), 0.0) / reg.size(), reg.back(), onf, W.size(), tg / W.size());
}

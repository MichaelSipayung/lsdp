// psl_nn.hpp - small multilayer perceptron with manual backpropagation and Adam (pure C++).
#pragma once
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

struct MLP {
  // layers: sizes[0] = input, sizes.back() = 1 (score); tanh hidden activations
  std::vector<int> sizes;
  std::vector<std::vector<double>> W, b;   // W[l] is (sizes[l+1] x sizes[l]), row-major
  explicit MLP(std::vector<int> s = {}, unsigned seed = 1) : sizes(s) {
    std::mt19937_64 g(seed);
    for (size_t l = 0; l + 1 < sizes.size(); ++l) {
      double sd = std::sqrt(1.0 / sizes[l]);
      std::normal_distribution<double> nd(0.0, sd);
      W.emplace_back(sizes[l + 1] * sizes[l]);
      b.emplace_back(sizes[l + 1], 0.0);
      for (auto& w : W.back()) w = nd(g);
    }
  }
  size_t nparams() const {
    size_t n = 0;
    for (size_t l = 0; l < W.size(); ++l) n += W[l].size() + b[l].size();
    return n;
  }
  // forward pass keeping activations (acts[0] = input)
  double forward(const double* x, std::vector<std::vector<double>>* acts = nullptr) const {
    std::vector<double> a(x, x + sizes[0]), z;
    if (acts) { acts->clear(); acts->push_back(a); }
    for (size_t l = 0; l < W.size(); ++l) {
      const int n = sizes[l + 1], m = sizes[l];
      z.assign(n, 0.0);
      for (int i = 0; i < n; ++i) {
        double s = b[l][i];
        const double* w = &W[l][(size_t)i * m];
        for (int j = 0; j < m; ++j) s += w[j] * a[j];
        z[i] = (l + 1 < W.size()) ? std::tanh(s) : s;
      }
      a.swap(z);
      if (acts) acts->push_back(a);
    }
    return a[0];
  }
  // accumulate d(score)/d(params) * coef into grad (same layout as W, b)
  void backward(const std::vector<std::vector<double>>& acts, double coef,
                std::vector<std::vector<double>>& gW, std::vector<std::vector<double>>& gb) const {
    std::vector<double> delta(1, coef), prev;
    for (int l = (int)W.size() - 1; l >= 0; --l) {
      const int n = sizes[l + 1], m = sizes[l];
      const auto& ain = acts[l];
      for (int i = 0; i < n; ++i) {
        gb[l][i] += delta[i];
        double* gw = &gW[l][(size_t)i * m];
        for (int j = 0; j < m; ++j) gw[j] += delta[i] * ain[j];
      }
      if (l == 0) break;
      prev.assign(m, 0.0);
      for (int i = 0; i < n; ++i) {
        const double* w = &W[l][(size_t)i * m];
        for (int j = 0; j < m; ++j) prev[j] += w[j] * delta[i];
      }
      for (int j = 0; j < m; ++j) prev[j] *= (1.0 - ain[j] * ain[j]);  // tanh'
      delta.swap(prev);
    }
  }
  void zero_like(std::vector<std::vector<double>>& gW, std::vector<std::vector<double>>& gb) const {
    gW.resize(W.size()); gb.resize(b.size());
    for (size_t l = 0; l < W.size(); ++l) { gW[l].assign(W[l].size(), 0.0); gb[l].assign(b[l].size(), 0.0); }
  }
  bool save(const std::string& f) const {
    FILE* fp = fopen(f.c_str(), "wb");
    if (!fp) return false;
    int L = (int)sizes.size(); fwrite(&L, sizeof L, 1, fp); fwrite(sizes.data(), sizeof(int), L, fp);
    for (size_t l = 0; l < W.size(); ++l) {
      fwrite(W[l].data(), sizeof(double), W[l].size(), fp); fwrite(b[l].data(), sizeof(double), b[l].size(), fp);
    }
    fclose(fp); return true;
  }
  bool load(const std::string& f) {
    FILE* fp = fopen(f.c_str(), "rb");
    if (!fp) return false;
    int L = 0; if (fread(&L, sizeof L, 1, fp) != 1) { fclose(fp); return false; }
    std::vector<int> s(L); if (fread(s.data(), sizeof(int), L, fp) != (size_t)L) { fclose(fp); return false; }
    *this = MLP(s, 1);
    for (size_t l = 0; l < W.size(); ++l) {
      if (fread(W[l].data(), sizeof(double), W[l].size(), fp) != W[l].size() ||
          fread(b[l].data(), sizeof(double), b[l].size(), fp) != b[l].size()) { fclose(fp); return false; }
    }
    fclose(fp); return true;
  }
};

struct Adam {
  double lr, b1 = 0.9, b2 = 0.999, eps = 1e-8;
  long t = 0;
  std::vector<std::vector<double>> mW, vW, mb, vb;
  explicit Adam(const MLP& net, double lr_ = 1e-3) : lr(lr_) {
    net.zero_like(mW, mb); net.zero_like(vW, vb);
  }
  // gradient DESCENT step on loss with gradient (gW, gb)
  void step(MLP& net, const std::vector<std::vector<double>>& gW, const std::vector<std::vector<double>>& gb,
            double clip = 1.0) {
    double nrm = 0.0;
    for (size_t l = 0; l < gW.size(); ++l) {
      for (double g : gW[l]) nrm += g * g;
      for (double g : gb[l]) nrm += g * g;
    }
    nrm = std::sqrt(nrm);
    const double sc = (nrm > clip) ? clip / nrm : 1.0;
    ++t;
    const double c1 = 1.0 - std::pow(b1, (double)t), c2 = 1.0 - std::pow(b2, (double)t);
    auto upd = [&](std::vector<double>& p, const std::vector<double>& g, std::vector<double>& m, std::vector<double>& v) {
      for (size_t i = 0; i < p.size(); ++i) {
        double gi = g[i] * sc;
        m[i] = b1 * m[i] + (1 - b1) * gi;
        v[i] = b2 * v[i] + (1 - b2) * gi * gi;
        p[i] -= lr * (m[i] / c1) / (std::sqrt(v[i] / c2) + eps);
      }
    };
    for (size_t l = 0; l < net.W.size(); ++l) { upd(net.W[l], gW[l], mW[l], vW[l]); upd(net.b[l], gb[l], mb[l], vb[l]); }
  }
};

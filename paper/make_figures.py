#!/usr/bin/env python3
"""make_figures.py - build every figure and table of the manuscript from the suite results.

usage (from the repository root, inside the virtualenv):
    python3 paper/make_figures.py --suite suite --out paper

writes
    paper/figures/fig_front.pdf      projection of an exact front (SKU1 seed 1, continuous)
    paper/figures/fig_scaling.pdf    one inner DP: time and labels against horizon T (E3)
    paper/figures/fig_milp.pdf       DP versus epsilon-constraint MILP (E5a)
    paper/figures/fig_nsga.pdf       NSGA-III sets against the exact front (E5b)
    paper/tables/tab_e2.tex, tab_e3fit.tex, tab_e4.tex, tab_e5a.tex, tab_e5b.tex
    paper/tables/software.tex        versions of every tool used (for the manuscript)
Every number in the manuscript tables comes from these files, so they can be regenerated.
"""
import argparse
import csv
import glob
import os
import platform
import re
import statistics as st
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import numpy as np  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--suite", default="suite")
ap.add_argument("--out", default="paper")
ap.add_argument("--bench", default="bench/instances")
a = ap.parse_args()
FIG = os.path.join(a.out, "figures")
TAB = os.path.join(a.out, "tables")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.linewidth": 0.6,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "savefig.dpi": 300, "figure.dpi": 100})
C1, C2 = "#1f4e79", "#c55a11"  # SKU1, SKU2


def read(path):
    return open(path).read() if os.path.exists(path) else ""


def num(pat, s, cast=float):
    m = re.search(pat, s)
    return cast(m.group(1)) if m else float("nan")


def log_stats(path):
    s = read(path)
    return {"time": num(r"TOTAL ([\d.]+)s", s), "labels": num(r"labels generated \(total\): (\d+)", s),
            "front": num(r"Pareto front: (\d+)", s), "keys": num(r", (\d+) distinct requirement vectors", s),
            "cands": num(r"-> (\d+) (?:combinations|candidates)", s),
            "timeout": num(r"TIMEOUT (\d+)s", s),
            "solvedcls": num(r"(\d+) non-dominated classes solved", s)}


def ms(v, fmt="{:.2f}"):
    v = [x for x in v if x == x]
    if not v:
        return "--"
    if len(v) == 1:
        return fmt.format(v[0])
    return (fmt + " $\\pm$ " + fmt).format(st.mean(v), st.stdev(v))


def thousands(v):
    v = [x for x in v if x == x]
    if not v:
        return "--"
    if len(v) == 1:
        return f"{v[0]/1000:.1f}k"
    return f"{st.mean(v)/1000:.1f}k $\\pm$ {st.stdev(v)/1000:.1f}k"


def write(name, text):
    with open(os.path.join(TAB, name), "w") as f:
        f.write(text)
    print("wrote", os.path.join(TAB, name))


def pareto_mask(F):
    """non-dominated mask for minimisation, F (n, m); used only for small sets."""
    n = len(F)
    keep = np.ones(n, bool)
    for i in range(n):
        if keep[i]:
            dom = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
            if dom.any():
                keep[i] = False
    return keep


# ---------------------------------------------------------------- E2 table
rows = []
for sku in (1, 2):
    for outer in ("grid", "continuous"):
        L = [log_stats(f) for f in sorted(glob.glob(f"{a.suite}/E2/sku{sku}_s*_{outer}.log"))]
        if not L:
            continue
        rows.append(f"SKU {sku} & {outer} & {len(L)} & {ms([d['cands'] for d in L], '{:.0f}')} & "
                    f"{ms([d['keys'] for d in L], '{:.0f}')} & {ms([d['solvedcls'] for d in L], '{:.0f}')} & "
                    f"{thousands([d['front'] for d in L])} & "
                    f"{ms([d['time'] for d in L])} \\\\")
# speed-up against the independent reference run (same machine and threads)
sp = []
for f in glob.glob(f"{a.suite}/E2/sku*_s*_*.log"):
    g = f"{a.suite}/ref/E2/" + os.path.basename(f)
    t1, t0 = log_stats(f)["time"], log_stats(g)["time"]
    if t1 == t1 and t0 == t0 and t1 > 0:
        sp.append(t0 / t1)
if sp:
    write("e2_macros.tex", f"\\newcommand{{\\EtwoSpeedLo}}{{{min(sp):.1f}}}\n"
          f"\\newcommand{{\\EtwoSpeedHi}}{{{max(sp):.1f}}}\n"
          f"\\newcommand{{\\EtwoSpeedMed}}{{{np.median(sp):.1f}}}\n")
write("tab_e2.tex", "\\begin{tabular}{@{}llrrrrrr@{}}\n\\toprule\n"
      "Item & Outer domain & $n$ & Candidates & Classes $\\kappa$ & Solved & Front size & Time (s) \\\\\n\\midrule\n"
      + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# ---------------------------------------------------------------- Fig. front projection
fcsv = f"{a.suite}/E2/sku1_s1_continuous.csv"
if os.path.exists(fcsv):
    R = list(csv.DictReader(open(fcsv)))
    P = np.array([[float(r["POC"]), float(r["HC"]), float(r["CSL"]), float(r["ITO"])] for r in R])
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.8), constrained_layout=True)
    o = np.argsort(P[:, 2])
    s0 = ax[0].scatter(P[o, 0] / 1e3, P[o, 1], c=P[o, 2], s=0.4, cmap="viridis", rasterized=True)
    ax[0].set_xlabel("POC (thousand)")
    ax[0].set_ylabel("HC")
    o = np.argsort(P[:, 3])
    s1 = ax[1].scatter(P[o, 1], P[o, 3], c=P[o, 2], s=0.4, cmap="viridis", rasterized=True)
    ax[1].set_xlabel("HC")
    ax[1].set_ylabel("ITO")
    fig.colorbar(s1, ax=ax, label="CSL", shrink=0.9)
    for x, t in zip(ax, "ab"):
        x.set_title(f"({t})", loc="left", fontsize=9)
    fig.savefig(os.path.join(FIG, "fig_front.pdf"))
    plt.close(fig)
    print("wrote fig_front.pdf with", len(P), "points")

# ---------------------------------------------------------------- E3 scaling + fit table
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6), constrained_layout=True)
fitrows = []
allT = set()
for sku, col in ((1, C1), (2, C2)):
    Ts, tm, lb = [], [], []
    for T in range(12, 200, 12):
        L = [log_stats(f) for f in glob.glob(f"{a.suite}/E3/one_sku{sku}_T{T}_s*.log")]
        L = [d for d in L if d["time"] == d["time"]]
        if L:
            Ts.append(T)
            tm.append(st.mean(d["time"] for d in L))
            lb.append(st.mean(d["labels"] for d in L))
    if len(Ts) < 3:
        continue
    Ts, tm, lb = map(np.array, (Ts, tm, lb))
    allT.update(Ts.tolist())
    big = Ts >= 36  # asymptotic regime: exclude the smallest horizons from the fit
    et = np.polyfit(np.log(Ts[big]), np.log(tm[big]), 1)[0]
    el = np.polyfit(np.log(Ts[big]), np.log(lb[big]), 1)[0]
    etl = np.polyfit(np.log(lb[big]), np.log(tm[big]), 1)[0]
    ax[0].loglog(Ts, tm, "o-", color=col, ms=3, lw=1, label=f"SKU {sku} (slope {et:.2f})")
    ax[1].loglog(Ts, lb, "s-", color=col, ms=3, lw=1, label=f"SKU {sku} (slope {el:.2f})")
    fitrows.append(f"SKU {sku} & {Ts[big].min()}--{Ts.max()} & {et:.2f} & {el:.2f} & {etl:.2f} \\\\")
ax[0].set_xlabel("horizon $T$")
ax[0].set_ylabel("time of one inner DP (s)")
ax[1].set_xlabel("horizon $T$")
ax[1].set_ylabel("labels generated")
for x, t in zip(ax, "ab"):
    if allT:
        x.set_xticks(sorted(allT))
        x.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        x.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    x.legend(frameon=False, fontsize=8)
    x.set_title(f"({t})", loc="left", fontsize=9)
fig.savefig(os.path.join(FIG, "fig_scaling.pdf"))
plt.close(fig)
write("tab_e3fit.tex", "\\begin{tabular}{@{}lrrrr@{}}\n\\toprule\n"
      "Item & $T$ range of fit & time vs $T$ & labels vs $T$ & time vs labels \\\\\n\\midrule\n"
      + "\n".join(fitrows) + "\n\\bottomrule\n\\end{tabular}\n")
full = log_stats(f"{a.suite}/E3/full_sku1_T96_s1.log")
write("e3_full.tex", f"\\newcommand{{\\EfullTime}}{{{full['time']/60:.1f}}}\n"
      f"\\newcommand{{\\EfullFront}}{{{full['front']:,.0f}}}\n".replace(",", "\\,"))

# ---------------------------------------------------------------- E4 table
rows = []
for tag, label in (("cv0.1", "$CV=0.10$"), ("cv0.25", "$CV=0.25$"), ("cv0.5", "$CV=0.50$"),
                   ("r35", "$r=35$"), ("r70", "$r=70$"), ("r140", "$r=140$"),
                   ("c50", "$c=50$"), ("c200", "$c=200$"), ("c800", "$c=800$")):
    L = [log_stats(f) for f in sorted(glob.glob(f"{a.suite}/E4/{tag}_s*.log"))]
    if L:
        rows.append(f"{label} & {len(L)} & {ms([d['keys'] for d in L], '{:.0f}')} & "
                    f"{thousands([d['front'] for d in L])} & {ms([d['time'] for d in L])} \\\\")
# decision-set invariance in c (Proposition on ordering cost). The front file keeps one
# representative plan per objective vector, and which of several equivalent plans is kept
# can depend on the thread schedule. We therefore compare the c-independent signature of
# each point: (total quantity X(T'), number of orders k, HC, CSL, ITO).
def signature(f):
    S = set()
    for r in csv.DictReader(open(f)):
        q = [int(float(r[k])) for k in r if k.startswith("Q")]
        S.add((sum(q), sum(1 for x in q if x > 0), round(float(r["HC"]), 6),
               round(float(r["CSL"]), 10), round(float(r["ITO"]), 8)))
    return S


inv, inv_raw, detail = [], [], []
for s in range(1, 6):
    files = [f"{a.suite}/E4/c{c}_s{s}.csv" for c in (50, 200, 800)]
    if all(os.path.exists(f) for f in files):
        sig = [signature(f) for f in files]
        raw = [{tuple(r[k] for k in r if k in ("SS", "SOT") or k.startswith("Q"))
                for r in csv.DictReader(open(f))} for f in files]
        inv.append(sig[0] == sig[1] == sig[2])
        inv_raw.append(raw[0] == raw[1] == raw[2])
        detail.append(f"seed {s}: signatures 50=200 {sig[0] == sig[1]}, 200=800 {sig[1] == sig[2]} | "
                      f"raw plans 50=200 {raw[0] == raw[1]}, 200=800 {raw[1] == raw[2]} | "
                      f"|50-200| {len(sig[0] ^ sig[1])}, |200-800| {len(sig[1] ^ sig[2])}")
print("E4 ordering-cost invariance:\n  " + "\n  ".join(detail))
write("tab_e4.tex", "\\begin{tabular}{@{}lrrrr@{}}\n\\toprule\n"
      "Setting & $n$ & Classes $\\kappa$ & Front size & Time (s) \\\\\n\\midrule\n"
      + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
write("e4_inv.tex", f"\\newcommand{{\\EinvOK}}{{{sum(inv)}}}\\newcommand{{\\EinvN}}{{{len(inv)}}}\n")

# ---------------------------------------------------------------- E5a DP vs MILP
rows, pts = [], {1: [], 2: []}
for sku in (1, 2):
    for T in (16, 24, 32, 40, 48):
        dp = log_stats(f"{a.suite}/E5/dpnoito_sku{sku}_T{T}.log")["time"]
        ml = read(f"{a.suite}/E5/milp_sku{sku}_T{T}.log")
        wall = num(r"wall ([\d.]+)s", ml)
        nsol = num(r": (\d+) MILP solves", ml)
        same = "--"
        fa, fb = f"{a.suite}/E5/dpnoito_sku{sku}_T{T}.csv", f"{a.suite}/E5/milp_sku{sku}_T{T}.csv"
        if os.path.exists(fa) and os.path.exists(fb):
            A = sorted((round(float(r["POC"]), 3), round(float(r["HC"]), 3)) for r in csv.DictReader(open(fa)))
            B = sorted((round(float(r["POC"]), 3), round(float(r["HC"]), 3)) for r in csv.DictReader(open(fb)))
            same = f"yes ({len(A)})" if A == B else f"NO ({len(A)}/{len(B)})"
        if dp == dp:
            pts[sku].append((T, dp, wall))
            ratio = f"{wall/dp:,.0f}".replace(",", "\\,") if wall == wall and dp > 0 else "--"
            rows.append(f"SKU {sku} & {T} & {dp*1e3:.3f} & "
                        f"{('%.1f' % wall) if wall == wall else 'timeout'} & "
                        f"{('%.0f' % nsol) if nsol == nsol else '--'} & {ratio} & {same} \\\\")
# macros for the text: DP time and ratio at the largest horizon
def sci(x):
    e = int(np.floor(np.log10(x)))
    return f"{x / 10**e:.1f}\\times10^{{{e}}}"
r48 = [(sku, T, dp, ml) for sku in (1, 2) for (T, dp, ml) in pts[sku] if T == 48 and ml == ml and dp > 0]
if r48:
    ratios = [ml / dp for (_, _, dp, ml) in r48]
    write("e5a_macros.tex",
          f"\\newcommand{{\\EfiveDPone}}{{{r48[0][2]*1e3:.1f}}}\n"
          f"\\newcommand{{\\EfiveDPtwo}}{{{r48[-1][2]*1e3:.1f}}}\n"
          f"\\newcommand{{\\EfiveMILPone}}{{{r48[0][3]:,.0f}}}\n".replace(",", "\\,") +
          f"\\newcommand{{\\EfiveMILPtwo}}{{{r48[-1][3]:,.0f}}}\n".replace(",", "\\,") +
          f"\\newcommand{{\\EfiveRatioLo}}{{{sci(min(ratios))}}}\n"
          f"\\newcommand{{\\EfiveRatioHi}}{{{sci(max(ratios))}}}\n"
          f"\\newcommand{{\\EfiveOrders}}{{{['zero','one','two','three','four','five','six','seven','eight','nine'][int(np.floor(np.log10(min(ratios))))]}}}\n")
write("tab_e5a.tex", "\\begin{tabular}{@{}lrrrrrl@{}}\n\\toprule\n"
      "Item & $T$ & DP (ms) & MILP (s) & MILP solves & Ratio & Identical front (points) \\\\\n\\midrule\n"
      + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
fig, ax = plt.subplots(figsize=(3.4, 2.6))
for sku, col in ((1, C1), (2, C2)):
    if pts[sku]:
        T, dp, ml = map(np.array, zip(*pts[sku]))
        ax.semilogy(T, dp, "o-", color=col, ms=3, lw=1, label=f"DP, SKU {sku}")
        ax.semilogy(T, ml, "s--", color=col, ms=3, lw=1, label=f"MILP, SKU {sku}")
ax.set_xlabel("horizon $T$")
ax.set_ylabel("wall-clock time (s)")
ax.legend(frameon=False, fontsize=7)
fig.savefig(os.path.join(FIG, "fig_milp.pdf"))
plt.close(fig)

# ---------------------------------------------------------------- E5b NSGA-III
txt = read(f"{a.suite}/E5/E5b_compare.txt")
blocks = re.split(r"^== ", txt, flags=re.M)[1:]
agg = {}
for b in blocks:
    h = re.match(r"repair=(\d) SKU(\d) seed(\d+) vs exact (\w+)", b)
    if not h:
        continue
    key = (int(h[1]), int(h[2]), h[4])
    d = {"hits": num(r"exact front\s*:\s*\d+ points \| approximate set: (\d+)", b),
         "on": num(r"lying on exact front\s*:\s*\d+ \(([\d.]+)%\)", b),
         "dom": num(r"dominated by exact\s*:\s*\d+ \(([\d.]+)%\)", b),
         "domP": num(r"exact points dominated by approx\s*:\s*(\d+)", b),
         "rec": num(r"exact points recovered\s*:\s*([\d.]+)%", b),
         "igd": num(r"IGD\+ \(normalised\)\s*:\s*([\d.]+)", b),
         "eps": num(r"eps\+ \(normalised\)\s*:\s*([\d.]+)", b)}
    agg.setdefault(key, []).append(d)
rows = []
for (rep, sku, ref), L in sorted(agg.items()):
    rows.append(f"{'with' if rep else 'without'} & SKU {sku} & {ref} & {len(L)} & "
                f"{ms([d['hits'] for d in L], '{:.0f}')} & {ms([d['on'] for d in L])} & "
                f"{ms([d['dom'] for d in L], '{:.1f}')} & {ms([d['rec'] for d in L], '{:.3f}')} & "
                f"{ms([d['igd'] for d in L], '{:.4f}')} & {ms([d['eps'] for d in L], '{:.3f}')} & "
                f"{int(sum(d['domP'] for d in L))} \\\\")
write("tab_e5b.tex", "\\begin{tabular}{@{}lllrrrrrrrr@{}}\n\\toprule\n"
      "Repair & Item & Reference & $n$ & $|A|$ & On front (\\%) & Dominated (\\%) & "
      "Recovered (\\%) & IGD$^+$ & $\\varepsilon^+$ & Exact dom. \\\\\n\\midrule\n"
      + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
# figure: one projection, exact front versus NSGA-III without and with repair
ex = f"{a.suite}/E2/sku1_s1_grid.csv"
if os.path.exists(ex):
    E = np.array([[float(r["POC"]), float(r["HC"])] for r in csv.DictReader(open(ex))])
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    E2d = np.unique(E, axis=0)
    # 2-D non-dominated subset of the 4-D front, for readability
    o = np.lexsort((E2d[:, 1], E2d[:, 0]))
    E2d = E2d[o]
    best, nd = np.inf, []
    for p in E2d:
        if p[1] < best:
            nd.append(p)
            best = p[1]
    nd = np.array(nd)
    ax.plot(nd[:, 0] / 1e3, nd[:, 1], "-", color="k", lw=1, label="exact (POC, HC) frontier")
    for rep, col, mk in ((0, "#999999", "x"), (1, C2, "+")):
        f = f"{a.suite}/E5/nsga_rep{rep}_sku1_s1.csv"
        if os.path.exists(f):
            A = np.array([[float(r["POC"]), float(r["HC"])] for r in csv.DictReader(open(f))])
            ax.scatter(A[:, 0] / 1e3, A[:, 1], s=4, marker=mk, color=col, lw=0.6,
                       label=f"NSGA-III {'with' if rep else 'without'} repair")
    ax.set_xlabel("POC (thousand)")
    ax.set_ylabel("HC")
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(os.path.join(FIG, "fig_nsga.pdf"))
    plt.close(fig)

# ---------------------------------------------------------------- E6 public benchmark data
CLS = ("smooth", "erratic", "intermittent", "lumpy")


def q(v, f="{:.1f}"):
    v = np.array([x for x in v if x == x])
    if not len(v):
        return "--"
    return (f + " [" + f + ", " + f + "]").format(np.median(v), *np.percentile(v, [25, 75]))


man = []
for m in sorted(glob.glob(os.path.join(a.bench, "*", "manifest.csv"))):
    man += list(csv.DictReader(open(m)))
if man:
    rows, tim, chk, allst = [], {}, [0, 0], []
    SHOW_TO = any("TIMEOUT" in read(f) for f in glob.glob(f"{a.suite}/E6/*/*.log") if not f.endswith("_ref.log"))
    for src in ("m5", "uci"):
        for c in CLS:
            R = [r for r in man if r["source"] == src and r["cls"] == c]
            if not R:
                continue
            st_ = []
            for r in R:
                d = log_stats(f"{a.suite}/E6/{src}/{r['series']}_w{r['window']}.log")
                d["mur"] = float(r["mu"]) / float(r["r"])
                try:
                    dem = [int(x.split(",")[-1]) for x in open(r["demand_file"]).read().split()[1:]]
                    d["wr"] = sum(dem) / len(dem) / float(r["r"])   # realised window demand / r
                except OSError:
                    d["wr"] = float("nan")
                d["src"], d["cls"] = src, c
                st_.append(d)
                allst.append(d)
            done = [d for d in st_ if d["time"] == d["time"]]
            nto = sum(1 for d in st_ if d["timeout"] == d["timeout"])
            tim[(src, c)] = [d["time"] for d in done]
            rows.append(f"{'M5' if src == 'm5' else 'Online Retail II'} & {c} & {len(R)} & {len(done)} & " +
                        (f"{nto} & " if SHOW_TO else "") +
                        f"{q([float(r['mu']) for r in R])} & {q([d['mur'] for d in st_], '{:.2f}')} & "
                        f"{q([d['keys'] for d in done], '{:.0f}')} & {q([d['front'] for d in done], '{:.0f}')} & "
                        f"{q([d['time'] for d in done], '{:.2f}')} & "
                        f"{max([d['time'] for d in done]) if done else float('nan'):.1f} \\\\")
    for ref in glob.glob(f"{a.suite}/E6/*/*_ref.csv"):
        base = ref[:-8] + ".csv"
        if os.path.exists(base):
            O = lambda f: {(round(float(r["POC"]), 4), round(float(r["HC"]), 6), round(float(r["CSL"]), 10),
                            round(float(r["ITO"]), 8)) for r in csv.DictReader(open(f))}
            chk[1] += 1
            chk[0] += O(ref) == O(base)
    write("tab_e6.tex", "\\begin{tabular}{@{}ll" + ("r" * (10 if SHOW_TO else 9)) + "@{}}\n\\toprule\n"
          "Source & Demand class & Inst. & Solved & " + ("At limit & " if SHOW_TO else "") +
          "$\\mu$ & $\\mu/r$ & Classes $\\kappa$ & Front size & "
          "Time (s) & Max (s) \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    alltimes = [t for v in tim.values() for t in v]
    write("e6_macros.tex",
          f"\\newcommand{{\\EsixInst}}{{{len(man)}}}\n"
          f"\\newcommand{{\\EsixSolved}}{{{len(alltimes)}}}\n"
          f"\\newcommand{{\\EsixSeries}}{{{len({(r['source'], r['series']) for r in man})}}}\n"
          f"\\newcommand{{\\EsixChkOK}}{{{chk[0]}}}\\newcommand{{\\EsixChkN}}{{{chk[1]}}}\n"
          f"\\newcommand{{\\EsixMedTime}}{{{np.median(alltimes) if alltimes else float('nan'):.2f}}}\n"
          f"\\newcommand{{\\EsixMaxTime}}{{{max(alltimes) if alltimes else float('nan'):.1f}}}\n"
          f"\\newcommand{{\\EsixTotalMin}}{{{sum(alltimes)/60:.0f}}}\n"
          f"\\newcommand{{\\EsixTimeout}}{{{sum(1 for d in allst if d['timeout'] == d['timeout'])}}}\n"
          f"\\newcommand{{\\EsixTlim}}{{{int(max([d['timeout'] for d in allst if d['timeout'] == d['timeout']] or [1800]) / 60)}}}\n"
          f"\\newcommand{{\\EsixPctSolved}}{{{100 * len(alltimes) / len(man):.1f}}}\n")
    # figure: (a) classification of the selected series, (b) computing time per class
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5), constrained_layout=True)
    seen = {}
    for r in man:
        if r["window"] == "0":
            seen[(r["source"], r["series"])] = r
    cols = dict(zip(CLS, ("#1f4e79", "#c55a11", "#548235", "#7030a0")))
    for src, mk in (("m5", "o"), ("uci", "^")):
        for c in CLS:
            P = [(float(r["ADI"]), float(r["CV2"])) for r in seen.values() if r["source"] == src and r["cls"] == c]
            if P:
                P = np.array(P)
                ax[0].scatter(P[:, 0], P[:, 1], s=6, marker=mk, color=cols[c], alpha=0.7,
                              label=c if src == "m5" else None)
    ax[0].axvline(1.32, color="k", lw=0.6, ls="--")
    ax[0].axhline(0.49, color="k", lw=0.6, ls="--")
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax[0].xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax[0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax[0].set_xlabel("ADI (weeks)")
    ax[0].set_ylabel("CV$^2$ of non-zero demand")
    ax[0].legend(frameon=False, fontsize=6, loc="upper right", markerscale=1.5)
    data, labels = [], []
    for src in ("m5", "uci"):
        for c in CLS:
            v = tim.get((src, c), [])
            if v:
                data.append(np.maximum(v, 1e-3))
                labels.append(f"{c[0].upper()}\n{'M5' if src == 'm5' else 'UCI'}")
    if data:
        ax[1].boxplot(data, tick_labels=labels, showfliers=True, flierprops=dict(markersize=2))
        ax[1].set_yscale("log")
        ax[1].tick_params(axis="x", labelsize=6)
        ax[1].set_ylabel("time per instance (s)")
    for c in CLS:
        P = [(d["wr"], d["time"]) for d in allst if d["cls"] == c and d["time"] == d["time"]]
        if P:
            P = np.array(P)
            ax[2].scatter(P[:, 0], np.maximum(P[:, 1], 1e-3), s=4, color=cols[c], alpha=0.6, label=c)
    TO = [(d["wr"], d["timeout"]) for d in allst if d["timeout"] == d["timeout"]]
    if TO:
        TO = np.array(TO)
        ax[2].scatter(TO[:, 0], TO[:, 1], s=14, marker="x", color="k", lw=0.8, label="time limit")
    ax[2].set_xscale("log")
    ax[2].set_yscale("log")
    ax[2].xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax[2].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax[2].set_xlabel("window demand / $r$ (per week)")
    ax[2].set_ylabel("time (s)")
    ax[2].legend(frameon=False, fontsize=6)
    for x, t in zip(ax, "abc"):
        x.set_title(f"({t})", loc="left", fontsize=9)
    fig.savefig(os.path.join(FIG, "fig_e6.pdf"))
    plt.close(fig)
    print(f"E6: {len(alltimes)}/{len(man)} solved, {sum(1 for d in allst if d['timeout'] == d['timeout'])} at time limit, exactness checks {chk[0]}/{chk[1]}")

# ---------------------------------------------------------------- software versions
def cmd(c):
    try:
        return subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip().splitlines()[0]
    except Exception:
        return "n/a"


env = read(f"{a.suite}/environment.txt")
cpu = num(r"Model name:\s*(.+)", env, str) if env else cmd("lscpu | grep 'Model name' | sed 's/.*: *//'")
vers = {"cpu": str(cpu).strip(), "os": cmd(". /etc/os-release && echo $PRETTY_NAME"),
        "gcc": cmd("g++ --version"), "cmake": cmd("cmake --version").split()[-1], "ninja": cmd("ninja --version"),
        "python": platform.python_version(), "numpy": np.__version__, "matplotlib": matplotlib.__version__}
try:
    import scipy
    vers["scipy"] = scipy.__version__
except ImportError:
    vers["scipy"] = "n/a"
try:
    import pymoo
    vers["pymoo"] = pymoo.__version__
except ImportError:
    vers["pymoo"] = "n/a"
try:
    import highspy  # noqa: F401
    vers["highs"] = "HiGHS (via SciPy)"
except ImportError:
    vers["highs"] = "HiGHS (bundled with SciPy)"
tex = "".join(f"\\newcommand{{\\ver{k.capitalize()}}}{{{v.replace('_', ' ').replace('&', 'and').replace('~', '\\textasciitilde{}').replace('%', '\\%')}}}\n"
              for k, v in vers.items())
write("software.tex", tex)
print("\nsoftware:", vers)

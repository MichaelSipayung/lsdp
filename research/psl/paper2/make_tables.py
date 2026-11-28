#!/usr/bin/env python3
"""make_tables.py - tables, macros and the learning-curve figure of Paper 2 from the run outputs.

usage (from research/psl/paper2):  python3 make_tables.py --run .. --out .
Reads in <run>:  sum_syn_<model>.txt, sum_e6_<model>.txt, eval_final_info.txt,
                 model_v3.log, model_q.log, check.txt (output of ./psl_stage1 check)
Writes <out>/tables/{syn,real,ablation,check}.tex, tables/results_macros.tex, figures/fig_learning.pdf
"""
import argparse, os, re, platform, subprocess

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="..")
ap.add_argument("--out", default=".")
ap.add_argument("--main", default="model_v3")
ap.add_argument("--quant", default="model_q")
a = ap.parse_args()
os.makedirs(os.path.join(a.out, "tables"), exist_ok=True)
os.makedirs(os.path.join(a.out, "figures"), exist_ok=True)
R = lambda f: os.path.join(a.run, f)

ROW = re.compile(r"^(policy \(greedy decoding\)|policy \(best of 1\+\d+\)|greedy \+ class enumeration)\s+"
                 r"([-\d.e+]+)\s+([-\d.e+]+)\s+([-\d.e+]+)\s+([-\d.e+]+)\s+([\d.]+)%")

def parse(fn):
    d = {"rows": {}}
    for line in open(fn, errors="replace"):
        line = line.replace("\r", "\n").strip().split("\n")[-1]
        m = ROW.match(line)
        if m:
            key = "dec" if "greedy decoding" in m.group(1) else ("best" if "best" in m.group(1) else "base")
            d["rows"][key] = [float(m.group(i)) for i in range(2, 7)]
        m = re.match(r"^(\d+) \(instance, preference\) pairs", line)
        if m: d["pairs"] = int(m.group(1))
        m = re.search(r"greedy decoding ([\d.]+) ms; decoding \+ (\d+) samples ([\d.]+) ms", line)
        if m: d["dec_ms"], d["samples"], d["pol_ms"] = float(m.group(1)), int(m.group(2)), float(m.group(3))
        m = re.search(r"exact front per instance .*: median ([\d.]+) s, p90 ([\d.]+) s, max ([\d.]+) s over (\d+)", line)
        if m: d["ex_med"], d["ex_p90"], d["ex_max"], d["n"] = float(m.group(1)), float(m.group(2)), float(m.group(3)), int(m.group(4))
    return d

def f4(x): return f"{x:.4f}" if abs(x) < 1e3 else f"{x:.2e}"

def regret_table(d, fn, caption_rows):
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Method & Median & Mean & P90 & Max & On front (\%)\\", r"\midrule"]
    for key, label in caption_rows:
        if key in d["rows"]:
            v = d["rows"][key]
            lines.append(f"{label} & {f4(v[0])} & {f4(v[1])} & {f4(v[2])} & {f4(v[3])} & {v[4]:.1f}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(a.out, "tables", fn), "w").write("\n".join(lines) + "\n")

rows = [("dec", "Policy, greedy decoding"), ("best", "Policy, best of 17"), ("base", "Greedy heuristic + class enumeration")]
M = {}
syn = parse(R(f"sum_syn_{a.main}.txt")); regret_table(syn, "syn.tex", rows)
real = parse(R(f"sum_e6_{a.main}.txt")); regret_table(real, "real.tex", rows)
M.update(SynN=syn.get("n", "??"), SynPairs=syn.get("pairs", "??"), RealN=real.get("n", "??"),
         SynMedDec=f4(syn["rows"]["dec"][0]), SynMedBest=f4(syn["rows"]["best"][0]), SynMedGreedy=f4(syn["rows"]["base"][0]),
         RealMedDec=f4(real["rows"]["dec"][0]), RealMedBest=f4(real["rows"]["best"][0]), RealMedGreedy=f4(real["rows"]["base"][0]),
         RealExactMed=f"{real.get('ex_med', 0):.2f}", RealDecMs=f"{real.get('dec_ms', 0):.1f}", RealPolMs=f"{real.get('pol_ms', 0):.0f}",
         SynExactMax=f"{syn.get('ex_max', 0):.0f}", SynExactP=f"{syn.get('ex_p90', 0):.0f}")

# ---------------------------------------------------------------- ablation
def readlog(fn):
    L = []
    for line in open(fn):
        p = line.split()
        if len(p) >= 4: L.append((int(p[0]), float(p[1]), float(p[2]), float(p[3])))
    return L
lv, lq = readlog(R(f"{a.main}.log")), readlog(R(f"{a.quant}.log"))
qit = None
if os.path.exists(R("eval_final_info.txt")):
    m = re.search(r"iterasi (\d+)", open(R("eval_final_info.txt")).read())
    if m: qit = int(m.group(1))
tv, tq = lv[-1][3], lq[-1][3]
M["AblSpeed"] = f"{tq / tv:.2f}"
M["TrainHours"] = f"{tv / 3600:.1f}"
abl = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
       r"Action space & Iter. & Train time (h) & Median & Mean & P90 & On front (\%)\\", r"\midrule"]
def ablrow(label, model, it, t, data):
    fn = R(f"sum_{data}_{model}.txt")
    if not os.path.exists(fn): return None
    d = parse(fn); v = d["rows"]["dec"]
    return f"{label} & {it} & {t / 3600:.2f} & {f4(v[0])} & {f4(v[1])} & {f4(v[2])} & {v[4]:.1f}\\\\"
for data, title in (("syn", "Held-out synthetic"), ("e6", "Real (E6)")):
    abl.append(r"\multicolumn{7}{l}{\emph{" + title + r", greedy decoding}}\\")
    for r_ in (ablrow("Order events", a.main, lv[-1][0], tv, data),
               ablrow("Per-period quantities", a.quant, lq[-1][0], tq, data),
               ablrow("Per-period quantities (equal time)", f"{a.quant}.it{qit}", qit,
                      next((x[3] for x in lq if x[0] == qit), float("nan")), data) if qit else None):
        if r_: abl.append(r_)
abl += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(a.out, "tables", "ablation.tex"), "w").write("\n".join(abl) + "\n")

# learning curves (rolling median of the logged batch means)
try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt, statistics
    def roll(L, w=9):
        return [statistics.median([x[1] for x in L[max(0, i - w + 1):i + 1]]) for i in range(len(L))]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot([x[3] / 3600 for x in lv], roll(lv), label="order events (this paper)")
    ax.plot([x[3] / 3600 for x in lq], roll(lq), label="per-period quantities", linestyle="--")
    ax.set_xlabel("training time (h)"); ax.set_ylabel("mean Tchebycheff value (rolling median)")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(a.out, "figures", "fig_learning.pdf"))
    M["MPL"] = matplotlib.__version__
except Exception as e:
    print("figure skipped:", e)

# ---------------------------------------------------------------- environment check table
if os.path.exists(R("check.txt")):
    rows_c, cur = [], None
    for line in open(R("check.txt")):
        m = re.search(r"instance (SKU\d) T=(\d+) seed=(\d+): (\d+) classes, exact front (\d+)", line)
        if m: cur = m.groups()
        m = re.search(r"environment: (\d+) complete plans enumerated, (\d+) non-dominated, (\d+) of them on the exact front.*=> (\w+)", line)
        if m and cur: rows_c.append(cur + m.groups()); cur = None
    t = [r"\begin{tabular}{lrrrrrrl}", r"\toprule",
         r"Item & $T$ & Seed & Classes & Exact front & Episodes & Non-dominated & Result\\", r"\midrule"]
    for c in rows_c:
        t.append(f"{c[0]} & {c[1]} & {c[2]} & {c[3]} & {c[4]} & {c[5]} & {c[6]} & {c[8]}\\\\")
    t += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(a.out, "tables", "check.tex"), "w").write("\n".join(t) + "\n")

# ---------------------------------------------------------------- hardware / software
try:
    cpu = [l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")][0]
except Exception:
    cpu = platform.processor()
try:
    gcc = subprocess.run(["g++", "-dumpfullversion"], capture_output=True, text=True).stdout.strip()
except Exception:
    gcc = "??"
M.update(CPU=cpu.replace("(R)", "").replace("(TM)", ""), GCC=gcc, Python=platform.python_version())
M.setdefault("AblSentence", "see the ablation section.")
with open(os.path.join(a.out, "tables", "results_macros.tex"), "w") as f:
    f.write("% generated by make_tables.py\n\\makeatletter\n")
    for k, v in M.items():
        f.write(f"\\@namedef{{rv@{k}}}{{{v}}}\n")
    f.write("\\newcommand{\\Rv}[1]{\\@ifundefined{rv@#1}{??}{\\@nameuse{rv@#1}}}\n\\makeatother\n")
print("written:", ", ".join(sorted(M)))

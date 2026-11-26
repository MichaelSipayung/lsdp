#!/usr/bin/env python3
"""prepare.py - build benchmark instances for E6 from public demand data.

usage (from the repository root, inside the virtualenv):
    python3 bench/prepare.py m5  --data data/m5  --out bench/instances --per-class 50
    python3 bench/prepare.py uci --data data/online_retail_II.xlsx --out bench/instances --per-class 25

Sources
  m5   M5 Forecasting (Walmart), files sales_train_evaluation.csv, calendar.csv, sell_prices.csv
       from https://www.kaggle.com/competitions/m5-forecasting-accuracy (not redistributed).
  uci  UCI Online Retail II (CC BY 4.0), online_retail_II.xlsx,
       https://archive.ics.uci.edu/dataset/502/online+retail+ii (Chen et al., 2012).

All rules below are fixed before any run and are reported in the manuscript.
  * Demand is aggregated to weeks (M5: Walmart week wm_yr_wk; UCI: ISO week starting Monday);
    incomplete weeks at both ends are dropped. M5 is used at the item-store level. UCI demand
    is net of cancellations (weekly sum including cancelled quantities, clipped at zero).
  * Planning windows have T = 48 weeks and are preceded by H = 52 weeks of history. M5 uses the
    last three non-overlapping windows, UCI the last window.
  * Forecast parameters come from the history only (no look-ahead): mu = mean, sigma =
    max(sample std, 0.5); unit price p = median selling price in the history.
  * Demand class (Syntetos and Boylan, 2005) from the history of the first window:
    ADI = H / #non-zero weeks, CV2 = squared coefficient of variation of non-zero sizes;
    smooth (ADI < 1.32, CV2 < 0.49), erratic (ADI < 1.32, CV2 >= 0.49),
    intermittent (ADI >= 1.32, CV2 < 0.49), lumpy (ADI >= 1.32, CV2 >= 0.49).
  * Exclusions: fewer than 4 non-zero weeks in the history of any window, missing price.
  * Item parameters: lead time L = 2 weeks; case pack r = max(1, round(mu/2)) and moq = r
    (Assumption A2 holds); holding cost h = 0.1 p / 52 per week; MS = 1;
    ordering cost c chosen so that the classical EOQ covers four weeks of mean demand,
    i.e. c = 0.1 p (4 mu)^2 / (2 * 52 mu) = 0.1 p * 16 mu / 104.
  * Stratified sample of --per-class series per class (seed 2026); smaller classes are used whole.
"""
import argparse
import csv
import os
import sys

import numpy as np
import pandas as pd

T, H, L, SEED = 48, 52, 2, 2026


def sb_class(hist):
    nz = hist[hist > 0]
    if len(nz) < 2:
        return "excluded", np.inf, np.nan
    adi = len(hist) / len(nz)
    cv2 = (nz.std(ddof=1) / nz.mean()) ** 2
    if adi < 1.32:
        return ("smooth" if cv2 < 0.49 else "erratic"), adi, cv2
    return ("intermittent" if cv2 < 0.49 else "lumpy"), adi, cv2


def params(hist, price):
    mu = float(hist.mean())
    sigma = max(float(hist.std(ddof=1)), 0.5)
    r = max(1, int(round(mu / 2)))
    c = 0.1 * price * 16.0 * mu / 104.0
    return dict(mu=round(mu, 4), sigma=round(sigma, 4), p=round(float(price), 4), r=r, moq=r,
                L=L, c=round(c, 4), MS=1.0, hfrac=0.1)


def build(source, ids, Y, prices, starts, out, per_class):
    """Y: (n_series, n_weeks) weekly demand; prices: callable(i, w0, w1) -> median price or nan."""
    os.makedirs(os.path.join(out, source), exist_ok=True)
    rows, cls = [], {}
    for i, sid in enumerate(ids):
        ok = True
        for s in starts:
            hist = Y[i, s - H:s]
            if (hist > 0).sum() < 4:
                ok = False
                break
        if not ok:
            continue
        c, adi, cv2 = sb_class(Y[i, starts[0] - H:starts[0]])
        if c != "excluded":
            cls.setdefault(c, []).append((i, adi, cv2))
    rng = np.random.default_rng(SEED)
    chosen = []
    for c in ("smooth", "erratic", "intermittent", "lumpy"):
        L_ = cls.get(c, [])
        pick = L_ if len(L_) <= per_class else [L_[j] for j in sorted(rng.choice(len(L_), per_class, replace=False))]
        chosen += [(c, *x) for x in pick]
        print(f"{source}: class {c:12s} eligible {len(L_):6d}  selected {len(pick)}")
    nprice = 0
    for c, i, adi, cv2 in chosen:
        sid = ids[i]
        recs = []
        for k, s in enumerate(starts):
            pr = prices(i, s - H, s)
            if not np.isfinite(pr) or pr <= 0:
                recs = None
                nprice += 1
                break
            hist = Y[i, s - H:s]
            par = params(hist, pr)
            f = os.path.join(out, source, f"{sid}_w{k}.csv")
            with open(f, "w") as fh:
                fh.write("t,D\n")
                for t, d in enumerate(Y[i, s:s + T], 1):
                    fh.write(f"{t},{int(d)}\n")
            recs.append(dict(source=source, series=sid, cls=c, ADI=round(adi, 4), CV2=round(cv2, 4),
                             window=k, start_week=s, T=T, demand_file=f, **par))
        if recs:
            rows += recs
    man = os.path.join(out, source, "manifest.csv")
    with open(man, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{source}: {len(rows)} instances ({len(rows)//len(starts)} series x {len(starts)} windows), "
          f"{nprice} series dropped for missing price -> {man}")


def m5(a):
    cal = pd.read_csv(os.path.join(a.data, "calendar.csv"), usecols=["d", "wm_yr_wk"])
    sales = pd.read_csv(os.path.join(a.data, "sales_train_evaluation.csv"))
    dcols = [c for c in sales.columns if c.startswith("d_")]
    wk = cal.set_index("d").loc[dcols, "wm_yr_wk"].to_numpy()
    X = sales[dcols].to_numpy(dtype=np.int32)
    weeks, first, counts = np.unique(wk, return_index=True, return_counts=True)
    order = np.argsort(first)
    weeks, first, counts = weeks[order], first[order], counts[order]
    Y = np.add.reduceat(X, first, axis=1)
    full = counts == 7
    Y, weeks = Y[:, full], weeks[full]
    nW = Y.shape[1]
    starts = [nW - 3 * T, nW - 2 * T, nW - T]
    print(f"m5: {X.shape[0]} item-store series, {nW} complete weeks, windows start at {starts}")
    ids = (sales["item_id"] + "__" + sales["store_id"]).tolist()
    sp = pd.read_csv(os.path.join(a.data, "sell_prices.csv"))
    sp["key"] = sp["item_id"] + "__" + sp["store_id"]
    widx = {w: j for j, w in enumerate(weeks)}
    sp["j"] = sp["wm_yr_wk"].map(widx)
    sp = sp.dropna(subset=["j"])
    grp = {k: (g["j"].to_numpy(), g["sell_price"].to_numpy()) for k, g in sp.groupby("key")}

    def price(i, w0, w1):
        g = grp.get(ids[i])
        if g is None:
            return np.nan
        m = (g[0] >= w0) & (g[0] < w1)
        return float(np.median(g[1][m])) if m.any() else np.nan

    build("m5", ids, Y, price, starts, a.out, a.per_class)


def uci(a):
    sheets = pd.read_excel(a.data, sheet_name=None, dtype={"StockCode": str, "Invoice": str})
    df = pd.concat(sheets.values(), ignore_index=True)
    df = df[df["StockCode"].astype(str).str.match(r"^\d{5}")]        # products only (no POST, M, D...)
    # net demand: cancellations (invoice "C...", negative quantity) offset the original sale, so
    # that sales cancelled at once (e.g. the 74,215- and 80,995-unit orders) leave no spike;
    # rows with zero price (stock adjustments) and other negative rows are dropped
    canc = df["Invoice"].astype(str).str.startswith("C")
    df = df[(df["Price"] > 0) & ((df["Quantity"] > 0) | canc)]
    df["week"] = pd.to_datetime(df["InvoiceDate"]).dt.to_period("W-SUN").dt.start_time
    first, last = df["week"].min(), df["week"].max()
    df = df[(df["week"] > first) & (df["week"] < last)]               # drop partial end weeks
    allw = pd.date_range(df["week"].min(), df["week"].max(), freq="7D")
    Q = df.pivot_table(index="StockCode", columns="week", values="Quantity", aggfunc="sum",
                       fill_value=0).reindex(columns=allw, fill_value=0).clip(lower=0)
    sales = df[df["Quantity"] > 0]
    P = sales.pivot_table(index="StockCode", columns="week", values="Price", aggfunc="median") \
             .reindex(index=Q.index, columns=allw)
    Y = Q.to_numpy(dtype=np.int64)
    Pm = P.to_numpy(dtype=float)
    nW = Y.shape[1]
    starts = [nW - T]
    print(f"uci: {Y.shape[0]} products, {nW} complete weeks, window starts at {starts}")
    if starts[0] < H:
        sys.exit("not enough history for one window")

    def price(i, w0, w1):
        v = Pm[i, w0:w1]
        v = v[np.isfinite(v)]
        return float(np.median(v)) if len(v) else np.nan

    build("uci", Q.index.astype(str).tolist(), Y, price, starts, a.out, a.per_class)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("source", choices=["m5", "uci"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="bench/instances")
    ap.add_argument("--per-class", type=int, default=50)
    a = ap.parse_args()
    {"m5": m5, "uci": uci}[a.source](a)

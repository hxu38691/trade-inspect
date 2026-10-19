#!/usr/bin/env python3
"""e1fv2@19400 SHORT pos<-0.98, full-market panel (NOT top1000).
Filters: unfiltered vs mean(prior 5 RTH $vol) > $2M ($100k / mean5 < 5%).
Exit leave top50%/top5%; SL none/sl20. Fee/sizing same as e1_pos098_short_rerun.
Seed 42. CPU only — do not touch training.
"""
import json, os, sys, time
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k] = "2"
import numpy as np
import pandas as pd
np.random.seed(42)

ROOT = Path("/root/mlp_o2c73_run")
sys.path.insert(0, str(ROOT)); os.chdir(ROOT); sys.argv = [sys.argv[0]]
import strategies.strategy_e1_hyst as SH
from strategies.strategy_e1_hyst import run_ranked, k_abs
import hold_eval as H, hold_eval_sim as S, strategies.sizing_e1 as SZ, strategies.adj_patch_e1 as A

PANEL = Path("/root/autodl-tmp/hold_eval_s12000_val_pos098")
POSP = Path("/root/autodl-tmp/e1fv2_fullval_19400/e1_pos098_short/pos_panel.npy")
SUM5 = Path("/root/autodl-tmp/e1fv2_liq5med/sum5_prior5.npy")
OUTD = Path("/root/autodl-tmp/e1fv2_short_avg5_lt5pct"); OUTD.mkdir(parents=True, exist_ok=True)

RF = np.load("/root/autodl-tmp/px10_200_valtest/rf_table.npz")
KEY, RFV = RF["key"], RF["rf"]
def rf_rows(sym, day):
    k = np.asarray(sym).astype(np.int64) * 100000000 + np.asarray(day).astype(np.int64)
    p = np.minimum(np.searchsorted(KEY, k), KEY.size - 1)
    return np.where(KEY[p] == k, RFV[p], 1.0)
A.build_rf = lambda sym, day: None
A.rf_rows = rf_rows
SH.STOPS.update({"sl20": (1, 0.20)})

CFG = (0.10, 20, 2.0, 1.5)
THR = 0.98
MEAN5_MIN = 2e6  # $100k / mean5 < 5%
KIND = {0: "band", 1: "eod", 2: "nobar", 3: "stop"}
T0 = time.time()
def log(m): print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)


def view(pack, pos, entry_ok, exit_pct, d0, d1):
    px, ts, day, n = pack["px"], pack["ts"], pack["day"], pack["px"].size
    # exit rank over full $10-100 univ (same as screenshot), entry restricted by entry_ok
    mask = ((px >= 10) & (px <= 100) & np.isfinite(pos)
            & (pack["ts_path"][:, 0] > 0) & (pack["open_p"][:, 0] > 0)
            & (day >= d0) & (day <= d1))
    idx = np.flatnonzero(mask)
    idx = idx[np.argsort(ts[idx], kind="mergesort")]
    rank = np.full(n, 10**9, np.int64); ng = np.zeros(n, np.int64)
    t = ts[idx]
    ch = np.flatnonzero(t[1:] != t[:-1]) + 1
    for s, e in zip(np.r_[0, ch].tolist(), np.r_[ch, t.size].tolist()):
        sl = idx[s:e]
        o = np.argsort(-np.abs(pos[sl]), kind="mergesort")
        r = np.empty(sl.size, np.int64); r[o] = np.arange(sl.size)
        rank[sl] = r; ng[sl] = sl.size
    kx = np.array([k_abs(int(v), exit_pct) for v in range(int(ng.max()) + 1)])[ng]
    hold = rank < kx
    pe = np.where(np.isfinite(pos), pos, 0.0)
    el = (pe > THR) & entry_ok   # unused for short-only but keep
    es = (pe < -THR) & entry_ok
    seg_idx = idx[np.lexsort((ts[idx], day[idx], pack["sym"][idx]))]
    sym = pack["sym"][seg_idx]; ds = day[seg_idx]
    seg = np.cumsum(np.r_[0, (sym[1:] != sym[:-1]) | (ds[1:] != ds[:-1])]).astype(np.int32)
    late = pack["mp"][seg_idx, 0] > 945
    return dict(idx_seg=seg_idx, zdm=pos[seg_idx].astype(np.float64),
                el=el[seg_idx] & ~late, es=es[seg_idx] & ~late,
                hl=hold[seg_idx], hs=hold[seg_idx], seg=seg, sym=sym, day=ds.astype(np.int32),
                open=pack["open_p"][seg_idx], close=pack["close_p"][seg_idx], ts=pack["ts_path"][seg_idx],
                minute=pack["mp"][seg_idx], day_path=pack["day_path"][seg_idx]), float(np.median(ng[idx]))


def main():
    os.nice(19)
    try: os.system("ionice -c3 -p %d" % os.getpid())
    except Exception: pass

    pos = np.load(POSP).astype(np.float64)
    sum5 = np.load(SUM5).astype(np.float64)
    mean5 = sum5 / 5.0
    assert pos.size == sum5.size

    A.install(S, SZ)
    H.OUT = PANEL; S.OUT = PANEL; SZ.PANEL = PANEL
    pack = S._prepare()
    pack["ts_path"] = np.load(PANEL / "ts_path.npy")
    mp = np.array(pack["minute_path"]); mp[mp == 960] = 961; pack["mp"] = mp
    day = pack["day"]
    HP = np.load(PANEL / "high_path.npy"); LP = np.load(PANEL / "low_path.npy")
    px = pack["px"]
    minute = np.load(PANEL / "minute.npy")
    rth = (minute >= 570) & (minute <= 945)
    base = rth & (px >= 10) & (px <= 100) & np.isfinite(pos)

    ok_liq = np.isfinite(mean5) & (mean5 > MEAN5_MIN)
    entry_unf = np.ones(pos.size, bool)  # screenshot: no extra entry filter
    entry_liq = ok_liq

    filt_stats = dict(
        mean5_min_usd=MEAN5_MIN,
        rule="$100k/mean5 < 5%  <=>  mean5 > $2M",
        n_base=int(base.sum()),
        n_pos_lt_m098=int((base & (pos < -THR)).sum()),
        n_liq=int((base & ok_liq).sum()),
        n_short_unf=int((base & (pos < -THR)).sum()),
        n_short_liq=int((base & (pos < -THR) & ok_liq).sum()),
        frac_short_kept=float(((base & (pos < -THR) & ok_liq).sum()) / max((base & (pos < -THR)).sum(), 1)),
        mean5_on_short_unf_p50=float(np.nanmedian(mean5[base & (pos < -THR)])),
        mean5_on_short_liq_p50=float(np.nanmedian(mean5[base & (pos < -THR) & ok_liq])),
    )
    log(json.dumps(filt_stats))
    (OUTD / "filter_stats.json").write_text(json.dumps(filt_stats, indent=1))

    days_all = np.unique(day)
    mid = days_all[len(days_all) // 2]
    HALVES = {"H1": (int(days_all.min()), int(days_all[len(days_all) // 2 - 1])),
              "H2": (int(mid), int(days_all.max())), "VAL": (int(days_all.min()), int(days_all.max()))}

    rows = []; eqs = {}
    for filt_name, eok in (("unfiltered", entry_unf), ("avg5_gt2m", entry_liq)):
        for SL, X in (("none", 50.0), ("none", 5.0), ("sl20", 50.0), ("sl20", 5.0)):
            nm = f"{filt_name}_{SL}_x{int(X)}"
            for half, (d0, d1) in HALVES.items():
                days = np.unique(day[(day >= d0) & (day <= d1)])
                v, ngmed = view(pack, pos, eok, X, d0, d1)
                bk = run_ranked(v, high=HP[v["idx_seg"]], low=LP[v["idx_seg"]], atr=None,
                                stop=SL, sides="short", with_events=False)
                bk.exit_ts = np.where(bk.exit_kind == 3, bk.exit_ts + np.int64(900 * 10**9), bk.exit_ts)
                stopped = bk.exit_kind == 3
                tr = SZ.to_tr(bk, np.zeros(len(bk)))
                ok = np.isfinite(tr["net"])
                tr = {k: a[ok] for k, a in tr.items()}; stopped = stopped[ok]
                if tr["net"].size == 0:
                    rows.append(dict(filter=filt_name, exit=f"{SL}_x{int(X)}", half=half,
                                     cagr=0, sharpe=float("nan"), mdd=0, n_kept_A=0, n_signal_trades=0,
                                     mean_bps=None, win=None))
                    continue
                pa, _rb, kept = SZ.simulate(tr, days, *CFG, np.ones(tr["net"].size), None)
                a, eq = SZ.ann(pa)
                net = tr["net"]
                st = dict(filter=filt_name, exit=f"{SL}_x{int(X)}", half=half, n_days=int(days.size),
                          cagr=a["cagr"], sharpe=a["sharpe"], mdd=a["mdd"],
                          n_signal_trades=int(net.size), n_kept_A=int(kept.sum()),
                          trades_per_day_kept=float(kept.sum() / max(days.size, 1)),
                          univ_names_per_ts_median=ngmed,
                          mean_bps=float(net[kept].mean() * 1e4) if kept.any() else None,
                          win=float((net[kept] > 0).mean()) if kept.any() else None,
                          n_stop_kept=int((stopped & kept).sum()))
                rows.append(st)
                log(json.dumps({k: st[k] for k in ("filter", "exit", "half", "cagr", "sharpe", "mdd", "n_kept_A", "mean_bps", "win")}))
                if half == "VAL":
                    eqs[nm] = eq; eqs["days"] = days

    res = dict(status="done", ckpt="e1fv2 best_rankic.pt step19400",
               rule="SHORT pos<-0.98; full-market CS (same as screenshot); exit leave |pos| top X%; avg5 filter = mean(5 RTH $vol)> $2M",
               fee="0.5*min(0.01/raw open,0.02)/side", sizing="A 10%/20/200%/150%",
               halves=HALVES, seed=42, filter_stats=filt_stats, rows=rows,
               why_vs_noliq="截图/本脚本用全市场截面(~256名/时点)；noliq 误用 PIT top1000 restd(~55)，故 short 从 +700% 变 -39%")
    (OUTD / "summary.json").write_text(json.dumps(res, indent=1, default=float))
    pd.DataFrame(rows).to_csv(OUTD / "table.csv", index=False)
    if eqs:
        np.savez(OUTD / "equity.npz", **eqs)
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        x = pd.to_datetime(eqs["days"].astype(str))
        fig, ax = plt.subplots(figsize=(13, 6))
        styles = [
            ("unfiltered_none_x50", "C0", "--", "unfilt x50"),
            ("avg5_gt2m_none_x50", "C0", "-", "avg5>$2M x50"),
            ("unfiltered_none_x5", "C1", "--", "unfilt x5"),
            ("avg5_gt2m_none_x5", "C1", "-", "avg5>$2M x5"),
            ("unfiltered_sl20_x5", "C3", "--", "unfilt SL20/x5"),
            ("avg5_gt2m_sl20_x5", "C3", "-", "avg5>$2M SL20/x5"),
        ]
        for nm, c, ls, lab in styles:
            if nm not in eqs: continue
            r = [q for q in rows if f"{q['filter']}_{q['exit']}" == nm and q["half"] == "VAL"][0]
            ax.plot(x, eqs[nm], color=c, ls=ls, lw=1.3,
                    label=f"{lab}: CAGR {r['cagr']:.0%} Sh {r['sharpe']:.2f} n={r['n_kept_A']}")
        ax.axvline(pd.Timestamp(str(HALVES["H2"][0])), color="0.5", ls=":")
        ax.set_yscale("log"); ax.grid(True, alpha=0.3, which="both"); ax.legend(fontsize=8, loc="upper left")
        ax.set_title("e1fv2@19400 VAL SHORT pos<-0.98: unfiltered vs avg5>$2M ($100k<5% bar)")
        fig.tight_layout(); fig.savefig(OUTD / "equity.png", dpi=120); plt.close(fig)
    log("DONE " + str(OUTD))

if __name__ == "__main__":
    main()

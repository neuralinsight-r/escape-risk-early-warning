"""
SPC engine for the Escape Risk Early Warning project. Reads ./data, writes ./output.

Method notes:
  - u chart (defects per unit, variable subgroup size) per origin work center, by build week.
  - Phase I baseline = weeks 1-25. Limits are FROZEN after Phase I and applied to Phase II,
    so a process shift cannot inflate its own limits.
  - X-bar/R chart on WC20 fastener torque, subgroup n=5, baseline = first 100 subgroups.
  - Western Electric rules 1-4 evaluated on standardized values (z), which handles varying n.
"""
import os
import numpy as np
import pandas as pd

BASELINE_WEEKS = 25
A2, D3, D4, d2 = 0.577, 0.0, 2.114, 2.326   # constants for n=5
LSL, USL = 22.5, 27.5                        # torque spec, N*m

os.makedirs("output", exist_ok=True)
nc = pd.read_csv("data/nonconformances.csv", parse_dates=["build_date", "detect_date"])
prod = pd.read_csv("data/weekly_production.csv")
tq = pd.read_csv("data/torque_measurements.csv")


def we_rules(z):
    """Return list of violated Western Electric rule numbers for each point."""
    z = np.asarray(z)
    out = [[] for _ in z]
    for i in range(len(z)):
        if abs(z[i]) > 3:
            out[i].append(1)
        if i >= 2:
            w = z[i - 2:i + 1]
            if (w > 2).sum() >= 2 and z[i] > 2 or (w < -2).sum() >= 2 and z[i] < -2:
                out[i].append(2)
        if i >= 4:
            w = z[i - 4:i + 1]
            if (w > 1).sum() >= 4 and z[i] > 1 or (w < -1).sum() >= 4 and z[i] < -1:
                out[i].append(3)
        if i >= 7:
            w = z[i - 7:i + 1]
            if (w > 0).all() or (w < 0).all():
                out[i].append(4)
    return out


# ---------- u charts ----------
u_rows = []
for wc, g in nc.groupby("origin_wc"):
    counts = g.groupby("build_week").size().reindex(prod["week"], fill_value=0).values
    n = prod["units_built"].values
    base = prod["week"].values <= BASELINE_WEEKS
    ubar = counts[base].sum() / n[base].sum()
    sigma = np.sqrt(ubar / n)
    u = counts / n
    z = (u - ubar) / sigma
    rules = we_rules(z)
    for i, wk in enumerate(prod["week"]):
        u_rows.append({"origin_wc": wc, "week": wk, "units": n[i], "defects": counts[i],
                       "u": round(u[i], 4), "center": round(ubar, 4),
                       "ucl": round(ubar + 3 * sigma[i], 4), "lcl": round(max(0, ubar - 3 * sigma[i]), 4),
                       "z": round(z[i], 2), "rules_violated": ",".join(map(str, rules[i])),
                       "signal": int(bool(rules[i]) and z[i] > 0), "phase": "I" if base[i] else "II"})
uchart = pd.DataFrame(u_rows)
uchart.to_csv("output/spc_u_chart.csv", index=False)

# ---------- X-bar / R ----------
sg = tq.groupby("subgroup").agg(week=("week", "first"), xbar=("torque_nm", "mean"),
                                r=("torque_nm", lambda s: s.max() - s.min())).reset_index()
b = sg["subgroup"] <= 100
xbb, rbar = sg.loc[b, "xbar"].mean(), sg.loc[b, "r"].mean()
sg["x_center"], sg["x_ucl"], sg["x_lcl"] = xbb, xbb + A2 * rbar, xbb - A2 * rbar
sg["r_center"], sg["r_ucl"], sg["r_lcl"] = rbar, D4 * rbar, D3 * rbar
sg["z"] = (sg["xbar"] - xbb) / (A2 * rbar / 3)
r = we_rules(sg["z"].values)
sg["rules_violated"] = [",".join(map(str, x)) for x in r]
sg["signal"] = [int(bool(x)) for x in r]
sg.round(4).to_csv("output/spc_torque_xbar_r.csv", index=False)


def cpk(vals):
    m, s = vals.mean(), vals.std(ddof=1)
    return min(USL - m, m - LSL) / (3 * s)


cap = pd.DataFrame([
    {"period": "Baseline (wk 1-39)", "cpk": cpk(tq[tq.week <= 39].torque_nm)},
    {"period": "Late drift (wk 58-65)", "cpk": cpk(tq[tq.week.between(58, 65)].torque_nm)},
    {"period": "Post corrective action (wk 66-78)", "cpk": cpk(tq[tq.week >= 66].torque_nm)},
]).round(2)
cap.to_csv("output/torque_capability.csv", index=False)

# ---------- Pareto ----------
par = nc.groupby(["defect_code", "defect_desc"]).agg(count=("nc_id", "size"), cost=("cost_usd", "sum"),
                                                     escapes=("escaped", "sum")).reset_index()
par = par.sort_values("cost", ascending=False)
par["cum_cost_pct"] = (par["cost"].cumsum() / par["cost"].sum() * 100).round(1)
par.to_csv("output/pareto.csv", index=False)

# ---------- Stratification ----------
strat = {}
for col in ["operator_tenure", "shift", "supplier_id"]:
    s = nc[nc[col].astype(str) != ""].groupby(col).agg(defects=("nc_id", "size"), escapes=("escaped", "sum"))
    s["escape_pct"] = (s["escapes"] / s["defects"] * 100).round(1)
    strat[col] = s
    s.to_csv(f"output/strat_{col}.csv")

# ---------- Headline numbers ----------
def first_sustained(df, key):
    sig = df[(df[key] == 1)]
    return int(sig.iloc[0]["week"]) if len(sig) else None

wc20_sig = first_sustained(uchart[(uchart.origin_wc == "WC20") & (uchart.phase == "II")], "signal")
tq_sig = first_sustained(sg[sg.subgroup > 100], "signal")
m01_esc = nc[(nc.defect_code == "M01") & (nc.escaped == 1) & (nc.build_week >= 40)].sort_values("detect_date")
start = pd.Timestamp("2025-01-06")
first_esc_week = int((m01_esc.iloc[0]["detect_date"] - start).days // 7 + 1) if len(m01_esc) else None
third_esc_week = int((m01_esc.iloc[2]["detect_date"] - start).days // 7 + 1) if len(m01_esc) > 2 else None

total_units = prod.units_built.sum()
print("=== HEADLINE ===")
print(f"records {len(nc)}, units {total_units}, DPU {len(nc)/total_units:.3f}")
print(f"escapes {nc.escaped.sum()} ({nc.escaped.mean()*100:.1f}% of defects), "
      f"escapes per 100 units {nc.escaped.sum()/total_units*100:.2f}")
print(f"COPQ ${nc.cost_usd.sum():,.0f}; escape share of COPQ {nc[nc.escaped==1].cost_usd.sum()/nc.cost_usd.sum()*100:.0f}%")
print(f"Torque X-bar first signal: week {tq_sig} (drift injected wk 40)")
print(f"WC20 u-chart first signal: week {wc20_sig}")
print(f"First drift-era M01 field escape reported: week {first_esc_week}; third: week {third_esc_week}")
print("\nCpk:\n", cap.to_string(index=False))
print("\nPareto by cost:\n", par.head(6).to_string(index=False))
for k, v in strat.items():
    print(f"\n{k}:\n", v.sort_values("defects", ascending=False).head(5).to_string())
print("\nSignals by WC (Phase II weeks flagged):")
print(uchart[uchart.phase == "II"].groupby("origin_wc").signal.sum().to_string())
fa = uchart[(uchart.phase == "II") & uchart.origin_wc.isin(["WC10", "WC40", "WC50"])]
print("\nFirst flagged weeks per WC:")
for wc, g in uchart[(uchart.phase == "II") & (uchart.signal == 1)].groupby("origin_wc"):
    print(wc, g.week.tolist()[:12])

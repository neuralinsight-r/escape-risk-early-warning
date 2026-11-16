"""
Synthetic nonconformance data generator for the Escape Risk Early Warning project.
ALL DATA IS SYNTHETIC. No real program, company, or supplier data is represented.

Outputs (CSV, in ./data):
  weekly_production.csv   units built per week, new-operator mix at WC30
  nonconformances.csv     one row per nonconformance record
  torque_measurements.csv variable data for X-bar/R chart (daily subgroups, n=5)

Injected signals (ground truth, so the analysis can be validated):
  1. WC20 torque tool drift, weeks 40-65; corrective action (recalibration) at week 66
  2. WC30 workforce turnover during production ramp, weeks 30-55 (new operators = higher defect rate)
  3. Supplier S07 bad connector lots, weeks 55-60
  4. Second shift final inspection is less effective (higher escape probability)
"""
import os
import numpy as np
import pandas as pd

SEED = 42
WEEKS = 78
START = pd.Timestamp("2025-01-06")
END = START + pd.Timedelta(weeks=WEEKS)
rng = np.random.default_rng(SEED)

# code: (description, origin work center, category, base defects per unit, responsibility)
DEFECTS = {
    "M01": ("Fastener under-torque", "WC20", "Mechanical", 0.020, "Manufacturing"),
    "M02": ("Bracket misalignment", "WC20", "Mechanical", 0.015, "Manufacturing"),
    "M03": ("FOD found in assembly", "WC20", "Mechanical", 0.008, "Manufacturing"),
    "M04": ("Surface damage", "WC20", "Mechanical", 0.012, "Manufacturing"),
    "E01": ("Crimp defect", "WC30", "Electrical", 0.020, "Manufacturing"),
    "E02": ("Cold solder joint", "WC30", "Electrical", 0.012, "Manufacturing"),
    "E03": ("Wire routing or chafe", "WC30", "Electrical", 0.010, "Manufacturing"),
    "E04": ("Connector pin damage (purchased)", "WC30", "Electrical", 0.010, "Supplier"),
    "X01": ("Actuator backlash out of tolerance", "WC40", "Electro-mechanical", 0.010, "Engineering"),
    "X02": ("Sensor calibration offset", "WC40", "Electro-mechanical", 0.010, "Engineering"),
    "X03": ("Connector mate incomplete", "WC40", "Electro-mechanical", 0.008, "Manufacturing"),
    "A01": ("Missing or incorrect hardware", "WC40", "Assembly", 0.010, "Manufacturing"),
    "A02": ("Traveler or documentation error", "WC50", "Assembly", 0.012, "Planning"),
    "S01": ("Purchased part dimensional nonconformance", "WC10", "Mechanical", 0.015, "Supplier"),
}
WC_NAMES = {
    "WC10": "Receiving Inspection", "WC20": "Mechanical Assembly",
    "WC30": "Harness and Electrical", "WC40": "Electro-Mech Integration",
    "WC50": "Functional Test",
}
TENURE_BANDS = ["<6 mo", "6-24 mo", ">24 mo"]
TENURE_MULT = np.array([3.0, 1.1, 0.8])
SUPPLIERS = [f"S{n:02d}" for n in range(1, 13)]


def units_built(week):
    base = 60 if week < 26 else (110 if week > 52 else 60 + (week - 26) * 50 / 26)
    return int(round(base + rng.normal(0, 3)))


def new_operator_frac(week, wc):
    if wc != "WC30":
        return 0.12
    if week < 30:
        return 0.12
    if week <= 55:
        return 0.12 + 0.33 * min(1, (week - 30) / 10)
    return max(0.22, 0.45 - 0.03 * (week - 55))


def rate_multiplier(code, week):
    if code == "M01":
        if 40 <= week <= 65:
            return 1 + 3.0 * (week - 40) / 25
        if week >= 66:
            return 0.8
    return 1.0


def torque_mean(week):
    if 40 <= week <= 65:
        return 25.0 - 1.4 * (week - 40) / 25
    return 25.0


rows, prod, torque = [], [], []
nc_id = 0
for week in range(1, WEEKS + 1):
    wk_start = START + pd.Timedelta(weeks=week - 1)
    n_units = units_built(week)
    prod.append({"week": week, "week_start": wk_start.date(), "units_built": n_units,
                 "pct_new_operators_wc30": round(new_operator_frac(week, "WC30"), 3)})

    for day in range(5):
        vals = rng.normal(torque_mean(week), 0.5, 5)
        for i, v in enumerate(vals, 1):
            torque.append({"week": week, "date": (wk_start + pd.Timedelta(days=day)).date(),
                           "subgroup": (week - 1) * 5 + day + 1, "sample": i,
                           "torque_nm": round(v, 2)})

    for code, (desc, wc, cat, base, resp) in DEFECTS.items():
        f_new = new_operator_frac(week, wc)
        fracs = np.array([f_new, 0.35, 1 - 0.35 - f_new])
        weights = fracs * TENURE_MULT
        # normalize so that the baseline tenure mix yields the base rate
        base_fracs = np.array([0.12, 0.35, 0.53])
        norm = (base_fracs * TENURE_MULT).sum()
        lam = n_units * base * rate_multiplier(code, week) * weights.sum() / norm

        supplier_bad_lot = code == "E04" and 55 <= week <= 60
        if supplier_bad_lot:
            lam *= 5

        for _ in range(rng.poisson(lam)):
            nc_id += 1
            tenure = rng.choice(TENURE_BANDS, p=weights / weights.sum())
            shift = rng.choice([1, 2], p=[0.6, 0.4])
            supplier = ""
            if resp == "Supplier":
                supplier = "S07" if (supplier_bad_lot and rng.random() < 0.85) else rng.choice(SUPPLIERS)

            p_esc = 0.04 + (0.10 if code == "M01" else 0) + (0.04 if shift == 2 else 0) \
                + (0.03 if tenure == "<6 mo" else 0)
            rest = 1 - p_esc
            detect = rng.choice(["In-process", "Functional test", "Final inspection", "Post-delivery"],
                                p=[rest * 0.62, rest * 0.23, rest * 0.15, p_esc])
            build_date = wk_start + pd.Timedelta(days=int(rng.integers(0, 5)))
            if detect == "Post-delivery":
                detect_date = build_date + pd.Timedelta(weeks=int(rng.integers(8, 30)))
            else:
                detect_date = build_date + pd.Timedelta(days=int(rng.integers(0, 10)))

            if resp == "Supplier":
                dispo = rng.choice(["Return to vendor", "Scrap", "Use-as-is", "Rework"], p=[0.55, 0.2, 0.1, 0.15])
            else:
                dispo = rng.choice(["Rework", "Repair", "Use-as-is", "Scrap"], p=[0.62, 0.12, 0.10, 0.16])
            hours = rng.lognormal(0.6, 0.5)
            cost = hours * 95 + (rng.uniform(400, 2500) if dispo == "Scrap" else 0)
            if detect == "Post-delivery":
                cost *= 10

            rows.append({
                "nc_id": f"NC{nc_id:06d}", "unit_sn": f"SN{week:02d}{int(rng.integers(1, n_units + 1)):03d}",
                "build_week": week, "build_date": build_date.date(), "detect_date": detect_date.date(),
                "origin_wc": wc, "origin_wc_name": WC_NAMES[wc], "defect_code": code,
                "defect_desc": desc, "category": cat, "responsibility": resp, "supplier_id": supplier,
                "operator_tenure": tenure, "shift": shift, "detection_point": detect,
                "escaped": int(detect == "Post-delivery"), "disposition": dispo,
                "labor_hours": round(hours, 2), "cost_usd": round(cost, 2),
            })

nc = pd.DataFrame(rows)
# Right-censoring: an escape not yet reported by the data cutoff does not exist in the records.
hidden = (pd.to_datetime(nc["detect_date"]) > END).sum()
nc = nc[pd.to_datetime(nc["detect_date"]) <= END].reset_index(drop=True)

os.makedirs("data", exist_ok=True)
nc.to_csv("data/nonconformances.csv", index=False)
pd.DataFrame(prod).to_csv("data/weekly_production.csv", index=False)
pd.DataFrame(torque).to_csv("data/torque_measurements.csv", index=False)
print(f"nonconformances: {len(nc)} rows ({hidden} future escapes censored at cutoff {END.date()})")
print(f"units built: {sum(p['units_built'] for p in prod)}, torque readings: {len(torque)}")

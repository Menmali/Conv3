"""
generate_demo.py
Generates a UCI Heart Disease-like synthetic dataset (303 records).

Intentional coverage gaps built in to demonstrate the framework:
  - HighRiskCAD   : excluded (very restrictive multi-condition rule)
  - SevereIschaemia: excluded (requires oldpeak>3 AND thalach<120 AND thal≠3 AND ca≥2)
  - MISSING CATEGORIES for several categorical attributes.
  - Out‑of‑range and unexpected code failures for other columns.
"""

import os
import numpy as np
import pandas as pd

SEED = 42


def generate(n: int = 303, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # ── age ───────────────────────────────────────────────────────────────────
    age = rng.normal(loc=54, scale=9, size=n).clip(29, 77).astype(int)

    # ── sex: ALL MALE → female category missing ───────────────────────────────
    sex = np.ones(n, dtype=int)

    # ── cp: missing code 4 (asymptomatic) ─────────────────────────────────────
    # Expected codes: 1,2,3,4. We will sample only from 1,2,3.
    cp = rng.choice([1, 2, 3], size=n, p=[0.20, 0.40, 0.40])  # code 4 never appears

    # ── trestbps ──────────────────────────────────────────────────────────────
    trestbps = rng.normal(loc=131, scale=18, size=n).clip(90, 200).astype(int)

    # ── chol ─────────────────────────────────────────────────────────────────
    chol = rng.normal(loc=246, scale=52, size=n).clip(125, 564).astype(int)

    # ── fbs: missing code 1 (elevated) ────────────────────────────────────────
    fbs = np.zeros(n, dtype=int)   # all rows have fbs=0 (normal)

    # ── Break any HighRiskCAD combo ───────────────────────────────────────────
    # Now fbs is always 0, so HighRiskCAD is impossible anyway.
    high_risk_mask = (trestbps >= 140) & (chol >= 240) & (fbs == 1) & (sex == 1) & (age >= 45)
    # Since fbs is all 0, mask is empty. No need to fix.

    # ── restecg: missing code 1 (ST-T abnormality) ────────────────────────────
    # Expected: 0,1,2. Sample only 0 and 2.
    restecg = rng.choice([0, 2], size=n, p=[0.50, 0.50])

    # ── thalach ──────────────────────────────────────────────────────────────
    thalach = rng.normal(loc=150, scale=23, size=n).clip(71, 202).astype(int)

    # ── exang: missing code 1 (yes) ───────────────────────────────────────────
    exang = np.zeros(n, dtype=int)   # all rows have exang=0 (no exercise angina)

    # ── oldpeak — mostly zero / small, a few moderate/severe ─────────────────
    oldpeak_raw = rng.exponential(scale=1.0, size=n)
    oldpeak = np.where(rng.random(n) < 0.40, 0.0, oldpeak_raw).round(1).clip(0, 6.2)

    # ── slope: missing code 3 (downsloping) ───────────────────────────────────
    # Expected: 1,2,3. Sample only 1 and 2.
    slope = rng.choice([1, 2], size=n, p=[0.50, 0.50])

    # ── ca (0=62%, 1=23%, 2=10%, 3=5%, 4 records NaN) ────────────────────────
    ca_vals = rng.choice([0, 1, 2, 3], size=n, p=[0.62, 0.23, 0.10, 0.05]).astype(float)
    nan_idx = rng.choice(n, size=4, replace=False)
    ca_vals[nan_idx] = np.nan

    # ── thal: missing code 7 (reversible defect) ──────────────────────────────
    # Expected: 3,6,7. Sample only 3 and 6.
    thal_vals = rng.choice([3, 6], size=n, p=[0.85, 0.15]).astype(float)
    nan_idx2 = rng.choice(n, size=2, replace=False)
    thal_vals[nan_idx2] = np.nan

    # ── Break any SevereIschaemia combo ──────────────────────────────────────
    # (oldpeak>3 AND thalach<120 AND thal≠3 AND ca≥2)
    # With thal never 7, still possible (3 or 6). Ensure none appear.
    severe_mask = (
        (oldpeak > 3) & (thalach < 120) &
        (thal_vals != 3) & (ca_vals >= 2)
    )
    thalach[severe_mask] = rng.integers(120, 180, size=int(severe_mask.sum()))

    # ── num / target: missing code 3 (severe disease) ─────────────────────────
    # Expected: 0,1,2,3,4. Sample from 0,1,2,4.
    num = rng.choice([0, 1, 2, 4], size=n, p=[0.55, 0.20, 0.15, 0.10])

    # ==================== INTENTIONAL OUT-OF-RANGE / UNEXPECTED CODE FAILURES ===
    # (Additional failures for continuous columns)
    idx_out_bp = rng.choice(n, size=2, replace=False)
    trestbps[idx_out_bp] = 1000

    idx_out_chol = rng.choice(n, size=2, replace=False)
    chol[idx_out_chol] = 10000

    idx_out_hr = rng.choice(n, size=2, replace=False)
    thalach[idx_out_hr] = -1

    idx_bad_cp = rng.choice(n, size=2, replace=False)
    cp[idx_bad_cp] = 5      # unexpected code (even though code 4 is missing, 5 is also invalid)

    idx_bad_restecg = rng.choice(n, size=2, replace=False)
    restecg[idx_bad_restecg] = 4   # unexpected

    idx_bad_slope = rng.choice(n, size=2, replace=False)
    slope[idx_bad_slope] = 0       # unexpected

    # Build DataFrame
    df = pd.DataFrame({
        "age":      age,
        "sex":      sex,
        "cp":       cp,
        "trestbps": trestbps,
        "chol":     chol,
        "fbs":      fbs,
        "restecg":  restecg,
        "thalach":  thalach,
        "exang":    exang,
        "oldpeak":  oldpeak,
        "slope":    slope,
        "ca":       ca_vals,
        "thal":     thal_vals,
        "num":      num,
    })
    return df


if __name__ == "__main__":
    df = generate()
    out = os.path.join(os.path.dirname(__file__), "heart_demo.csv")
    df.to_csv(out, index=False)
    print(f"Demo dataset saved → {out}  ({len(df)} records, {len(df.columns)} columns)")
    print(df.head())
    print("\nColumn coverage failures introduced:")
    print("  - sex      : all rows male → female missing (coverage 1/2 = 50%)")
    print("  - cp       : code 4 (asymptomatic) missing → coverage 3/4 = 75%")
    print("  - fbs      : code 1 (elevated) missing → coverage 1/2 = 50%")
    print("  - restecg  : code 1 (ST-T abnormality) missing → coverage 2/3 ≈ 67%")
    print("  - exang    : code 1 (yes) missing → coverage 1/2 = 50%")
    print("  - slope    : code 3 (downsloping) missing → coverage 2/3 ≈ 67%")
    print("  - thal     : code 7 (reversible defect) missing → coverage 2/3 ≈ 67%")
    print("  - num      : code 3 (severe disease) missing → coverage 4/5 = 80%")
    print("\nAdditionally, out-of-range / unexpected code failures for continuous/categorical columns:")
    print("  - trestbps : 2 rows with 1000 (out of range)")
    print("  - chol     : 2 rows with 10000 (out of range)")
    print("  - thalach  : 2 rows with -1 (out of range)")
    print("  - cp       : 2 rows with 5 (unexpected code)")
    print("  - restecg  : 2 rows with 4 (unexpected code)")
    print("  - slope    : 2 rows with 0 (unexpected code)")
"""
generate_demo.py
Generates a UCI Heart Disease-like synthetic dataset (303 records).

Intentional coverage gaps built in to demonstrate the framework:
  - HighRiskCAD   : excluded  (very restrictive multi-condition rule)
  - SevereIschaemia: excluded  (requires oldpeak>3 AND thalach<120 AND thal≠3 AND ca≥2)

Run directly:
    python data/generate_demo.py
"""

import os
import numpy as np
import pandas as pd

SEED = 42


def generate(n: int = 303, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # ── age ───────────────────────────────────────────────────────────────────
    age = rng.normal(loc=54, scale=9, size=n).clip(29, 77).astype(int)

    # ── sex (67% male) ────────────────────────────────────────────────────────
    sex = rng.choice([0, 1], size=n, p=[0.33, 0.67])

    # ── cp  (1=typical 14%, 2=atypical 28%, 3=non-anginal 27%, 4=asymptomatic 31%)
    cp = rng.choice([1, 2, 3, 4], size=n, p=[0.14, 0.28, 0.27, 0.31])

    # ── trestbps — intentionally avoid the combo that triggers HighRiskCAD ──
    # (trestbps >= 140 AND chol >= 240 AND fbs=1 AND sex=1 AND age>=45 is excluded)
    trestbps = rng.normal(loc=131, scale=18, size=n).clip(90, 200).astype(int)

    # ── chol ─────────────────────────────────────────────────────────────────
    chol = rng.normal(loc=246, scale=52, size=n).clip(125, 564).astype(int)

    # ── fbs (15% elevated) ────────────────────────────────────────────────────
    fbs = rng.choice([0, 1], size=n, p=[0.85, 0.15])

    # ── Break any HighRiskCAD combo:
    # wherever trestbps>=140 AND chol>=240 AND fbs=1 AND sex=1 AND age>=45, reduce chol
    high_risk_mask = (trestbps >= 140) & (chol >= 240) & (fbs == 1) & (sex == 1) & (age >= 45)
    chol[high_risk_mask] = rng.integers(125, 239, size=int(high_risk_mask.sum()))

    # ── restecg (0=normal 49%, 1=ST-T 1%, 2=LVH 50%) ─────────────────────────
    restecg = rng.choice([0, 1, 2], size=n, p=[0.49, 0.01, 0.50])

    # ── thalach ──────────────────────────────────────────────────────────────
    thalach = rng.normal(loc=150, scale=23, size=n).clip(71, 202).astype(int)

    # ── exang (33% yes) ───────────────────────────────────────────────────────
    exang = rng.choice([0, 1], size=n, p=[0.67, 0.33])

    # ── oldpeak — mostly zero / small, a few moderate/severe ─────────────────
    oldpeak_raw = rng.exponential(scale=1.0, size=n)
    oldpeak = np.where(rng.random(n) < 0.40, 0.0, oldpeak_raw).round(1).clip(0, 6.2)

    # ── slope (1=up 47%, 2=flat 46%, 3=down 7%) ───────────────────────────────
    slope = rng.choice([1, 2, 3], size=n, p=[0.47, 0.46, 0.07])

    # ── ca (0=62%, 1=23%, 2=10%, 3=5%, 4 records NaN) ────────────────────────
    ca_vals = rng.choice([0, 1, 2, 3], size=n, p=[0.62, 0.23, 0.10, 0.05]).astype(float)
    nan_idx = rng.choice(n, size=4, replace=False)
    ca_vals[nan_idx] = np.nan

    # ── thal (3=normal 55%, 6=fixed 6%, 7=reversible 39%, 2 records NaN) ─────
    thal_vals = rng.choice([3, 6, 7], size=n, p=[0.55, 0.06, 0.39]).astype(float)
    nan_idx2 = rng.choice(n, size=2, replace=False)
    thal_vals[nan_idx2] = np.nan

    # ── Break any SevereIschaemia combo ──────────────────────────────────────
    # (oldpeak>3 AND thalach<120 AND thal≠3 AND ca>=2)
    severe_mask = (
        (oldpeak > 3) & (thalach < 120) &
        (thal_vals != 3) & (ca_vals >= 2)
    )
    # Simply raise thalach for those rows so thalach >= 120
    thalach[severe_mask] = rng.integers(120, 180, size=int(severe_mask.sum()))

    # ── num / target (0=no disease 54%, 1-4=disease 46%) ─────────────────────
    num = rng.choice([0, 1, 2, 3, 4], size=n, p=[0.54, 0.20, 0.13, 0.08, 0.05])

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

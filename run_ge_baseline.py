import os

import numpy as np
import pandas as pd


def _check_not_null(df: pd.DataFrame, col: str) -> dict:
    if col not in df.columns:
        return {
            "expectation": f"{col} not null",
            "success": False,
            "details": f"Column '{col}' not found",
        }
    na = int(df[col].isna().sum())
    return {
        "expectation": f"{col} not null",
        "success": na == 0,
        "details": f"null_count={na}",
    }


def _check_between(df: pd.DataFrame, col: str, lo: float, hi: float) -> dict:
    if col not in df.columns:
        return {
            "expectation": f"{col} between [{lo},{hi}]",
            "success": False,
            "details": f"Column '{col}' not found",
        }

    s = pd.to_numeric(df[col], errors="coerce")
    valid = s.dropna()
    if len(valid) == 0:
        return {
            "expectation": f"{col} between [{lo},{hi}]",
            "success": False,
            "details": "no numeric values after coercion",
        }

    out = valid[(valid < lo) | (valid > hi)]
    return {
        "expectation": f"{col} between [{lo},{hi}]",
        "success": len(out) == 0,
        "details": f"out_of_range_count={int(len(out))}",
        "examples": sorted(out.unique().tolist())[:5],
    }


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "data", "heart_disease.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)

    checks = []

    # Default expectations (no custom domain knowledge)
    for col in ["age", "sex", "cp", "trestbps", "chol", "thalach", "exang", "oldpeak", "num"]:
        checks.append(_check_not_null(df, col))

    # Generic range checks (no disease-specific knowledge)
    checks.append(_check_between(df, "age", 0, 120))
    checks.append(_check_between(df, "trestbps", 0, 300))
    checks.append(_check_between(df, "chol", 0, 600))
    checks.append(_check_between(df, "thalach", 0, 250))
    checks.append(_check_between(df, "oldpeak", 0, 10))

    passed = [c for c in checks if c["success"]]
    failed = [c for c in checks if not c["success"]]

    print("\nGreat-Expectations-like baseline validation (lightweight, version-agnostic)\n")
    print(f"Total checks: {len(checks)}")
    print(f"Passed: {len(passed)}")
    print(f"Failed: {len(failed)}\n")

    if failed:
        print("Failed checks:")
        for c in failed:
            line = f"- {c['expectation']}: {c['details']}"
            if "examples" in c:
                line += f" | examples={c['examples']}"
            print(line)


if __name__ == "__main__":
    main()


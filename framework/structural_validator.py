"""
structural_validator.py  —  Step 3

Scans the raw DataFrame and reports every structural problem:
  • Missing columns
  • NaN / missing values (per column)
  • Non-numeric values in numeric columns
  • Out-of-range values  (derived from ontology bins)
  • Unexpected categorical values  (not in defined ontology codes)
  • Duplicate rows
  • Insufficient record count for ML

Every issue carries a severity: critical | warning | info
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np
import pandas as pd

from .ontology_loader import DomainOntology, OntologyAttribute


# ── helpers re-used from ontology_loader ──────────────────────────────────────

def _find_column(df: pd.DataFrame, attr: OntologyAttribute) -> Optional[str]:
    # NEW: if the ontology attribute explicitly says no dataset column (e.g., HbA1c), skip
    if attr.dataset_column is None:
        return None

    candidates = attr.get_all_column_candidates()
    col_lower   = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name in df.columns:
            return name
        if name.lower() in col_lower:
            return col_lower[name.lower()]
    return None


# ── data classes ──────────────────────────────────────────────────────────────

@dataclass
class ValidationIssue:
    column:        Optional[str]   # None = dataset-wide
    issue_type:    str
    severity:      str             # 'critical' | 'warning' | 'info'
    affected_rows: int
    pct:           float
    details:       str
    examples:      List[Any] = field(default_factory=list)


@dataclass
class ColumnSummary:
    attribute_id:  str
    label:         str
    column:        Optional[str]
    dtype:         str
    total:         int
    missing_count: int
    missing_pct:   float
    unique_count:  int
    min_val:       Optional[float]
    max_val:       Optional[float]
    mean_val:      Optional[float]
    status:        str             # 'ok' | 'warning' | 'critical' | 'absent'


@dataclass
class ValidationReport:
    dataset_name:      str
    n_records:         int
    n_columns:         int
    duplicate_count:   int
    issues:            List[ValidationIssue]
    column_summaries:  List[ColumnSummary]

    @property
    def critical_issues(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "critical"]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def infos(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "info"]

    @property
    def overall_status(self) -> str:
        if self.critical_issues:
            return "critical"
        if self.warnings:
            return "warning"
        return "ok"


# ── main function ─────────────────────────────────────────────────────────────

def validate_structure(
    df: pd.DataFrame,
    ontology: DomainOntology,
    dataset_name: str = "dataset",
) -> ValidationReport:

    issues: List[ValidationIssue] = []
    summaries: List[ColumnSummary] = []
    n = len(df)

    # ── Dataset-wide checks ───────────────────────────────────────────────────

    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        issues.append(ValidationIssue(
            column=None,
            issue_type="Duplicate rows",
            severity="warning",
            affected_rows=dup_count,
            pct=dup_count / n,
            details=f"{dup_count} exact duplicate rows detected — may skew model training",
        ))

    if n < 100:
        issues.append(ValidationIssue(
            column=None,
            issue_type="Insufficient records",
            severity="warning",
            affected_rows=n,
            pct=1.0,
            details=f"Only {n} records — insufficient for reliable ML (minimum recommended: 100)",
        ))

    # ── Per-attribute checks ──────────────────────────────────────────────────

    for attr in ontology.attributes:
        col = _find_column(df, attr)

        # ── Column absent ─────────────────────────────────────────────────────
        if col is None:
            sev = "critical" if attr.required else "warning"
            issues.append(ValidationIssue(
                column=attr.dataset_column,
                issue_type="Missing column",
                severity=sev,
                affected_rows=n,
                pct=1.0,
                details=f"Column '{attr.dataset_column}' not found in dataset",
            ))
            summaries.append(ColumnSummary(
                attribute_id=attr.id, label=attr.label, column=None,
                dtype="—", total=n, missing_count=n, missing_pct=1.0,
                unique_count=0, min_val=None, max_val=None, mean_val=None,
                status="absent",
            ))
            continue

        series = df[col]

        # ── Missing values ────────────────────────────────────────────────────
        missing = int(series.isna().sum())
        missing_pct = missing / n

        if missing_pct > 0.5:
            sev = "critical"
        elif missing_pct > attr.missing_threshold: 
            sev = "warning"
        elif missing_pct > 0:
            sev = "info"
        else:
            sev = None

        if sev:
            issues.append(ValidationIssue(
                column=col,
                issue_type="Missing values (NaN)",
                severity=sev,
                affected_rows=missing,
                pct=missing_pct,
                details=f"'{col}': {missing} NaN values ({missing_pct:.1%})",
            ))

        # ── Type errors ───────────────────────────────────────────────────────
        numeric = pd.to_numeric(series, errors="coerce")
        type_errors = int(numeric.isna().sum()) - missing
        if type_errors > 0:
            bad_vals = series[numeric.isna() & series.notna()].head(5).tolist()
            issues.append(ValidationIssue(
                column=col,
                issue_type="Non-numeric values",
                severity="critical" if type_errors > n * 0.10 else "warning",
                affected_rows=type_errors,
                pct=type_errors / n,
                details=f"'{col}': {type_errors} cells cannot be parsed as numbers",
                examples=bad_vals,
            ))

        valid_num = numeric.dropna()

        # ── Out-of-range (continuous) ─────────────────────────────────────────
        if attr.type == "continuous" and attr.coverage_bins and len(valid_num) > 0:
            lo = min(b.min for b in attr.coverage_bins)
            hi = max(b.max for b in attr.coverage_bins)
            out = valid_num[(valid_num < lo) | (valid_num > hi)]
            if len(out) > 0:
                issues.append(ValidationIssue(
                    column=col,
                    issue_type="Out-of-range values",
                    severity="warning",
                    affected_rows=len(out),
                    pct=len(out) / n,
                    details=f"'{col}': {len(out)} values outside expected range [{lo}, {hi}]",
                    examples=sorted(out.unique().tolist())[:5],
                ))

        # ── Unexpected categorical codes ──────────────────────────────────────
        if attr.type in ("categorical", "ordinal") and attr.values and len(valid_num) > 0:
            valid_codes = {float(v.code) for v in attr.values}
            unexpected = valid_num[~valid_num.isin(valid_codes)]
            if len(unexpected) > 0:
                issues.append(ValidationIssue(
                    column=col,
                    issue_type="Unexpected category codes",
                    severity="warning",
                    affected_rows=len(unexpected),
                    pct=len(unexpected) / n,
                    details=(
                        f"'{col}': {len(unexpected)} values not in defined set "
                        f"{sorted(int(c) for c in valid_codes)}"
                    ),
                    examples=sorted(unexpected.unique().tolist())[:5],
                ))

        # ── Column summary ────────────────────────────────────────────────────
        col_issues = [i for i in issues if i.column == col]
        if any(i.severity == "critical" for i in col_issues):
            status = "critical"
        elif any(i.severity == "warning" for i in col_issues):
            status = "warning"
        elif any(i.severity == "info" for i in col_issues):
            status = "info"
        else:
            status = "ok"

        summaries.append(ColumnSummary(
            attribute_id=attr.id,
            label=attr.label,
            column=col,
            dtype=str(series.dtype),
            total=n,
            missing_count=missing,
            missing_pct=missing_pct,
            unique_count=int(series.nunique()),
            min_val=float(valid_num.min()) if len(valid_num) else None,
            max_val=float(valid_num.max()) if len(valid_num) else None,
            mean_val=float(valid_num.mean()) if len(valid_num) else None,
            status=status,
        ))

    return ValidationReport(
        dataset_name=dataset_name,
        n_records=n,
        n_columns=len(df.columns),
        duplicate_count=dup_count,
        issues=issues,
        column_summaries=summaries,
    )

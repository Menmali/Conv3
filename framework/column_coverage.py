"""
column_coverage.py  —  Step 4

For every attribute in the ontology, computes:
  • coverage_score  — what % of expected clinical bins/codes are observed
  • value_distribution — how many records fall in each bin
  • missing_values — which bins/codes are completely absent

Value matching for categorical attributes uses a three-pass strategy:
  1. Float equality  (handles numeric codes stored as strings: "0.0" == 0)
  2. Case-insensitive string match against the canonical code string
  3. Case-insensitive match against each value's value_aliases list

This means a dataset with sex="Female" / "Male" is correctly matched to
canonical codes 0 / 1 when the ontology defines value_aliases for those codes.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

from .ontology_loader import DomainOntology, OntologyAttribute


# ── helpers ───────────────────────────────────────────────────────────────────

def _find_column(df: pd.DataFrame, attr: OntologyAttribute) -> Optional[str]:
    # If the ontology attribute explicitly has no dataset column (e.g., HbA1c), return None immediately
    if attr.dataset_column is None:
        return None

    col_lower = {c.lower(): c for c in df.columns}
    for name in attr.get_all_column_candidates():
        if name in df.columns:
            return name
        if name.lower() in col_lower:
            return col_lower[name.lower()]
    return None


def _discretize(series: pd.Series, attr: OntologyAttribute) -> pd.Series:
    """
    Map raw column values to their canonical code strings.

    For categorical/ordinal attributes the matching order is:
        1. float(val) == float(code)          — numeric equality
        2. str(val).strip().lower() == str(code).strip().lower()
        3. str(val).strip().lower() in value_aliases (case-insensitive)

    Returns a Series of str canonical-code values (or None for unmatched/NaN).
    """
    if attr.type in ("categorical", "ordinal"):

        # Pre-build a lookup: normalised_string → canonical code string
        # Built once per attribute, not once per row.
        alias_map: Dict[str, str] = {}
        for v in attr.values:
            canonical = str(v.code)
            # canonical code itself
            alias_map[canonical.strip().lower()] = canonical
            # explicit value_aliases from the ontology
            for a in v.value_aliases:
                alias_map[str(a).strip().lower()] = canonical
            # label string (e.g. "female") as a convenience alias
            alias_map[v.label.strip().lower()] = canonical

        def cat_map(val):
            if pd.isna(val):
                return None
            # Pass 1: float equality (handles "0.0" == 0, etc.)
            try:
                f_val = float(val)
                for v in attr.values:
                    try:
                        if f_val == float(v.code):
                            return str(v.code)
                    except (TypeError, ValueError):
                        pass
            except (TypeError, ValueError):
                pass
            # Pass 2 & 3: normalised string lookup
            key = str(val).strip().lower()
            return alias_map.get(key, None)

        return series.apply(cat_map)

    elif attr.type == "continuous" and attr.coverage_bins:

        def cont_map(val):
            if pd.isna(val):
                return None
            v = float(val)
            for b in attr.coverage_bins:
                if b.min <= v <= b.max:
                    return b.label
            return None

        return series.apply(cont_map)

    return series.astype(str)


def _expected_labels(attr: OntologyAttribute) -> List[str]:
    """ Human-readable expected labels (for display)."""
    if attr.type in ("categorical", "ordinal"):
        return [f"{v.code} ({v.label})" for v in attr.values]
    elif attr.type == "continuous" and attr.coverage_bins:
        return [b.label for b in attr.coverage_bins]
    return []


def _expected_keys(attr: OntologyAttribute) -> List[str]:
    """Machine keys used by the discretizer (match disc output)."""
    if attr.type in ("categorical", "ordinal"):
        return [str(v.code) for v in attr.values]
    elif attr.type == "continuous" and attr.coverage_bins:
        return [b.label for b in attr.coverage_bins]
    return []


# ── data classes ──────────────────────────────────────────────────────────────

@dataclass
class ColumnCoverageResult:
    attribute_id:       str
    label:              str
    matched_column:     Optional[str]
    attr_type:          str
    required:           bool
    present:            bool
    coverage_score:     float           # 0.0 – 1.0
    expected_labels:    List[str]       # human-readable expected values
    observed_keys:      List[str]       # which keys were observed
    missing_labels:     List[str]       # human-readable missing values
    value_distribution: Dict[str, int]  # key → row count
    total_valid:        int             # non-NaN row count
    note:               str = ""


@dataclass
class ColumnCoverageReport:
    results:       List[ColumnCoverageResult]
    overall_score: float


# ── main function ─────────────────────────────────────────────────────────────

def compute_column_coverage(
    df: pd.DataFrame,
    ontology: DomainOntology,
) -> ColumnCoverageReport:
    """Compute per-column coverage against the ontology."""

    results: List[ColumnCoverageResult] = []

    for attr in ontology.attributes:
        col            = _find_column(df, attr)
        exp_labels     = _expected_labels(attr)
        exp_keys       = _expected_keys(attr)

        # ── Column absent ─────────────────────────────────────────────────────
        if col is None or not exp_keys:
            results.append(ColumnCoverageResult(
                attribute_id=attr.id,
                label=attr.label,
                matched_column=col,
                attr_type=attr.type,
                required=attr.required,
                present=col is not None,
                coverage_score=0.0 if col is None else 1.0,
                expected_labels=exp_labels,
                observed_keys=[],
                missing_labels=exp_labels if col is None else [],
                value_distribution={},
                total_valid=0,
                note="Column absent" if col is None else "No ontology bins defined",
            ))
            continue

        # ── Discretize ────────────────────────────────────────────────────────
        # For categorical columns that may contain strings, skip numeric coercion
        # so we preserve values like "Female", "Male" for the alias matcher.
        if attr.type in ("categorical", "ordinal"):
            raw      = df[col]
            # Still count numeric-parseable non-NaN as valid
            numeric  = pd.to_numeric(raw, errors="coerce")
            # total_valid = rows that are not NaN in the raw column
            total_valid = int(raw.notna().sum())
            disc = _discretize(raw, attr)
        else:
            numeric     = pd.to_numeric(df[col], errors="coerce")
            total_valid = int(numeric.notna().sum())
            disc        = _discretize(numeric, attr)

        observed_set = set(disc.dropna().unique())
        expected_set = set(exp_keys)
        covered      = expected_set & observed_set
        missing_keys = sorted(expected_set - observed_set)

        # ── Distribution: count per expected bin ──────────────────────────────
        distribution: Dict[str, int] = {}
        for key in exp_keys:
            distribution[key] = int((disc == key).sum())

        # ── Map missing keys → human labels ───────────────────────────────────
        key_to_label = dict(zip(exp_keys, exp_labels))
        missing_labels = [key_to_label.get(k, k) for k in missing_keys]

        score = len(covered) / len(expected_set) if expected_set else 1.0

        results.append(ColumnCoverageResult(
            attribute_id=attr.id,
            label=attr.label,
            matched_column=col,
            attr_type=attr.type,
            required=attr.required,
            present=True,
            coverage_score=score,
            expected_labels=exp_labels,
            observed_keys=sorted(covered),
            missing_labels=missing_labels,
            value_distribution=distribution,
            total_valid=total_valid,
            note="",
        ))

    scores  = [r.coverage_score for r in results]
    overall = float(np.mean(scores)) if scores else 0.0

    return ColumnCoverageReport(results=results, overall_score=overall)
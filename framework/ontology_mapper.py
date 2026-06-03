"""
ontology_mapper.py  —  Step 5

Semantic stage fix (vs. previous version):
  Missing columns are no longer silently dropped from the expected-combination count.
  Instead, a missing column contributes a Series of None to the disc_df, so
  dropna() eliminates all rows → 0 observed combinations for that stage.

  This means:
    • A dataset with ALL stage columns present  → realistic coverage %
    • A dataset MISSING a stage column          → 0 observed / N expected = 0%
    • A dataset with wrong value types (e.g. sex="Female") → alias-matched correctly
"""

from .column_coverage import ColumnCoverageReport

import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .ontology_loader import (
    ClassificationRule, DomainOntology, OntologyAttribute, RuleCondition,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _find_column(df: pd.DataFrame, attr: OntologyAttribute) -> Optional[str]:
    # If the ontology attribute explicitly has no dataset column (e.g., HbA1c), return None
    if attr.dataset_column is None:
        return None

    col_lower = {c.lower(): c for c in df.columns}
    for name in attr.get_all_column_candidates():
        if name is None:
            continue
        if name in df.columns:
            return name
        if name.lower() in col_lower:
            return col_lower[name.lower()]
    return None


def _discretize(series: pd.Series, attr: OntologyAttribute) -> pd.Series:
    """
    Three-pass value matching for categorical/ordinal attributes:
      1. float equality
      2. case-insensitive match against canonical code string
      3. case-insensitive match against value_aliases list
    """
    if attr.type in ("categorical", "ordinal"):

        # Build a single normalised-string → canonical-code lookup
        alias_map: Dict[str, str] = {}
        for v in attr.values:
            canonical = str(v.code)
            alias_map[canonical.strip().lower()] = canonical
            for a in v.value_aliases:
                alias_map[str(a).strip().lower()] = canonical
            alias_map[v.label.strip().lower()] = canonical

        def cat_map(val):
            if pd.isna(val):
                return None
            # Pass 1 – float equality (handles 0 == 0.0, "0" == 0, etc.)
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
            # Pass 2 & 3 – normalised string lookup (aliases + label)
            return alias_map.get(str(val).strip().lower(), None)

        return series.apply(cat_map)

    elif attr.type == "continuous" and attr.coverage_bins:

        def cont_map(val):
            if pd.isna(val):
                return None
            try:
                v = float(val)
            except (TypeError, ValueError):
                return None
            for b in attr.coverage_bins:
                if b.min <= v <= b.max:
                    return b.label
            return None

        return series.apply(cont_map)

    return series.astype(str)


def _apply_condition(series: pd.Series, cond: RuleCondition) -> pd.Series:
    ops = {
        ">=": lambda s, v: s >= v,
        "<=": lambda s, v: s <= v,
        ">":  lambda s, v: s > v,
        "<":  lambda s, v: s < v,
        "==": lambda s, v: s == v,
        "!=": lambda s, v: s != v,
    }
    fn = ops.get(cond.op)
    if fn is None:
        return pd.Series([False] * len(series), index=series.index)
    return fn(series, cond.value).fillna(False)


def _numeric_for_rule(
    df: pd.DataFrame,
    attr: OntologyAttribute,
    col: str,
    cache: Dict[str, pd.Series],
) -> pd.Series:
    """
    Return a numeric Series for rule evaluation.
    For string-coded columns (e.g. sex="Female") we route through the
    alias discretizer then coerce the resulting canonical code to float.
    """
    if col in cache:
        return cache[col]
    raw     = df[col]
    numeric = pd.to_numeric(raw, errors="coerce")
    # If more than half the non-null values failed numeric coercion → string column
    non_null = raw.notna().sum()
    coerce_fail = int(numeric.isna().sum()) - int(raw.isna().sum())
    if non_null > 0 and coerce_fail / non_null > 0.5:
        disc = _discretize(raw, attr)
        numeric = pd.to_numeric(disc, errors="coerce")
    cache[col] = numeric
    return numeric


def _apply_rule(
    df: pd.DataFrame,
    rule: ClassificationRule,
    col_map: Dict[str, Optional[str]],
    numeric_cache: Dict[str, pd.Series],
    attr_map: Dict[str, OntologyAttribute],
) -> pd.Series:
    masks = []
    for cond in rule.conditions:
        col  = col_map.get(cond.attribute)
        attr = attr_map.get(cond.attribute)
        if not col or attr is None:
            masks.append(pd.Series([False] * len(df), index=df.index))
        else:
            s = _numeric_for_rule(df, attr, col, numeric_cache)
            masks.append(_apply_condition(s, cond))

    if not masks:
        return pd.Series([False] * len(df), index=df.index)

    result = masks[0]
    for m in masks[1:]:
        result = (result & m) if rule.logic == "AND" else (result | m)
    return result


# ── data classes ──────────────────────────────────────────────────────────────

@dataclass
class SemanticStageResult:
    stage:                 int
    attributes:            List[str]
    description:           str
    expected_combinations: int
    observed_combinations: int
    score:                 float
    sample_missing_combos: List[Tuple]


@dataclass
class SubclassResult:
    subclass_id:  str
    label:        str
    short_label:  str
    priority:     str
    clinical_note: str
    guideline:    str
    covered:      bool
    sample_count: int
    adequate:     bool
    min_required: int


@dataclass
class OntologyMappingReport:
    semantic_stages:      List[SemanticStageResult]
    subclass_results:     List[SubclassResult]
    semantic_coverage:    float
    subclass_coverage:    float
    adequacy_score:       float
    overall_score:        float
    verdict:              str
    sufficient_threshold: float
    borderline_threshold: float


# ── main function ─────────────────────────────────────────────────────────────

def map_to_ontology(
    df: pd.DataFrame,
    ontology: DomainOntology,
    col_cov: ColumnCoverageReport | None = None,
) -> OntologyMappingReport:

    col_map: Dict[str, Optional[str]] = {
        attr.id: _find_column(df, attr) for attr in ontology.attributes
    }
    attr_map: Dict[str, OntologyAttribute] = {
        attr.id: attr for attr in ontology.attributes
    }
    numeric_cache: Dict[str, pd.Series] = {}

    # ── A) Semantic combination coverage ─────────────────────────────────────
    stages: List[SemanticStageResult] = []

    for stage in ontology.semantic_progression:

        # Collect ALL stage attributes (whether their column is present or not)
        stage_pairs: List[Tuple[OntologyAttribute, Optional[str]]] = []
        for attr_id in stage.attributes:
            attr = ontology.get_attribute(attr_id)
            if attr is None:
                continue
            col = col_map.get(attr_id)
            stage_pairs.append((attr, col if (col and col in df.columns) else None))

        if not stage_pairs:
            stages.append(SemanticStageResult(
                stage=stage.stage, attributes=stage.attributes,
                description=stage.description,
                expected_combinations=0, observed_combinations=0,
                score=0.0, sample_missing_combos=[],
            ))
            continue

        # Expected combinations — always from ALL stage attributes
        ranges = [attr.expected_discrete_values() for attr, _ in stage_pairs]
        if not ranges or any(len(r) == 0 for r in ranges):
            stages.append(SemanticStageResult(
                stage=stage.stage, attributes=stage.attributes,
                description=stage.description,
                expected_combinations=0, observed_combinations=0,
                score=0.0, sample_missing_combos=[],
            ))
            continue

        expected = set(itertools.product(*ranges))

        # Discretize — missing columns become all-None Series.
        # dropna() will then remove ALL rows for stages with missing columns,
        # giving 0 observed combinations (correct: can't form full combos
        # when a dimension is unknown).
        disc_cols: Dict[str, pd.Series] = {}
        for attr, col in stage_pairs:
            if col is not None:
                disc_cols[attr.id] = _discretize(df[col], attr)
            else:
                disc_cols[attr.id] = pd.Series([None] * len(df), index=df.index, dtype=object)

        disc_df  = pd.DataFrame(disc_cols).dropna()
        observed = set(map(tuple, disc_df.values.tolist()))

        covered = expected & observed
        missing = sorted(expected - covered)
        score   = len(covered) / len(expected) if expected else 1.0

        stages.append(SemanticStageResult(
            stage=stage.stage, attributes=stage.attributes,
            description=stage.description,
            expected_combinations=len(expected),
            observed_combinations=len(covered),
            score=score,
            sample_missing_combos=missing[:10],
        ))

    # ── B) Sub-class coverage ─────────────────────────────────────────────────
    subclass_results: List[SubclassResult] = []

    for sc in ontology.subclasses:
        rule_col_map: Dict[str, Optional[str]] = {
            cond.attribute: col_map.get(cond.attribute)
            for cond in sc.rule.conditions
        }
        mask  = _apply_rule(df, sc.rule, rule_col_map, numeric_cache, attr_map)
        count = int(mask.sum())

        subclass_results.append(SubclassResult(
            subclass_id=sc.id,
            label=sc.label,
            short_label=sc.short_label,
            priority=sc.priority,
            clinical_note=sc.clinical_note,
            guideline=sc.guideline,
            covered=count > 0,
            sample_count=count,
            adequate=count >= sc.min_required_samples,
            min_required=ontology.min_samples_per_subclass,
        ))

    # ── Scores ────────────────────────────────────────────────────────────────
    semantic_cov = stages[-1].score if stages else 0.0
    n_sc         = len(subclass_results)
    subclass_cov = sum(1 for r in subclass_results if r.covered)  / n_sc if n_sc else 0.0
    adequacy     = sum(1 for r in subclass_results if r.adequate) / n_sc if n_sc else 0.0

    base_score = 0.35 * semantic_cov + 0.45 * subclass_cov + 0.20 * adequacy

    # ── Structural penalty ────────────────────────────────────────────────────
    structural_penalty = 0.0
    if col_cov is not None:
        required_cols_missing = sum(
            1 for r in col_cov.results if not r.present and r.required
        )
        critical_sparse = sum(
            1 for r in col_cov.results
            if r.present and (
                r.coverage_score < 0.50
                or (r.total_valid < 0.80 * len(df))
            )
        )
        structural_penalty = (0.15 * required_cols_missing) + (0.05 * critical_sparse)

    overall = max(0.0, base_score - structural_penalty)

    if overall >= ontology.sufficient_threshold:
        verdict = "sufficient"
    elif overall >= ontology.borderline_threshold:
        verdict = "borderline"
    else:
        verdict = "insufficient"

    return OntologyMappingReport(
        semantic_stages=stages,
        subclass_results=subclass_results,
        semantic_coverage=semantic_cov,
        subclass_coverage=subclass_cov,
        adequacy_score=adequacy,
        overall_score=overall,
        verdict=verdict,
        sufficient_threshold=ontology.sufficient_threshold,
        borderline_threshold=ontology.borderline_threshold,
    )

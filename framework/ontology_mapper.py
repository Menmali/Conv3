"""
ontology_mapper.py  —  Step 5

Answers the question: does this dataset cover the disease ontology well enough
to support reliable ML predictions?

Two measurements:
  A) Semantic combination coverage (progressive) — L3
  B) Sub-class coverage — L4 + L5

Each SemanticStageResult now carries four scores:
  score             — raw (observed / total expected combinations)
  theoretical_max   — best any dataset of this size could achieve
                      = min(n, expected) / expected
  normalized_score  — raw / theoretical_max  (% of what is achievable)
  random_baseline   — coupon-collector expectation: what a uniform
                      random sample of n records would achieve
                      = C * (1 - ((C-1)/C)^n) / C

OntologyMappingReport carries n_records so the chart layer does not need
a separate argument.
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
    col_lower = {c.lower(): c for c in df.columns}
    for name in attr.get_all_column_candidates():
        if name in df.columns:
            return name
        if name.lower() in col_lower:
            return col_lower[name.lower()]
    return None


def _discretize(series: pd.Series, attr: OntologyAttribute) -> pd.Series:
    if attr.type in ("categorical", "ordinal"):
        valid = {v.code for v in attr.values}

        def cat_map(val):
            if pd.isna(val):
                return None
            for code in valid:
                try:
                    if float(val) == float(code):
                        return str(code)
                except (TypeError, ValueError):
                    pass
            return None

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


def _apply_rule(
    df: pd.DataFrame,
    rule: ClassificationRule,
    col_map: Dict[str, str],
    numeric_cache: Dict[str, pd.Series],
) -> pd.Series:
    masks = []
    for cond in rule.conditions:
        col = col_map.get(cond.attribute)
        if not col:
            masks.append(pd.Series([False] * len(df), index=df.index))
        else:
            s = numeric_cache.get(col, pd.to_numeric(df[col], errors="coerce"))
            masks.append(_apply_condition(s, cond))

    if not masks:
        return pd.Series([False] * len(df), index=df.index)

    result = masks[0]
    for m in masks[1:]:
        result = (result & m) if rule.logic == "AND" else (result | m)
    return result


def _coupon_collector_expected(C: int, n: int) -> float:
    """
    Expected number of unique combinations when drawing n records
    uniformly at random from C equally-likely combinations.

    Formula: E[unique] = C * (1 - ((C-1)/C)^n)
    Edge cases handled: C=0 → 0, C=1 → 1, n=0 → 0.
    """
    if C <= 0 or n <= 0:
        return 0.0
    if C == 1:
        return 1.0
    return C * (1.0 - ((C - 1) / C) ** n)


# ── data classes ──────────────────────────────────────────────────────────────

@dataclass
class SemanticStageResult:
    stage:                 int
    attributes:            List[str]
    description:           str
    expected_combinations: int
    observed_combinations: int
    score:                 float        # raw: observed / expected
    theoretical_max:       float        # min(n, expected) — absolute count
    normalized_score:      float        # score / (theoretical_max/expected)
    random_baseline:       float        # coupon-collector expected score
    sample_missing_combos: List[Tuple]  # first 10 missing


@dataclass
class SubclassResult:
    subclass_id:   str
    label:         str
    short_label:   str
    priority:      str
    clinical_note: str
    guideline:     str
    covered:       bool
    sample_count:  int
    adequate:      bool
    min_required:  int


@dataclass
class OntologyMappingReport:
    semantic_stages:      List[SemanticStageResult]
    subclass_results:     List[SubclassResult]
    semantic_coverage:    float   # final-stage raw semantic score
    subclass_coverage:    float   # % sub-classes with ≥ 1 sample
    adequacy_score:       float   # % sub-classes with ≥ min samples
    overall_score:        float   # weighted combination
    verdict:              str     # 'sufficient'|'borderline'|'insufficient'
    sufficient_threshold: float
    borderline_threshold: float
    n_records:            int     # ← NEW: carried through for chart layer


# ── main function ─────────────────────────────────────────────────────────────

def map_to_ontology(
    df:      pd.DataFrame,
    ontology: DomainOntology,
    col_cov: Optional["ColumnCoverageReport"] = None,
) -> OntologyMappingReport:

    n = len(df)   # record count used throughout

    # Build column and numeric caches once
    col_map: Dict[str, Optional[str]] = {
        attr.id: _find_column(df, attr) for attr in ontology.attributes
    }
    numeric_cache: Dict[str, pd.Series] = {
        col: pd.to_numeric(df[col], errors="coerce")
        for col in col_map.values()
        if col and col in df.columns
    }

    # ── A) Semantic combination coverage ─────────────────────────────────────
    stages: List[SemanticStageResult] = []

    for stage in ontology.semantic_progression:
        active = []
        for attr_id in stage.attributes:
            attr = ontology.get_attribute(attr_id)
            col  = col_map.get(attr_id)
            if attr and col and col in df.columns:
                active.append((attr, col))

        if not active:
            stages.append(SemanticStageResult(
                stage=stage.stage, attributes=stage.attributes,
                description=stage.description,
                expected_combinations=0, observed_combinations=0,
                score=0.0,
                theoretical_max=0.0, normalized_score=0.0,
                random_baseline=0.0,
                sample_missing_combos=[],
            ))
            continue

        ranges   = [attr.expected_discrete_values() for attr, _ in active]
        expected = set(itertools.product(*ranges))
        C        = len(expected)

        disc_cols = {
            attr.id: _discretize(
                numeric_cache.get(col, pd.to_numeric(df[col], errors="coerce")),
                attr,
            )
            for attr, col in active
        }
        disc_df  = pd.DataFrame(disc_cols).dropna()
        observed = set(map(tuple, disc_df.values.tolist()))

        covered = expected & observed
        missing = sorted(expected - observed)
        raw_score = len(covered) / C if C else 1.0

        # Theoretical maximum: best achievable for a dataset of size n
        theo_max_count = min(n, C)
        theo_max_score = theo_max_count / C if C else 1.0

        # Normalized: what fraction of the achievable ceiling is reached
        normalized = raw_score / theo_max_score if theo_max_score > 0 else 0.0

        # Random baseline (coupon-collector)
        rand_expected = _coupon_collector_expected(C, n)
        rand_score    = rand_expected / C if C else 0.0

        stages.append(SemanticStageResult(
            stage=stage.stage, attributes=stage.attributes,
            description=stage.description,
            expected_combinations=C, observed_combinations=len(covered),
            score=raw_score,
            theoretical_max=theo_max_count,
            normalized_score=min(normalized, 1.0),   # cap at 100 %
            random_baseline=rand_score,
            sample_missing_combos=missing[:10],
        ))

    # ── B) Sub-class coverage ─────────────────────────────────────────────────
    subclass_results: List[SubclassResult] = []

    for sc in ontology.subclasses:
        rule_col_map = {
            cond.attribute: col_map.get(cond.attribute)
            for cond in sc.rule.conditions
            if col_map.get(cond.attribute)
        }
        mask  = _apply_rule(df, sc.rule, rule_col_map, numeric_cache)  # type: ignore
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
    subclass_cov = (
        sum(1 for r in subclass_results if r.covered) / n_sc if n_sc else 0.0
    )
    adequacy     = (
        sum(1 for r in subclass_results if r.adequate) / n_sc if n_sc else 0.0
    )

    base_score = 0.35 * semantic_cov + 0.45 * subclass_cov + 0.20 * adequacy

    # ── Structural penalty ────────────────────────────────────────────────────
    structural_penalty = 0.0
    if col_cov is not None:
        required_cols_missing = sum(
            1 for r in col_cov.results
            if not r.present and r.required
        )
        critical_sparse = sum(
            1 for r in col_cov.results
            if r.present and (
                r.coverage_score < 0.50 or
                (r.total_valid < 0.80 * n)
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
        n_records=n,   # ← NEW
    )
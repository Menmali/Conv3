"""
ontology_mapper.py  —  Step 5

Answers the question: does this dataset cover the disease ontology well enough
to support reliable ML predictions?

Two measurements:
  A) Semantic combination coverage (progressive) — L3
     Does the dataset have patients for all clinically meaningful
     attribute-combination profiles?  Added one attribute at a time.

  B) Sub-class coverage — L4 + L5
     Do all disease sub-classes (defined by official clinical rules) have
     representative samples?  And are there enough samples per sub-class?


# Weights justified by DREAMER framework (Ahangaran et al., BMC Med Inform 2024):
# subclass presence weighted highest (0.45) because complete absence of a
# patient population is a more severe failure than incomplete combination coverage.
# Adequacy weighted lowest (0.20) because presence of any samples proves
# population exists; statistical insufficiency is a lesser failure than absence.


# Structural penalty — justified by Jarmakovica (Front AI 2025) and
# systematic review on missing EHR data (Health Data Science 2024):
# completeness is a prerequisite dimension; missing data directly degrades
# prediction model reliability independent of semantic coverage.


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


# ── data classes ──────────────────────────────────────────────────────────────

@dataclass
class SemanticStageResult:
    stage:                      int
    attributes:                 List[str]
    description:                str
    expected_combinations:      int
    observed_combinations:      int
    score:                      float
    sample_missing_combos:      List[Tuple]   # first 10 missing


@dataclass
class SubclassResult:
    subclass_id:    str
    label:          str
    short_label:    str
    priority:       str
    clinical_note:  str
    guideline:      str
    covered:        bool
    sample_count:   int
    adequate:       bool
    min_required:   int


@dataclass
class OntologyMappingReport:
    semantic_stages:            List[SemanticStageResult]
    subclass_results:           List[SubclassResult]
    semantic_coverage:          float   # final-stage semantic score
    subclass_coverage:          float   # % sub-classes with ≥ 1 sample
    adequacy_score:             float   # % sub-classes with ≥ min samples
    overall_score:              float   # weighted combination
    verdict:                    str     # 'sufficient'|'borderline'|'insufficient'
    sufficient_threshold:       float
    borderline_threshold:       float


# ── main function ─────────────────────────────────────────────────────────────

def map_to_ontology(
    df: pd.DataFrame,
    ontology: DomainOntology,
    col_cov: ColumnCoverageReport | None = None,
) -> OntologyMappingReport:

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
                score=0.0, sample_missing_combos=[],
            ))
            continue

        ranges   = [attr.expected_discrete_values() for attr, _ in active]
        expected = set(itertools.product(*ranges))

        disc_cols = {
            attr.id: _discretize(numeric_cache.get(col, pd.to_numeric(df[col], errors="coerce")), attr)
            for attr, col in active
        }
        disc_df   = pd.DataFrame(disc_cols).dropna()
        observed  = set(map(tuple, disc_df.values.tolist()))

        covered = expected & observed
        missing = sorted(expected - observed)
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
        rule_col_map = {
            cond.attribute: col_map.get(cond.attribute)
            for cond in sc.rule.conditions
            if col_map.get(cond.attribute)
        }
        mask  = _apply_rule(df, sc.rule, rule_col_map, numeric_cache) # type: ignore
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
            adequate=count >= ontology.min_samples_per_subclass,
            min_required=ontology.min_samples_per_subclass,
        ))

  # ── Scores ────────────────────────────────────────────────────────────────
    semantic_cov = stages[-1].score if stages else 0.0
    n_sc         = len(subclass_results)
    subclass_cov = sum(1 for r in subclass_results if r.covered)  / n_sc if n_sc else 0.0
    adequacy     = sum(1 for r in subclass_results if r.adequate) / n_sc if n_sc else 0.0

    # Weights: subclass presence 45%, semantic 35%, adequacy 20%
    # Justified by DREAMER framework (Ahangaran et al., BMC Med Inform 2024):
    # complete absence of a patient population is a more severe failure
    # than incomplete combination coverage, so subclass presence is weighted highest.
    base_score = 0.35 * semantic_cov + 0.45 * subclass_cov + 0.20 * adequacy

    # ── Structural penalty ────────────────────────────────────────────────────
    # Justified by Jarmakovica (Front AI 2025) and missing EHR data systematic
    # review (Health Data Science 2024): completeness is a prerequisite dimension;
    # missing data degrades prediction reliability independent of semantic coverage.
    structural_penalty = 0.0
    if col_cov is not None:
        required_cols_missing = sum(
            1 for r in col_cov.results
            if not r.present and r.required
        )
        critical_sparse = sum(
            1 for r in col_cov.results
            if r.present and r.coverage_score < 0.50
        )
        # 15% penalty per missing required column, 5% per critically sparse column
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

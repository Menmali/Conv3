"""
ontology_mapper.py  —  Step 5

Answers the question: does this dataset cover the disease ontology well enough
to support reliable ML predictions?

Two measurements:
  A) Semantic combination coverage (progressive)
  B) Sub-class coverage

Overall Dataset Adequacy Score
-------------------------------
A single evidence-based score replacing the previous ambiguous weighted
composite. Built on two published standards:

STEP 1 — Hard Disqualifiers (binary, logical — checked before scoring)
  These represent logical prerequisites for supervised ML training that no
  amount of EPV score can compensate for. If any trigger: verdict =
  'insufficient', score not computed from subclasses.
    - Any required column absent
    - Target variable has missing values
    - No subclass at all is adequately covered (EPV < 20 for every subclass)
      i.e. the dataset has zero usable signal for any clinically defined
      patient profile.

  Soft Flags (recorded and shown, but do NOT force the verdict; the EPV
  score and Step 3 thresholds are what determine sufficiency/borderline
  status for these cases)
    - A critical-priority subclass has 0 samples
    - A subclass has imbalance ratio > 100:1  (He & Garcia, IEEE TKDE 2009)
  These are real, important caveats — they're surfaced to the reader and
  factored into the EPV mean (an empty subclass scores epv_score = 0, which
  already pulls the mean down) — but a dataset that otherwise has multiple
  well-powered subclasses should not be unconditionally graded
  'insufficient' just because one profile is absent or imbalanced.

STEP 2 — EPV Coverage Score  (Ogundimu et al., J Clin Epidemiol 2016)
  For each subclass s with n_s samples:
      epv_score(s) = min(n_s / 20, 1.0)
  Ogundimu et al. demonstrate through simulation that EPV >= 20 eliminates
  coefficient bias for low-prevalence predictors. Below EPV 20, bias scales
  continuously with distance from the threshold — hence the proportional score.

  overall_score = mean(epv_score(s) for s in subclasses)

STEP 3 — Verdict
  score = 1.0        → 'sufficient'    all subclasses meet EPV >= 20
  0.5 <= score < 1.0 → 'borderline'   mean EPV >= 10 (Ogundimu bias boundary)
  score < 0.5        → 'insufficient'  mean EPV < 10 (severe bias region)
  (A hard disqualifier — see Step 1 — overrides this and forces
   'insufficient' regardless of score.)

Diagnostic outputs (retained for Chart 3 — do NOT feed into verdict):
  - semantic_coverage   (final-stage raw semantic score)
  - subclass_coverage   (% subclasses with >= 1 sample)
  - adequacy_score      (% subclasses meeting EPV >= 20)
  - ir_violations       (subclasses flagged by He & Garcia 100:1 rule)
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
    score:                 float
    theoretical_max:       float
    normalized_score:      float
    random_baseline:       float
    sample_missing_combos: List[Tuple]


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
    adequate:       bool        # meets EPV >= 20  (Ogundimu et al. 2016)
    min_required:   int
    epv_score:      float       # min(n_s / 20, 1.0)
    imbalance_ratio: float      # N / n_s  (inf when n_s = 0)
    ir_violation:   bool        # True when imbalance_ratio > 100  (He & Garcia 2009)


@dataclass
class HardDisqualifier:
    reason:      str
    description: str
    hard:        bool = True   # True = forces verdict; False = soft flag, shown only


@dataclass
class OntologyMappingReport:
    semantic_stages:      List[SemanticStageResult]
    subclass_results:     List[SubclassResult]

    # ── Diagnostic scores — Chart 3 only, NOT used in verdict ────────────────
    semantic_coverage:    float   # final-stage raw semantic score
    subclass_coverage:    float   # % subclasses with >= 1 sample
    adequacy_score:       float   # % subclasses meeting EPV >= 20

    # ── Evidence-based overall score — Chart 4 + verdict ─────────────────────
    overall_score:        float   # mean(epv_score) per subclass
    verdict:              str     # 'sufficient' | 'borderline' | 'insufficient'

    # ── Hard disqualifiers ────────────────────────────────────────────────────
    # Contains BOTH hard (verdict-forcing) and soft (caveat-only) entries.
    # Use `disqualified` to check whether a hard one fired; inspect each
    # entry's `.hard` flag to distinguish them for display purposes.
    hard_disqualifiers:   List[HardDisqualifier]
    disqualified:         bool

    # ── Efficiency statement context ──────────────────────────────────────────
    n_records:            int
    n_subclasses_present: int     # subclasses with >= 1 sample
    n_subclasses_total:   int
    n_subclasses_adequate: int    # subclasses meeting EPV >= 20

    # ── Thresholds ────────────────────────────────────────────────────────────
    sufficient_threshold: float   # 1.0
    borderline_threshold: float   # 0.5


# ── main function ─────────────────────────────────────────────────────────────

def map_to_ontology(
    df:       pd.DataFrame,
    ontology: DomainOntology,
    col_cov:  Optional["ColumnCoverageReport"] = None,
) -> OntologyMappingReport:

    N = len(df)

    col_map: Dict[str, Optional[str]] = {
        attr.id: _find_column(df, attr) for attr in ontology.attributes
    }
    numeric_cache: Dict[str, pd.Series] = {
        col: pd.to_numeric(df[col], errors="coerce")
        for col in col_map.values()
        if col and col in df.columns
    }

    # ── STEP 1: Hard disqualifiers (logical prerequisites only) ─────────────
    # These are the only triggers that can force verdict = 'insufficient'
    # outright, independent of the EPV score.
    hard_disqualifiers: List[HardDisqualifier] = []

    # 1a. Required columns absent — hard
    if col_cov is not None:
        for r in col_cov.results:
            if r.required and not r.present:
                hard_disqualifiers.append(HardDisqualifier(
                    reason="Required column absent",
                    description=(
                        f"'{r.attribute_id}' is required by the ontology "
                        f"but absent from the dataset. The model cannot "
                        f"compute predictions without it."
                    ),
                    hard=True,
                ))

    # 1b. Target variable has missing values — hard
    target_attr = next((a for a in ontology.attributes if a.is_target), None)
    if target_attr:
        target_col = col_map.get(target_attr.id)
        if target_col and target_col in df.columns:
            target_missing = int(df[target_col].isna().sum())
            if target_missing > 0:
                hard_disqualifiers.append(HardDisqualifier(
                    reason="Target variable has missing values",
                    description=(
                        f"Target column '{target_col}' has {target_missing} "
                        f"missing values. Supervised training requires a "
                        f"complete target variable."
                    ),
                    hard=True,
                ))

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
                score=0.0, theoretical_max=0.0, normalized_score=0.0,
                random_baseline=0.0, sample_missing_combos=[],
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

        covered   = expected & observed
        missing   = sorted(expected - observed)
        raw_score = len(covered) / C if C else 1.0

        theo_max_count = min(N, C)
        theo_max_score = theo_max_count / C if C else 1.0
        normalized     = raw_score / theo_max_score if theo_max_score > 0 else 0.0

        rand_expected = _coupon_collector_expected(C, N)
        rand_score    = rand_expected / C if C else 0.0

        stages.append(SemanticStageResult(
            stage=stage.stage, attributes=stage.attributes,
            description=stage.description,
            expected_combinations=C, observed_combinations=len(covered),
            score=raw_score,
            theoretical_max=theo_max_count,
            normalized_score=min(normalized, 1.0),
            random_baseline=rand_score,
            sample_missing_combos=missing[:10],
        ))

    # ── B) Sub-class coverage + EPV scores ───────────────────────────────────
    subclass_results: List[SubclassResult] = []
    # Soft flags are tracked separately while looping subclasses, then
    # appended to hard_disqualifiers afterwards (marked hard=False) so they
    # still show up in Chart 4 / terminal output as caveats, but don't
    # participate in the `disqualified` check below.
    soft_flags: List[HardDisqualifier] = []

    for sc in ontology.subclasses:
        rule_col_map = {
            cond.attribute: col_map.get(cond.attribute)
            for cond in sc.rule.conditions
            if col_map.get(cond.attribute)
        }
        mask = _apply_rule(df, sc.rule, rule_col_map, numeric_cache)  # type: ignore
        n_s  = int(mask.sum())

        # EPV score — Ogundimu et al. 2016
        epv_score = min(n_s / 20.0, 1.0)

        # Imbalance ratio — He & Garcia 2009
        imbalance_ratio = (N / n_s) if n_s > 0 else float("inf")
        ir_violation    = imbalance_ratio > 100.0

        # Soft flag: critical subclass with 0 samples.
        # This is a real, important caveat (and the EPV mean already takes
        # the hit, since epv_score = 0 for this subclass) — but it no
        # longer unconditionally forces 'insufficient' on its own. See the
        # "all subclasses inadequate" hard check below for the case where
        # this should still be fatal.
        if n_s == 0 and sc.priority == "critical":
            soft_flags.append(HardDisqualifier(
                reason=f"Critical subclass absent: {sc.short_label}",
                description=(
                    f"'{sc.label}' is a critical-priority subclass with zero "
                    f"samples. A model trained on this data has no signal for "
                    f"this patient population and will fail silently in deployment."
                ),
                hard=False,
            ))

        # Soft flag: imbalance ratio > 100:1.
        if ir_violation and n_s > 0:
            soft_flags.append(HardDisqualifier(
                reason=f"Severe class imbalance: {sc.short_label}",
                description=(
                    f"'{sc.label}' has an imbalance ratio of {imbalance_ratio:.0f}:1 "
                    f"({n_s} samples out of {N}). He & Garcia (2009) demonstrate "
                    f"that classifiers degenerate to majority-class prediction "
                    f"at ratios exceeding 100:1."
                ),
                hard=False,
            ))

        subclass_results.append(SubclassResult(
            subclass_id=sc.id,
            label=sc.label,
            short_label=sc.short_label,
            priority=sc.priority,
            clinical_note=sc.clinical_note,
            guideline=sc.guideline,
            covered=n_s > 0,
            sample_count=n_s,
            adequate=n_s >= sc.min_required_samples,
            min_required=ontology.min_samples_per_subclass,
            epv_score=epv_score,
            imbalance_ratio=imbalance_ratio,
            ir_violation=ir_violation,
        ))

    # ── Diagnostic scores (Chart 3) ───────────────────────────────────────────
    semantic_cov = stages[-1].score if stages else 0.0
    n_sc         = len(subclass_results)
    subclass_cov = (
        sum(1 for r in subclass_results if r.covered) / n_sc if n_sc else 0.0
    )
    adequacy = (
        sum(1 for r in subclass_results if r.adequate) / n_sc if n_sc else 0.0
    )
    n_adequate = sum(1 for r in subclass_results if r.adequate)

    # 1c. Hard disqualifier: NO subclass at all is adequately covered.
    # This is the one case where "the dataset has nothing usable" really is
    # a logical prerequisite failure, not just a coverage gap — so it stays
    # hard. A dataset with at least one well-powered subclass should not
    # hit this.
    if n_sc > 0 and n_adequate == 0:
        hard_disqualifiers.append(HardDisqualifier(
            reason="No subclass meets the adequacy threshold",
            description=(
                f"None of the {n_sc} clinically defined subclasses reach "
                f"EPV >= 20. The dataset provides no statistically reliable "
                f"signal for any patient profile in the ontology."
            ),
            hard=True,
        ))

    # Soft flags are recorded for display but kept out of the hard list
    # used for the disqualification check.
    hard_disqualifiers.extend(soft_flags)

    # ── STEP 2: EPV Coverage Score (Ogundimu et al. 2016) ────────────────────
    overall = (
        float(np.mean([r.epv_score for r in subclass_results]))
        if subclass_results else 0.0
    )

    # ── STEP 3: Verdict ───────────────────────────────────────────────────────
    # Only entries marked hard=True can force 'insufficient' outright.
    disqualified = any(d.hard for d in hard_disqualifiers)

    if disqualified:
        verdict = "insufficient"
    elif overall >= 1.0:
        verdict = "sufficient"
    elif overall >= 0.5:
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
        hard_disqualifiers=hard_disqualifiers,
        disqualified=disqualified,
        n_records=N,
        n_subclasses_present=sum(1 for r in subclass_results if r.covered),
        n_subclasses_total=n_sc,
        n_subclasses_adequate=n_adequate,
        sufficient_threshold=1.0,
        borderline_threshold=0.5,
    )

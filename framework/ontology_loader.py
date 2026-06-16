"""
ontology_loader.py
Parse a domain ontology JSON file into clean Python dataclasses.
All downstream code works with these typed objects, never raw dicts.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ── Value-level structures ────────────────────────────────────────────────────

@dataclass
class CoverageBin:
    """One clinical range for a continuous attribute (e.g. Stage 2 HTN: 140–999)."""
    label: str
    min: float
    max: float
    note: str = ""


@dataclass
class CategoricalValue:
    """
    One defined code for a categorical / ordinal attribute.

    value_aliases  — additional string representations that should map to this
                     canonical code.  The discretiser checks them case-insensitively
                     after the primary float-equality comparison fails.

    Example JSON:
        {"code": 0, "label": "female",
         "value_aliases": ["female", "f", "woman", "Female", "F", "0"]}
    """
    code: Any                          # int or float — whatever is in the CSV
    label: str
    note: str = ""
    value_aliases: List[str] = field(default_factory=list)


# ── Attribute ─────────────────────────────────────────────────────────────────

@dataclass
class OntologyAttribute:
    id: str
    label: str
    dataset_column: str            # primary expected column name in the CSV
    aliases: List[str]             # fallback column *names* to try
    standard: str                  # LOINC / SNOMED CT / ICD-11
    code: str
    type: str                      # 'continuous' | 'categorical' | 'ordinal'
    required: bool
    missing_threshold: float       # max acceptable missing rate (0.05 = 5%)
    coverage_bins: List[CoverageBin] = field(default_factory=list)
    values: List[CategoricalValue] = field(default_factory=list)
    is_target: bool = False
    unit: str = ""
    guideline_note: str = ""

    def get_all_column_candidates(self) -> List[str]:
        """Return all column names to try when searching a DataFrame."""
        seen = []
        for name in [self.dataset_column] + self.aliases:
            if name not in seen:
                seen.append(name)
        return seen

    def expected_discrete_values(self) -> List[str]:
        """Return the complete set of expected discrete labels for coverage checks."""
        if self.type in ("categorical", "ordinal"):
            return [str(v.code) for v in self.values]
        elif self.type == "continuous" and self.coverage_bins:
            return [b.label for b in self.coverage_bins]
        return []


# ── Sub-class rule ─────────────────────────────────────────────────────────────

@dataclass
class RuleCondition:
    attribute: str    # ontology attribute id (e.g. 'trestbps')
    op: str           # '>=', '<=', '>', '<', '==', '!='
    value: Any        # numeric threshold


@dataclass
class ClassificationRule:
    logic: str                       # 'AND' | 'OR'
    conditions: List[RuleCondition]


@dataclass
class OntologySubclass:
    id: str
    label: str
    short_label: str
    priority: str                    # 'critical' | 'high' | 'medium' | 'low'
    guideline: str
    clinical_note: str
    key_attributes: List[str]
    min_required_samples: int
    rule: ClassificationRule


# ── Semantic progression stage ────────────────────────────────────────────────

@dataclass
class SemanticStage:
    stage: int
    attributes: List[str]            # list of ontology attribute ids
    description: str


# ── Top-level ontology ────────────────────────────────────────────────────────

@dataclass
class DomainOntology:
    domain: str
    full_name: str
    icd_codes: List[str]
    guidelines: Dict[str, str]
    description: str
    attributes: List[OntologyAttribute]
    subclasses: List[OntologySubclass]
    semantic_progression: List[SemanticStage]
    sufficient_threshold: float
    borderline_threshold: float
    min_samples_per_subclass: int

    # ── Convenience lookups ───────────────────────────────────────────────────

    def get_attribute(self, attr_id: str) -> Optional[OntologyAttribute]:
        return next((a for a in self.attributes if a.id == attr_id), None)

    def required_attributes(self) -> List[OntologyAttribute]:
        return [a for a in self.attributes if a.required]

    def optional_attributes(self) -> List[OntologyAttribute]:
        return [a for a in self.attributes if not a.required]


# ── Parser ────────────────────────────────────────────────────────────────────

def load_ontology(path: str) -> DomainOntology:
    """Load a JSON ontology file and return a fully typed DomainOntology."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data["meta"]

    # ── Attributes ────────────────────────────────────────────────────────────
    attributes: List[OntologyAttribute] = []
    for a in data["attributes"]:
        bins = [
            CoverageBin(
                label=b["label"],
                min=float(b["min"]),
                max=float(b["max"]),
                note=b.get("note", "")
            )
            for b in a.get("coverage_bins", [])
        ]
        values = [
            CategoricalValue(
                code=v["code"],
                label=v["label"],
                note=v.get("note", ""),
                value_aliases=v.get("value_aliases", [])   # ← NEW
            )
            for v in a.get("values", [])
        ]
        attributes.append(OntologyAttribute(
            id=a["id"],
            label=a["label"],
            dataset_column=a["dataset_column"],
            aliases=a.get("aliases", []),
            standard=a["standard"],
            code=a["code"],
            type=a["type"],
            required=a["required"],
            missing_threshold=float(a["missing_threshold"]),
            coverage_bins=bins,
            values=values,
            is_target=a.get("is_target", False),
            unit=a.get("unit", ""),
            guideline_note=a.get("guideline_note", "")
        ))

    # ── Sub-classes ───────────────────────────────────────────────────────────
    subclasses: List[OntologySubclass] = []
    for sc in data["subclasses"]:
        conditions = [
            RuleCondition(
                attribute=c["attribute"],
                op=c["op"],
                value=c["value"]
            )
            for c in sc["rule"]["conditions"]
        ]
        rule = ClassificationRule(
            logic=sc["rule"]["logic"],
            conditions=conditions
        )
        subclasses.append(OntologySubclass(
            id=sc["id"],
            label=sc["label"],
            short_label=sc["short_label"],
            priority=sc["priority"],
            guideline=sc["guideline"],
            clinical_note=sc["clinical_note"],
            key_attributes=sc["key_attributes"],
            min_required_samples=sc["min_required_samples"],
            rule=rule
        ))

    # ── Semantic stages ───────────────────────────────────────────────────────
    stages = [
        SemanticStage(
            stage=s["stage"],
            attributes=s["attributes"],
            description=s["description"]
        )
        for s in data["semantic_progression"]
    ]

    thresh = data["thresholds"]

    return DomainOntology(
        domain=meta["domain"],
        full_name=meta["full_name"],
        icd_codes=meta["icd_codes"],
        guidelines=meta.get("guidelines", {}),
        description=meta["description"],
        attributes=attributes,
        subclasses=subclasses,
        semantic_progression=stages,
        sufficient_threshold=float(thresh["sufficient"]),
        borderline_threshold=float(thresh["borderline"]),
        min_samples_per_subclass=int(thresh["min_samples_per_subclass"])
    )
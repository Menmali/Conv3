"""
owl_exporter.py  —  Step 6

Two responsibilities:
  1. export_owl()          — builds and saves the OWL/XML file from pipeline results.
  2. validate_owl()        — loads the saved OWL, runs four SPARQL rule families
                             against the dataset. The OWL is the source of truth.
  3. print_owl_validation() — prints the validation report to the terminal.

Four SPARQL rule families:
  R1  Required columns must exist in the dataset.
  R2  Continuous values must fall within at least one OWL-defined bin.
  R3  Categorical values must match only OWL-defined codes.
  R4  Subclass sample counts must meet the OWL minRequiredSamples annotation.

All SPARQL results go through _sparql_rows() which returns List[List[str]].
This gives Pylance a concrete type and eliminates all indexing errors.
"""

import os
import re
from dataclasses import dataclass, field
from typing import List

import pandas as pd
from rdflib import Graph, Namespace, RDF, RDFS, OWL, Literal, XSD
from rdflib.query import Result

from .ontology_loader import DomainOntology
from .column_coverage import ColumnCoverageReport
from .ontology_mapper import OntologyMappingReport


EX_URI = "http://example.org/heartdisease#"


# ═══════════════════════════════════════════════════════════════════════════════
#  Private helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _sparql_rows(g: Graph, query: str) -> List[List[str]]:
    """
    Execute a SELECT SPARQL query and return results as List[List[str]].
    Every cell is cast to str, so downstream code works with plain strings
    and Pylance sees a fully concrete type — no _TripleType / bool ambiguity.
    """
    result: Result = g.query(query)
    rows: List[List[str]] = []
    for row in result:
        rows.append([str(cell) for cell in row])   # type: ignore[union-attr]
    return rows


def _class_name(label: str) -> str:
    """'Resting Blood Pressure (Systolic)' → 'RestingBloodPressureSystolic'"""
    return re.sub(r"[^a-zA-Z0-9]", "", label.title().replace(" ", ""))


def _norm_code(code: str) -> str:
    """Normalise '3.0' and '3' to the same string '3' for set comparison."""
    try:
        f = float(code)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return code


# ═══════════════════════════════════════════════════════════════════════════════
#  Data classes
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class OWLViolation:
    rule:          str
    severity:      str          # 'critical' | 'warning'
    column:        str
    description:   str
    affected_rows: int
    examples:      List[str] = field(default_factory=list)


@dataclass
class OWLValidationReport:
    owl_path:      str
    total_triples: int
    violations:    List[OWLViolation]
    passed_rules:  List[str]

    @property
    def critical(self) -> List[OWLViolation]:
        return [v for v in self.violations if v.severity == "critical"]

    @property
    def warnings(self) -> List[OWLViolation]:
        return [v for v in self.violations if v.severity == "warning"]

    @property
    def is_conformant(self) -> bool:
        return len(self.critical) == 0


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 6a — Export
# ═══════════════════════════════════════════════════════════════════════════════

def export_owl(
    ontology:   DomainOntology,
    col_cov:    ColumnCoverageReport,
    mapping:    OntologyMappingReport,
    output_dir: str = "outputs",
) -> str:
    """
    Build and serialise a full OWL/XML ontology from pipeline results.
    Returns the path to the saved .owl file.
    """
    g  = Graph()
    EX = Namespace(EX_URI)
    g.bind("ex",   EX)
    g.bind("owl",  OWL)
    g.bind("rdfs", RDFS)
    g.bind("xsd",  XSD)

    # ── Ontology header ───────────────────────────────────────────────────────
    ont = EX["HeartDiseaseOntology"]
    g.add((ont, RDF.type,     OWL.Ontology))
    g.add((ont, RDFS.label,   Literal(ontology.full_name)))
    g.add((ont, RDFS.comment, Literal(
        f"Auto-generated OWL ontology for {ontology.full_name}. "
        f"ICD-11 codes: {', '.join(ontology.icd_codes)}."
    )))

    # ── Top-level classes ─────────────────────────────────────────────────────
    for cls_name in [
        "Patient", "RiskFactor", "MedicalTest", "Diagnosis",
        "AgeGroup", "CholesterolLevel", "BloodPressureLevel",
        "ClinicalFinding", "SubclassProfile",
    ]:
        g.add((EX[cls_name], RDF.type,   OWL.Class))
        g.add((EX[cls_name], RDFS.label, Literal(cls_name)))

    g.add((EX.Patient, RDFS.comment,
           Literal("A patient whose cardiovascular health data is recorded.")))

    # ── Attribute parent mapping ───────────────────────────────────────────────
    _PARENT = {
        "age":      "AgeGroup",
        "chol":     "CholesterolLevel",
        "trestbps": "BloodPressureLevel",
        "cp":       "RiskFactor",
        "sex":      "RiskFactor",
        "fbs":      "RiskFactor",
        "exang":    "RiskFactor",
        "restecg":  "MedicalTest",
        "slope":    "MedicalTest",
        "thal":     "MedicalTest",
        "ca":       "ClinicalFinding",
        "oldpeak":  "ClinicalFinding",
        "thalach":  "ClinicalFinding",
        "num":      "Diagnosis",
    }

    for attr in ontology.attributes:
        if attr.dataset_column is None:
            continue

        parent_name = _PARENT.get(attr.id)
        if parent_name is None:
            print(f"  ⚠️  owl_exporter: attribute '{attr.id}' not in parent map "
                f"— defaulting to ClinicalFinding")
            parent_name = "ClinicalFinding"
        parent = EX[parent_name]
        attr_cls = EX[_class_name(attr.label)]

        g.add((attr_cls, RDF.type,         OWL.Class))
        g.add((attr_cls, RDFS.subClassOf,  parent))
        g.add((attr_cls, RDFS.label,       Literal(attr.label)))
        g.add((attr_cls, RDFS.comment,     Literal(
            f"Standard: {attr.standard} | Code: {attr.code}"
            + (f" | Unit: {attr.unit}" if attr.unit else "")
        )))
        # These two triples are read back by R1 SPARQL query
        g.add((attr_cls, EX.datasetColumn, Literal(attr.dataset_column, datatype=XSD.string)))
        g.add((attr_cls, EX.isRequired,    Literal(attr.required,       datatype=XSD.boolean)))

        if attr.type in ("categorical", "ordinal"):
            for v in attr.values:
                ind = EX[f"{attr.id}_{v.label}"]
                g.add((ind, RDF.type,   attr_cls))
                g.add((ind, RDF.type,   OWL.NamedIndividual))
                g.add((ind, RDFS.label, Literal(v.label)))
                # ex:code is read back by R3 SPARQL query
                g.add((ind, EX.code,    Literal(str(v.code), datatype=XSD.string)))
                if v.note:
                    g.add((ind, RDFS.comment, Literal(v.note)))

        elif attr.type == "continuous" and attr.coverage_bins:
            for b in attr.coverage_bins:
                ind = EX[f"{attr.id}_{b.label}"]
                g.add((ind, RDF.type,    attr_cls))
                g.add((ind, RDF.type,    OWL.NamedIndividual))
                g.add((ind, RDFS.label,  Literal(b.label)))
                # ex:rangeMin / ex:rangeMax are read back by R2 SPARQL query
                g.add((ind, EX.rangeMin, Literal(b.min, datatype=XSD.float)))
                g.add((ind, EX.rangeMax, Literal(b.max, datatype=XSD.float)))
                if b.note:
                    g.add((ind, RDFS.comment, Literal(b.note)))

    # ── Disease subclass profiles ─────────────────────────────────────────────
    for sc in ontology.subclasses:
        sc_uri = EX[sc.id]
        g.add((sc_uri, RDF.type,              OWL.Class))
        g.add((sc_uri, RDFS.subClassOf,       EX.SubclassProfile))
        g.add((sc_uri, RDFS.label,            Literal(sc.label)))
        g.add((sc_uri, RDFS.comment,          Literal(sc.clinical_note)))
        g.add((sc_uri, EX.priority,           Literal(sc.priority,  datatype=XSD.string)))
        g.add((sc_uri, EX.guideline,          Literal(sc.guideline, datatype=XSD.string)))
        # ex:minRequiredSamples is compared to ex:sampleCount by R4 SPARQL FILTER
        g.add((sc_uri, EX.minRequiredSamples,
               Literal(sc.min_required_samples, datatype=XSD.integer)))

        for r in mapping.subclass_results:
            if r.subclass_id == sc.id:
                g.add((sc_uri, EX.sampleCount,
                       Literal(r.sample_count, datatype=XSD.integer)))
                g.add((sc_uri, EX.isCovered,
                       Literal(r.covered,      datatype=XSD.boolean)))
                g.add((sc_uri, EX.isStatisticallyAdequate,
                       Literal(r.adequate,     datatype=XSD.boolean)))

    # ── Data properties ───────────────────────────────────────────────────────
    for attr in ontology.attributes:
        prop      = EX[attr.id]
        rng       = XSD.float if attr.type == "continuous" else XSD.integer
        code_pred = EX.loincCode if attr.standard == "LOINC" else EX.snomedCode
        g.add((prop, RDF.type,    OWL.DatatypeProperty))
        g.add((prop, RDFS.domain, EX.Patient))
        g.add((prop, RDFS.range,  rng))
        g.add((prop, RDFS.label,  Literal(attr.label)))
        g.add((prop, code_pred,   Literal(attr.code, datatype=XSD.string)))
        if attr.guideline_note:
            g.add((prop, RDFS.comment, Literal(attr.guideline_note)))

    # ── Object properties ─────────────────────────────────────────────────────
    for name, range_cls in [
        ("hasRiskFactor",         EX.RiskFactor),
        ("hasMedicalTest",        EX.MedicalTest),
        ("hasDiagnosis",          EX.Diagnosis),
        ("hasAgeGroup",           EX.AgeGroup),
        ("hasCholesterolLevel",   EX.CholesterolLevel),
        ("hasBloodPressureLevel", EX.BloodPressureLevel),
        ("hasClinicalFinding",    EX.ClinicalFinding),
        ("belongsToSubclass",     EX.SubclassProfile),
    ]:
        g.add((EX[name], RDF.type,    OWL.ObjectProperty))
        g.add((EX[name], RDFS.domain, EX.Patient))
        g.add((EX[name], RDFS.range,  range_cls))
        g.add((EX[name], RDFS.label,  Literal(name)))

    # ── Coverage score annotations ────────────────────────────────────────────
    for r in col_cov.results:
        if r.present:
            g.add((EX[r.attribute_id], EX.coverageScore,
                   Literal(round(r.coverage_score, 4), datatype=XSD.float)))

    g.add((ont, EX.overallMappingScore,
           Literal(round(mapping.overall_score,     4), datatype=XSD.float)))
    g.add((ont, EX.mappingVerdict,
           Literal(mapping.verdict,                    datatype=XSD.string)))
    g.add((ont, EX.semanticCoverage,
           Literal(round(mapping.semantic_coverage, 4), datatype=XSD.float)))
    g.add((ont, EX.subclassCoverage,
           Literal(round(mapping.subclass_coverage, 4), datatype=XSD.float)))

    # ── Serialise ─────────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    domain = getattr(ontology, 'meta', None)
    if domain is None:
        domain_name = "ontology"
    else:
        domain_name = getattr(domain, 'domain', "unknown")
    out_path = os.path.join(output_dir, f"{domain_name}_ontology.owl")
    g.serialize(destination=out_path, format="xml")
    return out_path


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 6b — SPARQL validation
# ═══════════════════════════════════════════════════════════════════════════════

def validate_owl(
    owl_path: str,
    df:       pd.DataFrame,
    ontology: DomainOntology,
) -> OWLValidationReport:
    """
    Load the saved OWL file, run four SPARQL rule families against the
    dataset, and return an OWLValidationReport.

    R1: Required columns must exist in the dataset.
    R2: Continuous values must fall within OWL-defined bins.
    R3: Categorical values must match OWL-defined codes.
    R4: Subclass sample counts must meet minRequiredSamples.
    """
    g = Graph()
    g.parse(owl_path, format="xml")

    violations:   List[OWLViolation] = []
    passed_rules: List[str]          = []

    # ── R1: Required columns must exist in the dataset ────────────────────────
    r1_rows = _sparql_rows(g, """
        PREFIX ex:   <http://example.org/heartdisease#>
        PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?label ?col WHERE {
            ?cls ex:isRequired    "true"^^xsd:boolean ;
                 ex:datasetColumn  ?col ;
                 rdfs:label        ?label .
        }
    """)
    for r in r1_rows:
        label: str = r[0]
        col:   str = r[1]
        if col not in df.columns:
            violations.append(OWLViolation(
                rule="R1 — Required column presence",
                severity="critical",
                column=col,
                description=(
                    f"OWL declares '{label}' (column '{col}') as required "
                    f"but it is absent from the dataset."
                ),
                affected_rows=len(df),
            ))
        else:
            passed_rules.append(f"R1 ✓  '{col}' — required column present")

    # ── R2: Continuous values within OWL-defined bins ────────────────────────
    r2_rows = _sparql_rows(g, """
        PREFIX ex:  <http://example.org/heartdisease#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        SELECT ?col ?lo ?hi WHERE {
            ?ind rdf:type         ?cls ;
                 ex:rangeMin      ?lo  ;
                 ex:rangeMax      ?hi  .
            ?cls ex:datasetColumn ?col .
        }
    """)
    bins_by_col: dict = {}
    for r in r2_rows:
        bins_by_col.setdefault(r[0], []).append((float(r[1]), float(r[2])))

    for col, bins in bins_by_col.items():
        # Column missing → critical violation
        if col not in df.columns:
            violations.append(OWLViolation(
                rule="R2 — Continuous range conformance",
                severity="critical",
                column=col,
                description=(
                    f"Column '{col}' expected for range validation but is absent."
                ),
                affected_rows=0,
            ))
            continue

        numeric = pd.to_numeric(df[col], errors="coerce").dropna()
        if numeric.empty:
            # Column exists but no numeric data → warning
            violations.append(OWLViolation(
                rule="R2 — Continuous range conformance",
                severity="warning",
                column=col,
                description=(
                    f"Column '{col}' exists but has no numeric values. "
                    f"Range conformance cannot be verified."
                ),
                affected_rows=0,
            ))
            continue

        global_lo = min(b[0] for b in bins)
        global_hi = max(b[1] for b in bins)
        out_of_range = numeric[(numeric < global_lo) | (numeric > global_hi)]
        if len(out_of_range) > 0:
            violations.append(OWLViolation(
                rule="R2 — Continuous range conformance",
                severity="warning",
                column=col,
                description=(
                    f"OWL defines range [{global_lo}, {global_hi}] for '{col}'. "
                    f"{len(out_of_range)} value(s) outside all bins."
                ),
                affected_rows=len(out_of_range),
                examples=[str(x) for x in sorted(out_of_range.unique().tolist())[:5]],
            ))
        else:
            passed_rules.append(
                f"R2 ✓  '{col}' — all {len(numeric)} values within OWL-defined bins"
            )

    # ── R3: Categorical values match only OWL-defined codes ──────────────────
    r3_rows = _sparql_rows(g, """
        PREFIX ex:  <http://example.org/heartdisease#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        SELECT ?col ?code WHERE {
            ?ind rdf:type          ?cls ;
                 ex:code           ?code .
            ?cls ex:datasetColumn  ?col .
        }
    """)
    codes_by_col: dict = {}
    for r in r3_rows:
        codes_by_col.setdefault(r[0], set()).add(r[1])

    for col, raw_valid in codes_by_col.items():
        if col not in df.columns:
            violations.append(OWLViolation(
                rule="R3 — Categorical code conformance",
                severity="critical",
                column=col,
                description=(
                    f"Column '{col}' expected for categorical validation but is absent."
                ),
                affected_rows=0,
            ))
            continue

        valid_norm = {_norm_code(c) for c in raw_valid}
        numeric = pd.to_numeric(df[col], errors="coerce").dropna()
        if numeric.empty:
            violations.append(OWLViolation(
                rule="R3 — Categorical code conformance",
                severity="warning",
                column=col,
                description=(
                    f"Column '{col}' exists but has no categorical values. "
                    f"Code conformance cannot be verified."
                ),
                affected_rows=0,
            ))
            continue

        obs_norm = {_norm_code(str(v)) for v in numeric}
        unexpected = obs_norm - valid_norm
        if unexpected:
            bad_count = int(numeric.apply(
                lambda v, u=unexpected: _norm_code(str(v)) in u
            ).sum())
            violations.append(OWLViolation(
                rule="R3 — Categorical code conformance",
                severity="critical",
                column=col,
                description=(
                    f"OWL defines valid codes {sorted(valid_norm)} for '{col}'. "
                    f"Found {bad_count} row(s) with undefined code(s): "
                    f"{sorted(unexpected)}."
                ),
                affected_rows=bad_count,
                examples=sorted(unexpected)[:5],
            ))
        else:
            passed_rules.append(
                f"R3 ✓  '{col}' — all values match OWL-defined codes"
            )

    # ── R4: Subclass sample counts vs minRequiredSamples ─────────────────────
    r4_rows = _sparql_rows(g, """
        PREFIX ex:   <http://example.org/heartdisease#>
        PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?label ?count ?required WHERE {
            ?sc rdfs:subClassOf       ex:SubclassProfile ;
                rdfs:label             ?label ;
                ex:sampleCount         ?count ;
                ex:minRequiredSamples  ?required .
            FILTER (xsd:integer(?count) < xsd:integer(?required))
        }
    """)
    for r in r4_rows:
        label:    str = r[0]
        count:    int = int(r[1])
        required: int = int(r[2])
        violations.append(OWLViolation(
            rule="R4 — Subclass statistical adequacy",
            severity="critical" if count == 0 else "warning",
            column="(subclass)",
            description=(
                f"OWL declares '{label}' needs ≥ {required} samples. "
                f"Dataset has {count}."
            ),
            affected_rows=count,
        ))
    if not r4_rows:
        passed_rules.append(
            "R4 ✓  All subclasses meet minRequiredSamples threshold"
        )

    return OWLValidationReport(
        owl_path=owl_path,
        total_triples=len(g),
        violations=violations,
        passed_rules=passed_rules,
    )

# ═══════════════════════════════════════════════════════════════════════════════
#  Terminal report
# ═══════════════════════════════════════════════════════════════════════════════

def print_owl_validation(report: OWLValidationReport) -> None:
    W = "═" * 72
    print(f"\n{W}")
    print("  STEP 6 — OWL VALIDATION  (SPARQL queries against dataset)")
    print(W)
    print(f"  OWL file     : {report.owl_path}")
    print(f"  Total triples: {report.total_triples:,}")
    conformant_str = (
        "✅  YES — no critical violations"
        if report.is_conformant
        else f"❌  NO — {len(report.critical)} critical violation(s)"
    )
    print(f"  Conformant   : {conformant_str}\n")

    if report.passed_rules:
        print(f"  Passed ({len(report.passed_rules)}):")
        for r in report.passed_rules:
            print(f"    {r}")

    if report.violations:
        print(f"\n  Violations ({len(report.violations)}):")
        print("  " + "─" * 70)
        for v in report.violations:
            icon = "❌" if v.severity == "critical" else "⚠️ "
            print(f"\n  {icon} [{v.rule}]")
            print(f"       Column      : {v.column}")
            print(f"       Affected    : {v.affected_rows} row(s)")
            print(f"       Description : {v.description}")
            if v.examples:
                print(f"       Examples    : {v.examples}")
    else:
        print("\n  ✅  No violations — dataset fully conforms to OWL ontology.")

    print(f"\n{W}\n")
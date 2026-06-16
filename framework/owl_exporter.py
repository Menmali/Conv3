"""
owl_exporter.py  —  Step 6

Three-file OWL export:

  Stage 1  (schema)      — Patient class + all attribute classes, their
                           bins / categorical value individuals, and
                           data/object properties.  No subclass profiles,
                           no coverage results.
                           Saved as {domain}_ontology_stage1_schema.owl

  Stage 2a (diagnoses)   — Diagnosis class hierarchy only.
                           One OWL class per disease subclass,
                           each subClassOf Diagnosis, carrying:
                           sampleCount / isCovered / isStatisticallyAdequate /
                           minRequiredSamples / coverageVerdict.
                           One named individual per subclass.
                           Loads in WebVOWL as the subclass hierarchy.
                           Saved as {domain}_ontology_diagnoses.owl

  Stage 2b (combinations)— SemanticCoverage class hierarchy only.
                           One OWL class per semantic progression stage,
                           each subClassOf SemanticCoverage, carrying:
                           expectedCombinations / observedCombinations /
                           rawCoverageScore / theoreticalMaxScore /
                           stageAttributes / up to 20 missing combinations.
                           One named individual per stage.
                           Saved as {domain}_ontology_combinations.owl

SPARQL validation (validate_owl) runs against the diagnoses file for R4
and directly against the dataset + schema for R1/R2/R3 — unchanged logic.
"""

import os
import re
from dataclasses import dataclass, field
from typing import List

import pandas as pd
from rdflib import (
    BNode, Graph, Namespace, RDF, RDFS, OWL, Literal, XSD,
)
from rdflib.query import Result

from .ontology_loader import DomainOntology
from .column_coverage import ColumnCoverageReport
from .ontology_mapper import OntologyMappingReport


EX_URI = "http://example.org/heartdisease#"


# ═══════════════════════════════════════════════════════════════════════════════
#  Private helpers  (unchanged from original)
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
    return re.sub(r"[^a-zA-Z0-9]", "", label.title().replace(" ", ""))


def _norm_code(code: str) -> str:
    try:
        f = float(code)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return code


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


# ═══════════════════════════════════════════════════════════════════════════════
#  Data classes  (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class OWLViolation:
    rule:          str
    severity:      str
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
#  Shared graph-building helpers  (unchanged from original)
# ═══════════════════════════════════════════════════════════════════════════════

def _add_top_level_classes(g: Graph, EX: Namespace) -> None:
    for cls_name in [
        "Patient", "RiskFactor", "MedicalTest", "Diagnosis",
        "AgeGroup", "CholesterolLevel", "BloodPressureLevel",
        "ClinicalFinding",
    ]:
        g.add((EX[cls_name], RDF.type,   OWL.Class))
        g.add((EX[cls_name], RDFS.label, Literal(cls_name)))
    g.add((EX.Patient, RDFS.comment,
           Literal("A patient whose cardiovascular health data is recorded.")))


def _add_attribute_classes(g: Graph, EX: Namespace,
                           ontology: DomainOntology) -> None:
    for attr in ontology.attributes:
        if attr.dataset_column is None:
            continue
        parent_name = _PARENT.get(attr.id, "ClinicalFinding")
        attr_cls    = EX[_class_name(attr.label)]
        g.add((attr_cls, RDF.type,         OWL.Class))
        g.add((attr_cls, RDFS.subClassOf,  EX[parent_name]))
        g.add((attr_cls, RDFS.label,       Literal(attr.label)))
        g.add((attr_cls, RDFS.comment,     Literal(
            f"Standard: {attr.standard} | Code: {attr.code}"
            + (f" | Unit: {attr.unit}" if attr.unit else "")
        )))
        g.add((attr_cls, EX.datasetColumn,
               Literal(attr.dataset_column, datatype=XSD.string)))
        g.add((attr_cls, EX.isRequired,
               Literal(attr.required, datatype=XSD.boolean)))

        if attr.type in ("categorical", "ordinal"):
            for v in attr.values:
                ind = EX[f"{attr.id}_{v.label}"]
                g.add((ind, RDF.type,   attr_cls))
                g.add((ind, RDF.type,   OWL.NamedIndividual))
                g.add((ind, RDFS.label, Literal(v.label)))
                g.add((ind, EX.code,    Literal(str(v.code), datatype=XSD.string)))
                if v.note:
                    g.add((ind, RDFS.comment, Literal(v.note)))

        elif attr.type == "continuous" and attr.coverage_bins:
            for b in attr.coverage_bins:
                ind = EX[f"{attr.id}_{b.label}"]
                g.add((ind, RDF.type,    attr_cls))
                g.add((ind, RDF.type,    OWL.NamedIndividual))
                g.add((ind, RDFS.label,  Literal(b.label)))
                g.add((ind, EX.rangeMin, Literal(b.min, datatype=XSD.float)))
                g.add((ind, EX.rangeMax, Literal(b.max, datatype=XSD.float)))
                if b.note:
                    g.add((ind, RDFS.comment, Literal(b.note)))


def _add_data_properties(g: Graph, EX: Namespace,
                         ontology: DomainOntology) -> None:
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


def _add_object_properties(g: Graph, EX: Namespace) -> None:
    for name, range_cls in [
        ("hasRiskFactor",         EX.RiskFactor),
        ("hasMedicalTest",        EX.MedicalTest),
        ("hasDiagnosis",          EX.Diagnosis),
        ("hasAgeGroup",           EX.AgeGroup),
        ("hasCholesterolLevel",   EX.CholesterolLevel),
        ("hasBloodPressureLevel", EX.BloodPressureLevel),
        ("hasClinicalFinding",    EX.ClinicalFinding),
    ]:
        g.add((EX[name], RDF.type,    OWL.ObjectProperty))
        g.add((EX[name], RDFS.domain, EX.Patient))
        g.add((EX[name], RDFS.range,  range_cls))
        g.add((EX[name], RDFS.label,  Literal(name)))


# ═══════════════════════════════════════════════════════════════════════════════
#  Stage 1 — Schema OWL  (unchanged from original)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_stage1(ontology: DomainOntology) -> Graph:
    g  = Graph()
    EX = Namespace(EX_URI)
    g.bind("ex",   EX)
    g.bind("owl",  OWL)
    g.bind("rdfs", RDFS)
    g.bind("xsd",  XSD)

    ont = EX["HeartDiseaseOntology_Stage1"]
    g.add((ont, RDF.type,     OWL.Ontology))
    g.add((ont, RDFS.label,   Literal(f"{ontology.full_name} — Stage 1: Schema")))
    g.add((ont, RDFS.comment, Literal(
        "Stage 1 schema-level OWL ontology for OntoCov framework. "
        "Contains the Patient class, all clinical attribute classes with "
        "their value bins and categorical individuals, and data/object "
        "properties. Disease subclass profiles and coverage results are "
        "in the separate diagnoses and combinations files."
    )))
    g.add((ont, EX.stage, Literal("1 — Schema", datatype=XSD.string)))

    _add_top_level_classes(g, EX)
    _add_attribute_classes(g, EX, ontology)
    _add_data_properties(g, EX, ontology)
    _add_object_properties(g, EX)

    return g


# ═══════════════════════════════════════════════════════════════════════════════
#  Stage 2a — Diagnoses OWL  (NEW — replaces the subclass part of _build_stage2)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_diagnoses(
    ontology: DomainOntology,
    mapping:  OntologyMappingReport,
) -> Graph:
    """
    Diagnosis class hierarchy only.
    One OWL class per disease subclass, each subClassOf Diagnosis.
    Pipeline results attached as annotation properties.
    One named individual per subclass.
    Produces the WebVOWL picture: Diagnosis (red) with coloured subclass nodes.
    """
    g  = Graph()
    EX = Namespace(EX_URI)
    g.bind("ex",   EX)
    g.bind("owl",  OWL)
    g.bind("rdfs", RDFS)
    g.bind("xsd",  XSD)

    # Ontology header
    ont = EX["HeartDiseaseOntology_Diagnoses"]
    g.add((ont, RDF.type,     OWL.Ontology))
    g.add((ont, RDFS.label,
           Literal(f"{ontology.full_name} — Stage 2a: Diagnoses")))
    g.add((ont, RDFS.comment, Literal(
        "Diagnosis coverage layer. Contains the Diagnosis class hierarchy "
        "with one OWL class per disease subclass annotated with pipeline "
        "results: sample count, coverage verdict, statistical adequacy. "
        "Load in WebVOWL to see the subclass hierarchy."
    )))
    g.add((ont, EX.overallMappingScore,
           Literal(round(mapping.overall_score, 4),     datatype=XSD.float)))
    g.add((ont, EX.mappingVerdict,
           Literal(mapping.verdict,                     datatype=XSD.string)))
    g.add((ont, EX.subclassCoverage,
           Literal(round(mapping.subclass_coverage, 4), datatype=XSD.float)))

    # Diagnosis parent class
    g.add((EX.Diagnosis, RDF.type,     OWL.Class))
    g.add((EX.Diagnosis, RDFS.label,   Literal("Diagnosis")))
    g.add((EX.Diagnosis, RDFS.comment,
           Literal("Parent class for all disease subclass diagnoses.")))

    # Annotation properties
    for prop_name, comment, dtype in [
        ("sampleCount",             "number of matching records",              XSD.integer),
        ("isCovered",               "true if >= 1 record matches",             XSD.boolean),
        ("isStatisticallyAdequate", "true if sample count >= min required",    XSD.boolean),
        ("minRequiredSamples",      "minimum samples required for adequacy",   XSD.integer),
        ("coverageVerdict",         "adequate / present_insufficient / absent",XSD.string),
        ("subclassPriority",        "clinical priority: critical/high/medium", XSD.string),
        ("clinicalNote",            "clinical description of this subclass",   XSD.string),
        ("guideline",               "source clinical guideline",               XSD.string),
    ]:
        p = EX[prop_name]
        g.add((p, RDF.type,     OWL.AnnotationProperty))
        g.add((p, RDFS.label,   Literal(prop_name)))
        g.add((p, RDFS.comment, Literal(comment)))
        g.add((p, RDFS.range,   dtype))

    # Build result lookup
    result_map = {r.subclass_id: r for r in mapping.subclass_results}

    for sc in ontology.subclasses:
        sc_uri = EX[sc.id]
        r      = result_map.get(sc.id)

        # Determine verdict
        if r is None or not r.covered:
            verdict = "absent"
        elif not r.adequate:
            verdict = "present_insufficient"
        else:
            verdict = "adequate"

        # OWL class declaration
        g.add((sc_uri, RDF.type,        OWL.Class))
        g.add((sc_uri, RDFS.subClassOf, EX.Diagnosis))
        g.add((sc_uri, RDFS.label,      Literal(sc.label)))
        g.add((sc_uri, RDFS.comment,    Literal(sc.clinical_note)))
        g.add((sc_uri, EX.subclassPriority, Literal(sc.priority)))
        g.add((sc_uri, EX.guideline,        Literal(sc.guideline)))
        g.add((sc_uri, EX.minRequiredSamples,
               Literal(sc.min_required_samples, datatype=XSD.integer)))
        g.add((sc_uri, EX.coverageVerdict, Literal(verdict)))

        if r is not None:
            g.add((sc_uri, EX.sampleCount,
                   Literal(r.sample_count, datatype=XSD.integer)))
            g.add((sc_uri, EX.isCovered,
                   Literal(r.covered,      datatype=XSD.boolean)))
            g.add((sc_uri, EX.isStatisticallyAdequate,
                   Literal(r.adequate,     datatype=XSD.boolean)))
            g.add((sc_uri, EX.clinicalNote, Literal(r.clinical_note)))

        # Named individual — one per subclass
        ind = EX[f"Diagnosis_{sc.id}"]
        g.add((ind, RDF.type,  sc_uri))
        g.add((ind, RDF.type,  OWL.NamedIndividual))
        g.add((ind, RDFS.label,
               Literal(f"Diagnosis: {sc.short_label}")))
        g.add((ind, EX.coverageVerdict,   Literal(verdict)))
        g.add((ind, EX.subclassPriority,  Literal(sc.priority)))
        if r is not None:
            g.add((ind, EX.sampleCount,
                   Literal(r.sample_count, datatype=XSD.integer)))
            g.add((ind, EX.isCovered,
                   Literal(r.covered,      datatype=XSD.boolean)))
            g.add((ind, EX.isStatisticallyAdequate,
                   Literal(r.adequate,     datatype=XSD.boolean)))
            g.add((ind, EX.minRequiredSamples,
                   Literal(sc.min_required_samples, datatype=XSD.integer)))

    return g


# ═══════════════════════════════════════════════════════════════════════════════
#  Stage 2b — Combinations OWL  (NEW)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_combinations(
    ontology:  DomainOntology,
    mapping:   OntologyMappingReport,
    n_records: int,
) -> Graph:
    """
    SemanticCoverage class hierarchy.
    One OWL class per semantic progression stage, each subClassOf SemanticCoverage.
    Pipeline results and up to 20 missing combinations attached as annotations.
    """
    g  = Graph()
    EX = Namespace(EX_URI)
    g.bind("ex",   EX)
    g.bind("owl",  OWL)
    g.bind("rdfs", RDFS)
    g.bind("xsd",  XSD)

    # Ontology header
    ont = EX["HeartDiseaseOntology_Combinations"]
    g.add((ont, RDF.type,     OWL.Ontology))
    g.add((ont, RDFS.label,
           Literal(f"{ontology.full_name} — Stage 2b: Combinations")))
    g.add((ont, RDFS.comment, Literal(
        "Semantic combination coverage layer. Contains the SemanticCoverage "
        "class hierarchy with one OWL class per semantic progression stage, "
        "annotated with expected/observed combination counts, raw coverage "
        "score, theoretical maximum, and missing combination tuples."
    )))
    g.add((ont, EX.datasetRecords,
           Literal(n_records, datatype=XSD.integer)))
    g.add((ont, EX.semanticCoverage,
           Literal(round(mapping.semantic_coverage, 4), datatype=XSD.float)))

    # SemanticCoverage parent class
    g.add((EX.SemanticCoverage, RDF.type,     OWL.Class))
    g.add((EX.SemanticCoverage, RDFS.label,   Literal("SemanticCoverage")))
    g.add((EX.SemanticCoverage, RDFS.comment,
           Literal("Parent class for all semantic combination coverage stages.")))

    # Annotation properties
    for prop_name, comment, dtype in [
        ("expectedCombinations", "total ontology-defined combinations at this stage", XSD.integer),
        ("observedCombinations", "combinations actually present in the dataset",      XSD.integer),
        ("rawCoverageScore",     "observed / expected as a percentage",               XSD.float),
        ("theoreticalMaxScore",  "min(n,expected)/expected * 100",                   XSD.float),
        ("stageAttributes",      "comma-separated attribute list at this stage",      XSD.string),
        ("stageNumber",          "stage index 1 through 6",                           XSD.integer),
        ("missingCombination",   "a combination tuple absent from the dataset",       XSD.string),
    ]:
        p = EX[prop_name]
        g.add((p, RDF.type,     OWL.AnnotationProperty))
        g.add((p, RDFS.label,   Literal(prop_name)))
        g.add((p, RDFS.comment, Literal(comment)))
        g.add((p, RDFS.range,   dtype))

    # One class per semantic stage
    for stage in mapping.semantic_stages:
        safe_attrs = re.sub(r"[^a-zA-Z0-9_]", "_",
                            "_".join(stage.attributes))
        stage_cls  = EX[f"Stage{stage.stage}_{safe_attrs}"]

        expected  = stage.expected_combinations
        observed  = stage.observed_combinations
        raw_score = round(stage.score * 100, 2)

        # Theoretical maximum
        if hasattr(stage, "theoretical_max"):
            theo_max = round(stage.theoretical_max * 100, 2)
        else:
            theo_max = round(
                min(n_records, expected) / expected * 100
                if expected else 0.0, 2
            )

        attrs_str = ", ".join(stage.attributes)

        # Class declaration
        g.add((stage_cls, RDF.type,         OWL.Class))
        g.add((stage_cls, RDFS.subClassOf,  EX.SemanticCoverage))
        g.add((stage_cls, RDFS.label,
               Literal(f"Stage {stage.stage}: {stage.description}")))
        g.add((stage_cls, EX.stageNumber,
               Literal(stage.stage,   datatype=XSD.integer)))
        g.add((stage_cls, EX.stageAttributes,
               Literal(attrs_str)))
        g.add((stage_cls, EX.expectedCombinations,
               Literal(expected,      datatype=XSD.integer)))
        g.add((stage_cls, EX.observedCombinations,
               Literal(observed,      datatype=XSD.integer)))
        g.add((stage_cls, EX.rawCoverageScore,
               Literal(raw_score,     datatype=XSD.float)))
        g.add((stage_cls, EX.theoreticalMaxScore,
               Literal(theo_max,      datatype=XSD.float)))

        # Missing combinations — up to 20
        if hasattr(stage, "sample_missing_combos"):
            for combo in stage.sample_missing_combos[:20]:
                g.add((stage_cls, EX.missingCombination,
                       Literal(str(combo))))

        # Named individual per stage
        ind = EX[f"CoverageStage_{stage.stage}"]
        g.add((ind, RDF.type,  stage_cls))
        g.add((ind, RDF.type,  OWL.NamedIndividual))
        g.add((ind, RDFS.label,
               Literal(f"Coverage Stage {stage.stage}: {attrs_str}")))
        g.add((ind, EX.rawCoverageScore,
               Literal(raw_score, datatype=XSD.float)))
        g.add((ind, EX.theoreticalMaxScore,
               Literal(theo_max,  datatype=XSD.float)))
        g.add((ind, EX.expectedCombinations,
               Literal(expected,  datatype=XSD.integer)))
        g.add((ind, EX.observedCombinations,
               Literal(observed,  datatype=XSD.integer)))

    return g


# ═══════════════════════════════════════════════════════════════════════════════
#  Public export entry-point  (updated signature — adds n_records)
# ═══════════════════════════════════════════════════════════════════════════════

def export_owl(
    ontology:   DomainOntology,
    col_cov:    ColumnCoverageReport,
    mapping:    OntologyMappingReport,
    output_dir: str = "outputs",
    n_records:  int = 0,
) -> str:
    """
    Build and serialise all three OWL files.

    Returns the path of the diagnoses file (used by validate_owl for R4).
    n_records is passed to the combinations builder for theoretical-max
    calculation. If 0, it is derived from mapping.n_records if available.
    """
    os.makedirs(output_dir, exist_ok=True)
    domain_name = getattr(ontology, "domain", "ontology")

    # Resolve n_records
    if n_records == 0:
        n_records = getattr(mapping, "n_records", 0)

    # Stage 1 — Schema
    g1      = _build_stage1(ontology)
    path_s1 = os.path.join(output_dir,
                           f"{domain_name}_ontology_stage1_schema.owl")
    g1.serialize(destination=path_s1, format="xml")
    print(f"  Stage 1 OWL saved → {path_s1}  ({len(g1):,} triples)")

    # Stage 2a — Diagnoses
    g2a      = _build_diagnoses(ontology, mapping)
    path_s2a = os.path.join(output_dir,
                            f"{domain_name}_ontology_diagnoses.owl")
    g2a.serialize(destination=path_s2a, format="xml")
    print(f"  Stage 2a OWL saved → {path_s2a}  ({len(g2a):,} triples)")

    # Stage 2b — Combinations
    g2b      = _build_combinations(ontology, mapping, n_records)
    path_s2b = os.path.join(output_dir,
                            f"{domain_name}_ontology_combinations.owl")
    g2b.serialize(destination=path_s2b, format="xml")
    print(f"  Stage 2b OWL saved → {path_s2b}  ({len(g2b):,} triples)")

    # Return diagnoses path — validate_owl runs R4 against it
    return path_s2a


# ═══════════════════════════════════════════════════════════════════════════════
#  SPARQL validation  (unchanged from original — runs on diagnoses file for R4)
# ═══════════════════════════════════════════════════════════════════════════════

def validate_owl(
    owl_path: str,
    df:       pd.DataFrame,
    ontology: DomainOntology,
) -> OWLValidationReport:
    """
    Load the diagnoses OWL, run four SPARQL rule families.

    R1: Required columns must exist in the dataset.
    R2: Continuous values must fall within OWL-defined bins.
    R3: Categorical values must match OWL-defined codes.
    R4: Subclass sample counts must meet minRequiredSamples.

    R1/R2/R3 query the schema embedded in the diagnoses file
    (attribute classes with datasetColumn / isRequired / rangeMin etc.
    are not in the diagnoses file — they live in stage1_schema.owl).
    We therefore reload the schema file for R1-R3 and the diagnoses
    file for R4.
    """
    # Derive schema path from diagnoses path
    schema_path = owl_path.replace("_ontology_diagnoses.owl",
                                   "_ontology_stage1_schema.owl")

    g_schema = Graph()
    if os.path.exists(schema_path):
        g_schema.parse(schema_path, format="xml")

    g_diag = Graph()
    g_diag.parse(owl_path, format="xml")

    violations:   List[OWLViolation] = []
    passed_rules: List[str]          = []

    # ── R1: Required columns must exist ──────────────────────────────────────
    r1_rows = _sparql_rows(g_schema, """
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
        label, col = r[0], r[1]
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
    r2_rows = _sparql_rows(g_schema, """
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
        if col not in df.columns:
            violations.append(OWLViolation(
                rule="R2 — Continuous range conformance",
                severity="critical",
                column=col,
                description=f"Column '{col}' expected for range validation but is absent.",
                affected_rows=0,
            ))
            continue
        numeric = pd.to_numeric(df[col], errors="coerce").dropna()
        if numeric.empty:
            violations.append(OWLViolation(
                rule="R2 — Continuous range conformance",
                severity="warning",
                column=col,
                description=(
                    f"Column '{col}' exists but has no numeric values — "
                    f"range conformance cannot be verified."
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
                examples=[str(x) for x in
                          sorted(out_of_range.unique().tolist())[:5]],
            ))
        else:
            passed_rules.append(
                f"R2 ✓  '{col}' — all {len(numeric)} values within OWL-defined bins"
            )

    # ── R3: Categorical values match OWL-defined codes ────────────────────────
    r3_rows = _sparql_rows(g_schema, """
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
                description=f"Column '{col}' expected for categorical validation but is absent.",
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
                    f"Column '{col}' exists but has no categorical values — "
                    f"code conformance cannot be verified."
                ),
                affected_rows=0,
            ))
            continue
        obs_norm   = {_norm_code(str(v)) for v in numeric}
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
    r4_rows = _sparql_rows(g_diag, """
        PREFIX ex:   <http://example.org/heartdisease#>
        PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?label ?count ?required WHERE {
            ?sc rdfs:subClassOf       ex:Diagnosis ;
                rdfs:label             ?label ;
                ex:sampleCount         ?count ;
                ex:minRequiredSamples  ?required .
            FILTER (xsd:integer(?count) < xsd:integer(?required))
        }
    """)
    for r in r4_rows:
        label, count, required = r[0], int(r[1]), int(r[2])
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
        total_triples=len(g_diag),
        violations=violations,
        passed_rules=passed_rules,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Terminal report  (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

def print_owl_validation(report: OWLValidationReport) -> None:
    W = "═" * 72
    print(f"\n{W}")
    print("  STEP 6 — OWL VALIDATION  (SPARQL queries against OWL files)")
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
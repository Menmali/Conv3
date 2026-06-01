"""
main.py — OntoCov  |  5-step pipeline

Just run:   python main.py
It will ask you everything it needs interactively.

Or pass arguments directly:
    python main.py --dataset data/my_file.csv --ontology ontologies/cardiovascular.json
"""

import argparse
import os
import sys
import glob

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from framework.owl_exporter         import export_owl
from framework.ontology_loader      import load_ontology
from framework.structural_validator import validate_structure
from framework.column_coverage      import compute_column_coverage
from framework.ontology_mapper      import map_to_ontology
from framework.report_generator     import (
    print_validation, print_column_coverage,
    print_ontology_mapping, generate_charts,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Interactive prompts
# ═══════════════════════════════════════════════════════════════════════════════

def pick_ontology() -> str:
    """Let the user pick an ontology from the ontologies/ folder."""
    folder  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ontologies")
    options = sorted(glob.glob(os.path.join(folder, "*.json")))

    if not options:
        print("\n  ERROR: No ontology files found in ontologies/")
        print("  Please add a .json ontology file there first.")
        sys.exit(1)

    print("\n  ┌─────────────────────────────────────────────────────┐")
    print("  │  Which disease ontology do you want to use?         │")
    print("  └─────────────────────────────────────────────────────┘")
    for i, path in enumerate(options, 1):
        name = os.path.basename(path)
        print(f"    [{i}]  {name}")

    while True:
        choice = input("\n  Enter number: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        print("  Please enter a valid number.")


def pick_dataset() -> str:
    """Let the user enter the path to their CSV dataset."""
    print("\n  ┌─────────────────────────────────────────────────────┐")
    print("  │  Which dataset do you want to test?                 │")
    print("  └─────────────────────────────────────────────────────┘")

    # Show CSV files already in the data/ folder as suggestions
    data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    existing    = sorted(glob.glob(os.path.join(data_folder, "*.csv")))

    if existing:
        print("\n  CSV files found in data/ folder:")
        for i, path in enumerate(existing, 1):
            print(f"    [{i}]  {os.path.basename(path)}")
        print(f"\n  You can type a number to pick one of the above,")
        print(f"  or type the full path to any CSV file on your computer.")
        print(f"  (Press Enter with no input to use built-in demo data)")
    else:
        print("\n  No CSV files found in data/ folder yet.")
        print("  Type the full path to your CSV file, e.g.:  C:\\Users\\Ali\\heart.csv")
        print("  (Press Enter with no input to use built-in demo data)")

    while True:
        raw = input("\n  Your choice: ").strip()

        # Empty → use demo
        if raw == "":
            return None # type: ignore

        # Number → pick from list
        if raw.isdigit() and existing and 1 <= int(raw) <= len(existing):
            return existing[int(raw) - 1]

        # Path → check it exists
        if os.path.exists(raw):
            return raw

        # Try relative to data/ folder
        candidate = os.path.join(data_folder, raw)
        if os.path.exists(candidate):
            return candidate

        print(f"\n  File not found: '{raw}'")
        print("  Please check the path and try again, or press Enter for demo data.")


def confirm_output_folder() -> str:
    """Ask where to save charts."""
    default = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    print(f"\n  Charts will be saved to:  {default}")
    raw = input("  Press Enter to confirm, or type a different folder: ").strip()
    return raw if raw else default


def ask_charts() -> bool:
    raw = input("\n  Generate charts? [Y/n] (press Enter for Yes): ").strip().lower()
    return raw != "n"


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _banner():
    print("\n" + "═" * 72)
    print("  OntoCov — Ontology-Based Dataset Coverage Assessment")
    print("═" * 72)


def _step(n: int, title: str):
    print(f"\n  ── Step {n}: {title}")


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Parse optional CLI args (for power users) ─────────────────────────────
    parser = argparse.ArgumentParser(prog="ontocov", add_help=True)
    parser.add_argument("--dataset",   default=None)
    parser.add_argument("--ontology",  default=None)
    parser.add_argument("--output",    default=None)
    parser.add_argument("--no-charts", action="store_true")
    args = parser.parse_args()

    _banner()

    # ── If no CLI args, ask interactively ─────────────────────────────────────
    ontology_path = args.ontology  if args.ontology  else pick_ontology()
    dataset_path  = args.dataset   if args.dataset   else pick_dataset()
    output_dir    = args.output    if args.output     else confirm_output_folder()
    want_charts   = False if args.no_charts else ask_charts()

    # ── Step 1: Ontology ──────────────────────────────────────────────────────
    _step(1, "Loading disease ontology")
    ontology = load_ontology(ontology_path)
    print(f"  Domain     : {ontology.full_name}")
    print(f"  Attributes : {len(ontology.attributes)}"
          f"  |  Sub-classes : {len(ontology.subclasses)}"
          f"  |  Semantic stages : {len(ontology.semantic_progression)}")
    for sc in ontology.subclasses:
        print(f"    • {sc.label}  [{sc.priority}]")

    # ── Step 2: Dataset ───────────────────────────────────────────────────────
    _step(2, "Loading dataset")
    if dataset_path:
        df = normalise_columns(pd.read_csv(dataset_path))
        dataset_name = os.path.basename(dataset_path)
    else:
        print("  Using built-in demo data (UCI Heart Disease schema)…")
        print("  ⚠️  Note: demo data intentionally excludes HighRiskCAD and")
        print("       SevereIschaemia patients to demonstrate gap detection.")
        print("       Uncovered subclasses in the report below are expected.")
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
        from generate_demo import generate
        df = generate()
        dataset_name = "heart_demo.csv [synthetic demo]"

    print(f"  File    : {dataset_name}")
    print(f"  Records : {len(df):,}   |   Columns : {len(df.columns)}")
    print(f"  Columns : {list(df.columns)}")

    # ── Step 3 ────────────────────────────────────────────────────────────────
    _step(3, "Structural validation")
    validation = validate_structure(df, ontology, dataset_name=dataset_name)
    print_validation(validation)

    # ── Step 4 ────────────────────────────────────────────────────────────────
    _step(4, "Column-wise data coverage")
    col_cov = compute_column_coverage(df, ontology)
    print_column_coverage(col_cov)

    # ── Step 5 ────────────────────────────────────────────────────────────────
    _step(5, "Ontology mapping")
    mapping = map_to_ontology(df, ontology, col_cov=col_cov)   # ← add col_cov=col_cov
    print_ontology_mapping(mapping)

    # ── Step 6 ────────────────────────────────────────────────────────────────
    _step(6, "Exporting OWL ontology + validation")
    from framework.owl_exporter import export_owl, validate_owl, print_owl_validation

    # Generate the OWL file first
    owl_path = export_owl(ontology, col_cov, mapping, output_dir=output_dir)
    print(f"  OWL file saved → {owl_path}")

    # Then validate it
    owl_report = validate_owl(owl_path, df, ontology)
    print_owl_validation(owl_report)
    # ── Charts ────────────────────────────────────────────────────────────────
    if want_charts:
        print(f"\n  Saving charts → {output_dir}/")
        generate_charts(validation, col_cov, mapping, output_dir=output_dir)
    print("  Done.\n")
    input("  Press Enter to close...") 

if __name__ == "__main__":
    main()

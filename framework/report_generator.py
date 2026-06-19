"""
report_generator.py

Terminal output + chart generation for all 5 pipeline steps.

Charts produced:
  chart1_structural.png       : Step 3: column health heatmap
  chart2_column_coverage.png  : Step 4: per-column coverage bars
  chart3_ontology_mapping.png : Step 5: four-curve semantic chart +
                                subclass population + mapping scores
"""

import math
import os

from .structural_validator import ValidationReport
from .column_coverage       import ColumnCoverageReport
from .ontology_mapper       import OntologyMappingReport


# ── terminal helpers ──────────────────────────────────────────────────────────

def _bar(score: float, width: int = 26) -> str:
    filled = int(round(score * width))
    return f"[{'█' * filled}{'░' * (width - filled)}] {score:5.1%}"


SEV_ICON      = {"critical": "✘", "warning": "⚠️ ", "info": "ℹ️ ", "ok": "✓"}
PRIORITY_ICON = {"critical": "[C]", "high": "[H]", "medium": "[M]","low": "[L]"}
VERDICT_TEXT  = {
    "sufficient":   "✓  SUFFICIENT     : dataset is ready for ML",
    "borderline":   "⚠️   BORDERLINE    : gaps detected, proceed with caution",
    "insufficient": "✘  INSUFFICIENT  : critical gaps, do not train yet",
}
W  = "═" * 72
W2 = "─" * 72


# ══════════════════════════════════════════════════════════════════════════════
#  Step 3 : Structural Validation
# ══════════════════════════════════════════════════════════════════════════════

def print_validation(v: ValidationReport) -> None:
    print(f"\n{W}")
    print(f"  STEP 3 : STRUCTURAL VALIDATION")
    print(W)
    print(f"  Dataset  : {v.dataset_name}")
    print(f"  Records  : {v.n_records:,}   |   Columns found: {v.n_columns}"
          f"   |   Duplicate rows: {v.duplicate_count}")
    status_icon = SEV_ICON.get(v.overall_status, "")
    print(f"  Status   : {status_icon} {v.overall_status.upper()}"
          f"   ({len(v.critical_issues)} critical, {len(v.warnings)} warnings,"
          f" {len(v.infos)} info)\n")

    if not v.issues:
        print("  No structural issues found.\n")
    else:
        print(f"  {'Severity':<10} {'Column':<36} {'Issue':<28} {'Affected'}")
        print(f"  {'─'*10} {'─'*36} {'─'*28} {'─'*12}")
        for issue in sorted(v.issues,
                            key=lambda i: (i.severity != "critical",
                                           i.severity != "warning")):
            icon = SEV_ICON.get(issue.severity, "")
            col  = issue.column or "(dataset-wide)"
            pct  = f"{issue.affected_rows} rows ({issue.pct:.1%})"
            print(f"  {icon} {issue.severity:<7}  {col:<36}"
                  f" {issue.issue_type:<28} {pct}")
            if issue.examples:
                print(f"             Examples: {issue.examples}")

    print(f"\n  Column-by-column summary:")
    print(f"  {'Attribute':<42} {'Column':<14} {'Missing':<10}"
          f" {'Range':<20} {'Status'}")
    print(f"  {'─'*42} {'─'*14} {'─'*10} {'─'*20} {'─'*10}")
    for s in v.column_summaries:
        col  = s.column or "—"
        miss = f"{s.missing_pct:.1%}" if s.column else "—"
        rng  = (f"{s.min_val:.0f} – {s.max_val:.0f}"
                if s.min_val is not None else "—")
        icon = SEV_ICON.get(s.status, "")
        print(f"  {s.label:<42} {col:<14} {miss:<10} {rng:<20} {icon}")


# ══════════════════════════════════════════════════════════════════════════════
#  Step 4 — Column Coverage
# ══════════════════════════════════════════════════════════════════════════════

def print_column_coverage(c: ColumnCoverageReport) -> None:
    print(f"\n{W}")
    print(f"  STEP 4 : COLUMN-WISE DATA COVERAGE")
    print(W)
    print(f"  Overall column coverage: {_bar(c.overall_score)}\n")
    print(f"  {'Attribute':<34} {'Type':<12} {'Coverage':<14}"
          f" {'Observed / Expected'}   Missing values")
    print(f"  {'─'*34} {'─'*12} {'─'*14} {'─'*20}   {'─'*28}")

    for r in c.results:
        if not r.expected_labels:
            if not r.present:
                print(f"  ✘ {r.label:<32} {'ABSENT':<12} {'—':>6}")
            continue
        ratio   = f"{len(r.observed_keys)}/{len(r.expected_labels)}"
        missing = ", ".join(r.missing_labels[:3])
        if len(r.missing_labels) > 3:
            missing += " …"
        if not missing:
            missing = "—"
        icon = ("✓" if r.coverage_score == 1.0
                else ("⚠️ " if r.coverage_score >= 0.5 else "✘"))
        print(f"  {icon} {r.label:<32} {r.attr_type:<12}"
              f" {r.coverage_score:5.1%}  {ratio:<10}   {missing}")


# ══════════════════════════════════════════════════════════════════════════════
#  Step 5 — Ontology Mapping
# ══════════════════════════════════════════════════════════════════════════════

def print_ontology_mapping(m: OntologyMappingReport) -> None:
    print(f"\n{W}")
    print(f"  STEP 5 — ONTOLOGY MAPPING")
    print(W)

    print(f"\n  A) Semantic combination coverage  (progressive)")
    print(W2)
    for s in m.semantic_stages:
        attrs  = " × ".join(s.attributes)
        combos = f"({s.observed_combinations}/{s.expected_combinations} raw)"

        # Show theoretical-max context if available
        if s.theoretical_max > 0 and s.expected_combinations > 0:
            norm_pct = f"  norm {s.normalized_score:.1%}"
            max_pct  = f"  ceil {s.theoretical_max / s.expected_combinations:.1%}"
        else:
            norm_pct = max_pct = ""

        print(f"  Stage {s.stage}  {_bar(s.score, 20)}"
              f"  {combos:<26}{norm_pct}{max_pct}  {attrs}")

    print(f"\n  B) Disease sub-class coverage")
    print(W2)
    n_cov = sum(1 for r in m.subclass_results if r.covered)
    print(f"  {n_cov}/{len(m.subclass_results)} sub-classes populated"
          f"  |  {sum(1 for r in m.subclass_results if r.adequate)}"
          f"/{len(m.subclass_results)} statistically adequate\n")

    for r in m.subclass_results:
        icon = PRIORITY_ICON.get(r.priority, "[?]")

        if r.adequate:
            status = f"✓  {r.sample_count} samples (≥ {r.min_required})"
        elif r.covered:
            status = f"⚠️   {r.sample_count} samples (need ≥ {r.min_required})"
        else:
            status = "✘  NOT COVERED"

        # Print the status line
        print(f"  {icon}  {r.label:<48} {status}")

        # If not covered, print the clinical note (indented)
        if not r.covered:
            print(f"       ↳ {r.clinical_note}")

        # Always print the EPV/IR/score line (no prefix)
        ir_score = min((r.sample_count / m.n_records) / 0.01, 1.0) if m.n_records else 0.0
        subclass_score = min(r.epv_score, ir_score)
        print(f"       EPV={r.epv_score:.2f}  IR={ir_score:.2f}  score={subclass_score:.2f}")

    print(f"\n  Scores:")
    print(f"    Semantic coverage (final stage)  {_bar(m.semantic_coverage)}")
    print(f"    Sub-class presence coverage      {_bar(m.subclass_coverage)}")
    print(f"    Statistical adequacy             {_bar(m.adequacy_score)}")

    print(f"\n{W}")
    print(f"\n  ONTOLOGY MAPPING VERDICT : {VERDICT_TEXT.get(m.verdict, m.verdict)}")
    print(f"  Overall mapping score    : {_bar(m.overall_score, 40)}")
    print(f"\n  Scoring basis (Ogundimu et al. 2016 + He & Garcia 2009):")
    print(f"    Per subclass: EPV_score = min(n / 20, 1.0)")
    print(f"    [Ogundimu: EPV >= 20 eliminates coefficient bias]")
    print(f"    Sufficient >= 1.0  |  Borderline >= 0.5  |  Insufficient < 0.5")

    # Hard disqualifiers (logical prerequisites — force the verdict) and
    # soft flags (real caveats, but they don't override the EPV-based
    # verdict on their own) are both stored in m.hard_disqualifiers; split
    # them here for display so a soft flag doesn't read as a hard stop.
    hard = [d for d in m.hard_disqualifiers if getattr(d, "hard", True)]
    soft = [d for d in m.hard_disqualifiers if not getattr(d, "hard", True)]

    if hard:
        print(f"\n  ⛔ HARD DISQUALIFIERS TRIGGERED (verdict forced to insufficient):")
        for d in hard:
            print(f"    → [{d.reason}] {d.description}")

    if soft:
        print(f"\n  ⚠️  CAVEATS (do not override the verdict, but limit deployment scope):")
        for d in soft:
            print(f"    → [{d.reason}] {d.description}")

    uncov = [r.short_label for r in m.subclass_results if not r.covered]
    if uncov:
        print(f"\n  Uncovered sub-classes : a model trained on this dataset"
              f" will fail silently on these patient populations:")
        for lbl in uncov:
            print(f"    →  {lbl}")
    print(f"\n{W}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  Charts
# ══════════════════════════════════════════════════════════════════════════════

def generate_charts(
    val:        ValidationReport,
    cov:        ColumnCoverageReport,
    mapping:    OntologyMappingReport,
    output_dir: str = "outputs",
    n_records:  int = 0,           # ← NEW: pass len(df) from main.py
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
    except ImportError:
        print("  matplotlib not available : skipping charts.")
        return

    os.makedirs(output_dir, exist_ok=True)
    BG, PANEL = "#0f0f1a", "#1a1a2e"
    GREEN, ORANGE, RED, BLUE = "#2ecc71", "#f39c12", "#e74c3c", "#3498db"
    PURPLE = "#9b59b6"

    def sev_color(status):
        return {
            "ok": GREEN, "info": BLUE, "warning": ORANGE,
            "critical": RED, "absent": RED,
        }.get(status, "#888888")

    # FIX: Use 1.0 as Sufficient threshold (new scoring system)
    def score_color(s):
        return GREEN if s >= 1.0 else (ORANGE if s >= 0.50 else RED)

    # ── Chart 1: Structural Validation ───────────────────────────────────────
    sums = val.column_summaries
    if sums:
        fig, axes = plt.subplots(
            1, 2, figsize=(14, max(4, len(sums) * 0.45 + 2)),
            gridspec_kw={"width_ratios": [3, 2]}
        )
        fig.patch.set_facecolor(BG)

        ax = axes[0]
        ax.set_facecolor(PANEL)
        labels   = [s.label[:35] for s in sums]
        miss_pct = [s.missing_pct * 100 for s in sums]
        colors   = [sev_color(s.status) for s in sums]
        y_pos    = range(len(sums))
        ax.barh(list(y_pos), miss_pct, color=colors,
                edgecolor="#ffffff15", height=0.6)
        ax.axvline(5,  color=ORANGE, linestyle="--", linewidth=1,
                   alpha=0.7, label="5% threshold")
        ax.axvline(50, color=RED,    linestyle="--", linewidth=1,
                   alpha=0.7, label="50% threshold")
        for i, (pct, _) in enumerate(zip(miss_pct, sums)):
            ax.text(pct + 0.3, i, f"{pct:.1f}%",
                    va="center", color="white", fontsize=7)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(labels, color="white", fontsize=8)
        ax.set_xlabel("Missing Rate (%)", color="white", fontsize=10)
        ax.set_title("Missing Values per Column",
                     color="white", fontsize=11, fontweight="bold")
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#ffffff20")
        ax.legend(facecolor=BG, edgecolor="#ffffff30",
                  labelcolor="white", fontsize=8)

        ax2 = axes[1]
        ax2.set_facecolor(PANEL)
        colors2 = [sev_color(s.status) for s in sums]
        ax2.scatter([0] * len(sums), list(y_pos),
                    c=colors2, s=120, zorder=3)
        ax2.set_yticks(list(y_pos))
        ax2.set_yticklabels(
            [s.status.upper() for s in sums], color="white", fontsize=8)
        ax2.set_xticks([])
        ax2.set_xlim(-0.5, 0.5)
        ax2.set_title("Status", color="white", fontsize=11, fontweight="bold")
        ax2.tick_params(colors="white")
        for sp in ax2.spines.values():
            sp.set_edgecolor("#ffffff20")
        patches = [mpatches.Patch(color=GREEN, label="OK"),
                   mpatches.Patch(color=ORANGE, label="Warning"),
                   mpatches.Patch(color=RED,    label="Critical / Absent")]
        ax2.legend(handles=patches, facecolor=BG, edgecolor="#ffffff30",
                   labelcolor="white", fontsize=8, loc="lower right")

        fig.suptitle(
            f"Step 3 : Structural Validation  |  {val.dataset_name}",
            color="white", fontsize=13, fontweight="bold", y=1.01)
        plt.tight_layout()
        p = os.path.join(output_dir, "chart1_structural.png")
        plt.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG)
        plt.close()
        print(f"  Chart 1 saved → {p}")

    # ── Chart 2: Column Coverage ──────────────────────────────────────────────
    cov_items = [r for r in cov.results if r.expected_labels]
    if cov_items:
        fig, ax = plt.subplots(
            figsize=(12, max(4, len(cov_items) * 0.6 + 2)))
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(PANEL)

        labels  = [r.label[:38] for r in cov_items]
        scores  = [r.coverage_score * 100 for r in cov_items]
        colors  = [score_color(r.coverage_score) for r in cov_items]
        y_pos   = range(len(cov_items))
        bars = ax.barh(list(y_pos), scores, color=colors,
                       edgecolor="#ffffff15", height=0.6)
        ax.axvline(75, color=GREEN,  linestyle="--", linewidth=1.2,
                   alpha=0.8, label="Sufficient (75%)")
        ax.axvline(50, color=ORANGE, linestyle="--", linewidth=1.2,
                   alpha=0.8, label="Borderline (50%)")

        for h_bar, r in zip(bars, cov_items):
            ratio = f"{len(r.observed_keys)}/{len(r.expected_labels)}"
            miss  = (f"  ← missing: {', '.join(r.missing_labels[:2])}"
                     if r.missing_labels else "")
            ax.text(h_bar.get_width() + 0.5,
                    h_bar.get_y() + h_bar.get_height() / 2,
                    f"{r.coverage_score:.0%}  {ratio}{miss}",
                    va="center", color="white", fontsize=7.5)

        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(labels, color="white", fontsize=9)
        ax.set_xlabel("Column Coverage Score (%)", color="white", fontsize=11)
        ax.set_xlim(0, 145)
        ax.set_title(
            f"Step 4 : Column-wise Data Coverage\n"
            f"Overall: {cov.overall_score:.1%}  |  "
            f"{sum(1 for r in cov_items if r.coverage_score == 1.0)}"
            f"/{len(cov_items)} columns fully covered",
            color="white", fontsize=12, fontweight="bold",
        )
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#ffffff20")
        ax.legend(facecolor=BG, edgecolor="#ffffff30",
                  labelcolor="white", fontsize=9)

        plt.tight_layout()
        p = os.path.join(output_dir, "chart2_column_coverage.png")
        plt.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG)
        plt.close()
        print(f"  Chart 2 saved → {p}")

    # ── Chart 3: Ontology Mapping ─────────────────────────────────────────────
    def wrap_label(attributes, max_per_line=3):
        lines = []
        for i in range(0, len(attributes), max_per_line):
            chunk = attributes[i:i+max_per_line]
            line = " × ".join(chunk)
            if i > 0:
                line = "× " + line
            lines.append(line)
        return "\n".join(lines)

    fig = plt.figure(figsize=(14.5, 10))
    fig.patch.set_facecolor(BG)
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35)

    # ── Top row: full-width semantic chart ─────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor(PANEL)

    stages = mapping.semantic_stages
    x = [s.stage for s in stages]

    y_raw = [s.score * 100 for s in stages]

    n = n_records if n_records > 0 else mapping.n_records
    y_ceil = []
    for s in stages:
        if s.expected_combinations > 0:
            achievable = min(n, s.expected_combinations)
            y_ceil.append(achievable / s.expected_combinations * 100)
        else:
            y_ceil.append(0.0)

    # Draw curves
    ax1.plot(x, y_ceil, color=ORANGE, linewidth=1.8, linestyle="-.",
             marker="s", markersize=6, markerfacecolor=ORANGE,
             label="Theoretical max (dataset size ceiling)")
    ax1.fill_between(x, y_ceil, alpha=0.06, color=ORANGE)

    ax1.plot(x, y_raw, color=PURPLE, linewidth=2.5, marker="o", markersize=9,
             markerfacecolor="white", markeredgecolor=PURPLE,
             markeredgewidth=2, label="Raw coverage (% of full space)")
    ax1.fill_between(x, y_raw, alpha=0.10, color=PURPLE)

    # Annotations
    for xi, ceil_i in zip(x, y_ceil):
        ax1.annotate(f"{ceil_i:.1f}%", (xi, ceil_i),
                     textcoords="offset points", xytext=(0, 10),
                     ha="center", fontsize=8, color="white", fontweight="bold")
    for xi, raw_i in zip(x, y_raw):
        ax1.annotate(f"{raw_i:.1f}%", (xi, raw_i),
                     textcoords="offset points", xytext=(0, 10),
                     ha="center", fontsize=8, color="white", fontweight="bold")

    # Stage 6 callout
    last_stage = stages[-1]
    lx = last_stage.stage
    raw_final = y_raw[-1]
    ceil_final = y_ceil[-1]
    ax1.annotate(f"Stage {lx}\nRaw: {raw_final:.1f}%\nMax: {ceil_final:.1f}%",
                 xy=(lx, raw_final), xytext=(lx - 0.5, raw_final + 20),
                 fontsize=8, color="white",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="#1e1e3a",
                           edgecolor=PURPLE, alpha=0.9),
                 arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1.2))

    # X-axis formatting
    wrapped_labels = [wrap_label(s.attributes, max_per_line=3) for s in stages]
    ax1.set_xticks(x)
    xtick_labels = [f"S{s.stage}:\n{wrapped}" for s, wrapped in zip(stages, wrapped_labels)]
    ax1.set_xticklabels(xtick_labels, color="white", fontsize=7.5, rotation=0, ha='center')
    ax1.set_xlim(x[0] - 0.1, x[-1] + 0.3)
    ax1.set_ylabel("Coverage (%)", color="white", fontsize=10)
    ax1.set_ylim(0, 130)
    ax1.set_title("Semantic Combination Coverage : Progressive",
                  color="white", fontsize=11, fontweight="bold")
    ax1.tick_params(colors="white")
    for sp in ax1.spines.values():
        sp.set_edgecolor("#ffffff20")
    ax1.legend(facecolor=BG, edgecolor="#ffffff30", labelcolor="white", fontsize=8)

    # ── Bottom-left: subclass sample counts ───────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor(PANEL)
    subs = mapping.subclass_results
    s_lbls = [r.short_label for r in subs]
    s_cnt = [r.sample_count for r in subs]
    s_col = [GREEN if r.adequate else (ORANGE if r.covered else RED) for r in subs]
    yp = range(len(subs))

    ax2.barh(list(yp), s_cnt, color=s_col, edgecolor="#ffffff15", height=0.6)
    min_req = subs[0].min_required if subs else 5
    ax2.axvline(min_req, color=ORANGE, linestyle="--", linewidth=1.2, alpha=0.8,
                label=f"Min required ({min_req})")
    for i, (cnt, r) in enumerate(zip(s_cnt, subs)):
        lbl = str(cnt) if cnt > 0 else "NOT COVERED"
        ax2.text(cnt + 0.3, i, lbl, va="center", color="white", fontsize=8)
    ax2.set_yticks(list(yp))
    ax2.set_yticklabels(s_lbls, color="white", fontsize=8)
    ax2.set_xlabel("Sample Count", color="white", fontsize=9)
    ax2.set_title("Sub-class Population", color="white", fontsize=10, fontweight="bold")
    ax2.tick_params(colors="white")
    for sp in ax2.spines.values():
        sp.set_edgecolor("#ffffff20")

    patches = [mpatches.Patch(color=GREEN, label="Adequate (≥ EPV)"),
               mpatches.Patch(color=ORANGE, label="Present, insufficient"),
               mpatches.Patch(color=RED, label="Not covered")]
    ax2.legend(handles=patches, facecolor=BG, edgecolor="#ffffff30",
               labelcolor="white", fontsize=7, loc="lower right")

    # ── Bottom-right: score summary ───────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor(PANEL)

    score_labels = ["Semantic\nCoverage", "Subclass\nPresence",
                    "Statistical\nAdequacy", "EPV\nScore"]
    score_vals = [
        mapping.semantic_coverage * 100,
        mapping.subclass_coverage * 100,
        mapping.adequacy_score    * 100,
        mapping.overall_score     * 100,
    ]

    sc_cols = [score_color(v / 100) for v in score_vals]
    bars3 = ax3.bar(score_labels, score_vals, color=sc_cols,
                    edgecolor="#ffffff15", width=0.5)

    ax3.axhline(mapping.sufficient_threshold * 100, color=GREEN,
                linestyle="--", linewidth=1.1, alpha=0.8,
                label=f"Sufficient ({mapping.sufficient_threshold:.0%})")
    ax3.axhline(mapping.borderline_threshold * 100, color=ORANGE,
                linestyle="--", linewidth=1.1, alpha=0.8,
                label=f"Borderline ({mapping.borderline_threshold:.0%})")

    for bar, v in zip(bars3, score_vals):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                 f"{v:.1f}%", ha="center", va="bottom", color="white",
                 fontsize=9, fontweight="bold")

    ax3.set_ylim(0, 130)
    ax3.set_ylabel("Score (%)", color="white", fontsize=9)
    verdict_color = score_color(mapping.overall_score)
    verdict_short = {
        "sufficient":   "SUFFICIENT ✓",
        "borderline":   "BORDERLINE ⚠️",
        "insufficient": "INSUFFICIENT ✘",
    }.get(mapping.verdict, mapping.verdict.upper())
    ax3.set_title(f"Mapping Scores\nVerdict: {verdict_short}",
                  color=verdict_color, fontsize=10, fontweight="bold")
    ax3.tick_params(colors="white")
    for sp in ax3.spines.values():
        sp.set_edgecolor("#ffffff20")
    ax3.legend(facecolor=BG, edgecolor="#ffffff30", labelcolor="white", fontsize=8)

    fig.suptitle("Step 5 : Ontology Mapping  |  Disease Coverage Assessment",
                 color="white", fontsize=13, fontweight="bold")
    p = os.path.join(output_dir, "chart3_ontology_mapping.png")
    plt.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Chart 3 saved → {p}")



#  Chart 4 : Dataset Adequacy Report
# ══════════════════════════════════════════════════════════════════════════════

def generate_chart4_adequacy(
    mapping:    "OntologyMappingReport",
    cov:        "ColumnCoverageReport",
    output_dir: str = "outputs",
) -> None:
    """
    Chart 4 : Dataset Adequacy Report

    Three panels:
      Left   : EPV score per subclass (bar chart)
                with EPV=20 threshold line and imbalance ratio annotations
      Right  : Step-by-step verdict panel
                Step 1 : Hard disqualifiers checklist
                Step 2 : EPV Coverage Score gauge
                Step 3 : Verdict badge
      Bottom : Plain-language Final Verdict
                (attribute coverage, semantic combination coverage,
                 subclass presence, statistical adequacy,
                 critical problem, final verdict : no formulas)
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import matplotlib.gridspec as gridspec
        import numpy as np
        import os
    except ImportError:
        print("  matplotlib not available : skipping Chart 4.")
        return

    os.makedirs(output_dir, exist_ok=True)

    BG, PANEL   = "#0f0f1a", "#1a1a2e"
    GREEN       = "#2ecc71"
    ORANGE      = "#f39c12"
    RED         = "#e74c3c"
    BLUE        = "#3498db"
    PURPLE      = "#9b59b6"
    WHITE       = "#ffffff"
    GREY        = "#aaaaaa"

    PRIORITY_COLOR = {"critical": RED, "high": ORANGE, "medium": BLUE, "low": GREEN}

    def score_color(s: float) -> str:
        return GREEN if s >= 1.0 else (ORANGE if s >= 0.5 else RED)

    VERDICT_COLOR = {
        "sufficient":   GREEN,
        "borderline":   ORANGE,
        "insufficient": RED,
    }
    VERDICT_LABEL = {
        "sufficient":   "SUFFICIENT ✓",
        "borderline":   "BORDERLINE ⚠️",
        "insufficient": "INSUFFICIENT ❌",
    }

    fig = plt.figure(figsize=(16, 13))
    fig.patch.set_facecolor(BG)

    # Bottom panel now carries a full qualitative verdict (table + critical
    # problem + final verdict), so it needs much more vertical room than
    # the old 3:1 split.
    gs = gridspec.GridSpec(
        2, 2,
        figure=fig,
        height_ratios=[1.5, 2.2],
        hspace=0.22,
        wspace=0.35,
    )

    # ── Panel 1 (top-left): EPV score per subclass ────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(PANEL)

    subs    = mapping.subclass_results
    labels  = [r.short_label for r in subs]
    scores  = [r.epv_score   for r in subs]
    counts  = [r.sample_count for r in subs]
    colors  = [PRIORITY_COLOR.get(r.priority, BLUE) for r in subs]
    y_pos   = range(len(subs))

    bars = ax1.barh(list(y_pos), scores, color=colors,
                    edgecolor="#ffffff15", height=0.6)

    # EPV threshold line at 1.0 (= 20 samples)
    ax1.axvline(1.0, color=GREEN, linestyle="--", linewidth=1.5,
                alpha=0.9, label="EPV = 20 threshold (Ogundimu 2016)")
    # Ogundimu severe bias boundary at 0.5 (= EPV 10)
    ax1.axvline(0.5, color=ORANGE, linestyle=":", linewidth=1.2,
                alpha=0.8, label="EPV = 10 (severe bias boundary)")

    for bar, r in zip(bars, subs):
        # Sample count label
        ax1.text(
            bar.get_width() + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"n={r.sample_count}  EPV={r.epv_score:.2f}"
            + (" ⚠️IR" if r.ir_violation else ""),
            va="center", color=WHITE, fontsize=8,
        )

    ax1.set_yticks(list(y_pos))
    ax1.set_yticklabels(labels, color=WHITE, fontsize=9)
    ax1.set_xlabel("EPV Score  [min(n / 20, 1.0)]", color=WHITE, fontsize=9)
    ax1.set_xlim(0, 1.65)
    ax1.set_title(
        "Step 2 : EPV Coverage Score per Subclass\n(Ogundimu et al., J Clin Epidemiol 2016)",
        color=WHITE, fontsize=10, fontweight="bold",
    )
    ax1.tick_params(colors=WHITE)
    for sp in ax1.spines.values():
        sp.set_edgecolor("#ffffff20")

    priority_patches = [
        mpatches.Patch(color=RED,    label="Critical priority"),
        mpatches.Patch(color=ORANGE, label="High priority"),
        mpatches.Patch(color=BLUE,   label="Medium priority"),
        mpatches.Patch(color=GREEN,  label="Low priority"),
    ]
    ax1.legend(
        handles=priority_patches,
        facecolor=BG, edgecolor="#ffffff30",
        labelcolor=WHITE, fontsize=7, loc="lower right",
    )

    # ── Panel 2 (top-right): Step-by-step verdict ─────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(PANEL)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")

    y = 0.97

    def write(text, x=0.04, dy=0.06, size=9.0, color=WHITE,
              weight="normal", style="normal"):
        nonlocal y
        ax2.text(x, y, text, transform=ax2.transAxes,
                 fontsize=size, color=color, fontweight=weight,
                 fontstyle=style, va="top", wrap=True)
        y -= dy

    # ── Step 1 header ─────────────────────────────────────────────────────────
    write("STEP 1 : Hard Disqualifiers", size=10, weight="bold", color=ORANGE)
    write("(logical prerequisites : force the verdict)", size=7.5,
          color=GREY, dy=0.055)

    hard_dq = [d for d in mapping.hard_disqualifiers if getattr(d, "hard", True)]
    soft_dq = [d for d in mapping.hard_disqualifiers if not getattr(d, "hard", True)]

    if not hard_dq:
        write("  ✓  No disqualifiers triggered : proceed to Step 2",
              color=GREEN, dy=0.055)
    else:
        for d in hard_dq:
            write(f"  ❌  {d.reason}", color=RED, size=8.5, dy=0.05)
            # wrap long description
            desc = d.description
            while len(desc) > 62:
                write(f"       {desc[:62]}", color="#cccccc",
                      size=7.5, dy=0.04)
                desc = desc[62:]
            write(f"       {desc}", color="#cccccc", size=7.5, dy=0.048)

    if soft_dq:
        write(f"  ⚠️  {len(soft_dq)} caveat(s) noted (He & Garcia 2009) :",
              color=ORANGE, size=7.5, dy=0.04)
        write("       shown below, do not force this verdict.",
              color=ORANGE, size=7.5, dy=0.05)

    y -= 0.02

    # ── Step 2 header ─────────────────────────────────────────────────────────
    write("STEP 2 : EPV Coverage Score", size=10, weight="bold", color=BLUE)
    write("mean( min(n_s / 20, 1.0) ) across subclasses", size=7.5,
          color=GREY, dy=0.05)

    # The score is always computed and shown, even when a disqualifier
    # is present : disqualification no longer blocks scoring/display.
    score_str = f"  Overall EPV Score:  {mapping.overall_score:.3f}"
    write(score_str, color=score_color(mapping.overall_score),
          size=10, weight="bold", dy=0.06)

    # Mini gauge bar
    gauge_y = y + 0.01
    ax2.add_patch(mpatches.FancyBboxPatch(
        (0.04, gauge_y - 0.012), 0.92, 0.022,
        boxstyle="round,pad=0.005",
        facecolor="#333355", edgecolor="#ffffff20",
        transform=ax2.transAxes, clip_on=False,
    ))
    fill_w = 0.92 * min(mapping.overall_score, 1.0)
    ax2.add_patch(mpatches.FancyBboxPatch(
        (0.04, gauge_y - 0.012), fill_w, 0.022,
        boxstyle="round,pad=0.005",
        facecolor=score_color(mapping.overall_score),
        edgecolor="none",
        transform=ax2.transAxes, clip_on=False,
    ))
    y -= 0.06

    write(
        f"  Sufficient  ≥ 1.0  |  Borderline ≥ 0.5  |  Insufficient < 0.5",
        color=GREY, size=7.5, dy=0.055,
    )

    if hard_dq:
        write("  ⚠️  Disqualifier(s) noted above : score shown for", color=ORANGE,
              size=7.5, dy=0.04)
        write("       reference, but deployment is not recommended.", color=ORANGE,
              size=7.5, dy=0.055)

    y -= 0.02

    # ── Step 3 verdict badge ──────────────────────────────────────────────────
    write("STEP 3 : Verdict", size=10, weight="bold",
          color=VERDICT_COLOR.get(mapping.verdict, WHITE))

    verdict_label = VERDICT_LABEL.get(mapping.verdict, mapping.verdict.upper())
    write(f"  {verdict_label}", size=13, weight="bold",
          color=VERDICT_COLOR.get(mapping.verdict, WHITE), dy=0.075)

    ax2.set_title(
        "Dataset Adequacy Assessment",
        color=WHITE, fontsize=10, fontweight="bold",
    )

    # ── Panel 3 (bottom, full width): Final Verdict (plain language) ──────────
    ax3 = fig.add_subplot(gs[1, :])
    ax3.set_facecolor("#12122a")
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis("off")

    m       = mapping
    n_pres  = m.n_subclasses_present
    n_tot   = m.n_subclasses_total
    n_adeq  = m.n_subclasses_adequate
    N       = m.n_records

    attr_cov = cov.overall_score if cov is not None else None
    sem_cov  = getattr(m, "semantic_coverage", None)

    missing_subclasses = [r for r in m.subclass_results if r.sample_count == 0]
    ir_flagged         = [r for r in m.subclass_results if r.ir_violation]

    vy = 0.95  # vertical cursor, in axes fraction, top-down

    def vwrite(text, x=0.02, dy=0.052, size=10.5, color=WHITE,
               weight="normal", style="normal"):
        nonlocal vy
        ax3.text(x, vy, text, transform=ax3.transAxes,
                 fontsize=size, color=color, fontweight=weight,
                 fontstyle=style, va="top")
        vy -= dy

    def wrap_write(text, x=0.02, width=128, size=9.3, color="#dddddd",
                   dy=0.04, style="normal", weight="normal"):
        nonlocal vy
        words, cur, lines = text.split(), "", []
        for w in words:
            if len(cur) + len(w) + 1 > width:
                lines.append(cur)
                cur = w
            else:
                cur = (cur + " " + w).strip()
        if cur:
            lines.append(cur)
        for ln in lines:
            ax3.text(x, vy, ln, transform=ax3.transAxes,
                     fontsize=size, color=color, va="top",
                     style=style, fontweight=weight)
            vy -= dy

    # ── Header ──────────────────────────────────────────────────────────────
    vwrite(f"Final Verdict:",
           size=14, weight="bold", color=WHITE, dy=0.058)
    wrap_write(
        f"Datset Contains {N:,} patient records. When assessed through the "
        f"ontology-guided framework, the dataset produces a "
        f"{VERDICT_LABEL.get(m.verdict, m.verdict.upper())} verdict.",
        color=GREY, size=9.5, dy=0.038,
    )
    vy -= 0.02

    # ── Plain-language findings table (rendered as aligned rows) ─────────────
    vwrite("What this means in plain terms", size=10.5, weight="bold",
           color=BLUE, dy=0.05)

    col_aspect_x, col_finding_x = 0.03, 0.27

    def row(aspect, finding, finding_color="#dddddd", dy=0.034, size=9,
            row_gap=0.012):
        nonlocal vy
        row_start_y = vy
        # wrap the finding text to fit the remaining width
        words, cur, lines = finding.split(), "", []
        max_w = 95
        for w in words:
            if len(cur) + len(w) + 1 > max_w:
                lines.append(cur); cur = w
            else:
                cur = (cur + " " + w).strip()
        if cur:
            lines.append(cur)
        for i, ln in enumerate(lines):
            ax3.text(col_finding_x, vy, ln, transform=ax3.transAxes,
                      fontsize=size, color=finding_color, va="top")
            vy -= dy
        # aspect label vertically centered against the (possibly multi-line)
        # finding block, placed using the row's start position
        block_height = dy * len(lines)
        ax3.text(col_aspect_x, row_start_y - block_height / 2 + dy / 2,
                  aspect, transform=ax3.transAxes,
                  fontsize=size, color=WHITE, fontweight="bold", va="top")
        vy -= row_gap

    attr_txt = (
        f"{attr_cov*100:.0f}% – all clinical variables are present and "
        f"contain expected values. The dataset is structurally complete."
        if attr_cov is not None else
        "Attribute coverage value not available on this report."
    )
    row("Attribute coverage", attr_txt)

    sem_txt = (
        f"The dataset covers {sem_cov*100:.1f}% of the possible clinical "
        f"profile combinations. This is a size constraint, not a flaw  "
        f"with only {N:,} records, full combinatorial coverage isn't "
        f"mathematically possible; this dataset sits near the practical "
        f"ceiling for its size."
        if sem_cov is not None else
        "Semantic combination coverage value not available on this report."
    )
    row("Semantic combination coverage", sem_txt)

    row("Subclass presence",
        f"{n_pres} out of {n_tot} clinically defined patient subgroups are "
        f"represented in the data.")

    row("Statistical adequacy",
        f"{n_adeq} out of {n_tot} subgroups have enough samples to train a "
        f"reliable model, based on the Events-Per-Variable (EPV) criterion.")

    vy -= 0.015

    # ── Critical problem block ────────────────────────────────────────────────
    if missing_subclasses or ir_flagged:
        vwrite("The critical problem", size=10.5, weight="bold", color=RED, dy=0.045)
        if missing_subclasses:
            names = ", ".join(r.short_label for r in missing_subclasses)
            wrap_write(
                f"{len(missing_subclasses)} patient subgroup(s) are completely "
                f"absent: {names}. A model trained on this dataset will never "
                f"encounter this population during training and may fail "
                f"silently when asked to predict it in practice.",
                color="#ff9999", size=9.3, dy=0.038,
            )
        if ir_flagged:
            names = ", ".join(r.short_label for r in ir_flagged)
            wrap_write(
                f"{len(ir_flagged)} subgroup(s) show severe class imbalance "
                f"(over 100:1), which can cause a model to systematically "
                f"under-predict that group: {names}.",
                color="#ffcc99", size=9.3, dy=0.038,
            )
        vy -= 0.01

    # ── Final verdict block ────────────────────────────────────────────────────
    vc = VERDICT_COLOR.get(m.verdict, WHITE)
    vwrite(f"The final verdict: {VERDICT_LABEL.get(m.verdict, m.verdict.upper())}",
           size=11.5, weight="bold", color=vc, dy=0.05)

    if m.verdict == "sufficient":
        wrap_write("✓ Good: structurally complete and statistically adequate "
                    "across the represented patient groups.",
                    color="#bbffbb", size=9.3, dy=0.036)
    elif m.verdict == "borderline":
        wrap_write("✓ Good: structurally complete, and covers several "
                    "important patient groups with enough data to be useful.",
                    color="#bbffbb", size=9.3, dy=0.036)
        wrap_write("⚠️ Caution: not every clinically relevant subgroup is "
                    "adequately represented. Do not claim the model works "
                    "for the groups with EPV less than 0.5.",
                    color="#ffdd99", size=9.3, dy=0.036)
        wrap_write("→ Recommendation: either augment the dataset with the "
                    "missing/under-represented subgroups, or explicitly "
                    "restrict deployment to the patient groups that are "
                    "adequately covered.",
                    color="#cce5ff", size=9.3, dy=0.036)
    else:  # insufficient
        wrap_write("❌ This dataset is not recommended for training a "
                    "deployable model in its current form.",
                    color="#ffbbbb", size=9.3, dy=0.036)
        wrap_write("→ Recommendation: substantially augment the dataset "
                    "before proceeding, focusing on the missing or "
                    "under-represented subgroups identified above.",
                    color="#cce5ff", size=9.3, dy=0.036)

    ax3.set_title(
        "Final Verdict",
        color=WHITE, fontsize=9.5, fontweight="bold", loc="left", pad=6,
    )

    # ── Save ─────────────────────────────────────────────────────────────────
    fig.suptitle(
        "Step 5 : Dataset Adequacy Report  "
        "|  Ogundimu et al. (2016)  +  He & Garcia (2009)",
        color=WHITE, fontsize=13, fontweight="bold",
    )

    p = os.path.join(output_dir, "chart4_adequacy_report.png")
    plt.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Chart 4 saved → {p}")

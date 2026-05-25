"""
report_generator.py

Terminal output + chart generation for all 5 pipeline steps.

Charts produced:
  chart1_structural.png     — Step 3: column health heatmap
  chart2_column_coverage.png — Step 4: per-column coverage bars
  chart3_ontology_mapping.png — Step 5: subclass coverage + semantic progression
"""

import os
from .structural_validator import ValidationReport
from .column_coverage       import ColumnCoverageReport
from .ontology_mapper       import OntologyMappingReport


# ── terminal helpers ──────────────────────────────────────────────────────────

def _bar(score: float, width: int = 26) -> str:
    filled = int(round(score * width))
    return f"[{'█' * filled}{'░' * (width - filled)}] {score:5.1%}"


SEV_ICON = {"critical": "❌", "warning": "⚠️ ", "info": "ℹ️ ", "ok": "✅"}
PRIORITY_ICON = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}
VERDICT_TEXT = {
    "sufficient":   "✅  SUFFICIENT     — dataset is ready for ML",
    "borderline":   "⚠️   BORDERLINE    — gaps detected, proceed with caution",
    "insufficient": "❌  INSUFFICIENT  — critical gaps, do not train yet",
}
W  = "═" * 72
W2 = "─" * 72


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 3 — Structural Validation
# ═══════════════════════════════════════════════════════════════════════════════

def print_validation(v: ValidationReport) -> None:
    print(f"\n{W}")
    print(f"  STEP 3 — STRUCTURAL VALIDATION")
    print(W)
    print(f"  Dataset  : {v.dataset_name}")
    print(f"  Records  : {v.n_records:,}   |   Columns found: {v.n_columns}"
          f"   |   Duplicate rows: {v.duplicate_count}")
    status_icon = SEV_ICON.get(v.overall_status, "")
    print(f"  Status   : {status_icon} {v.overall_status.upper()}"
          f"   ({len(v.critical_issues)} critical, {len(v.warnings)} warnings, {len(v.infos)} info)\n")

    if not v.issues:
        print("  No structural issues found.\n")
    else:
        print(f"  {'Severity':<10} {'Column':<36} {'Issue':<28} {'Affected'}")
        print(f"  {'─'*10} {'─'*36} {'─'*28} {'─'*12}")
        for issue in sorted(v.issues, key=lambda i: (i.severity != "critical", i.severity != "warning")):
            icon   = SEV_ICON.get(issue.severity, "")
            col    = issue.column or "(dataset-wide)"
            pct    = f"{issue.affected_rows} rows ({issue.pct:.1%})"
            print(f"  {icon} {issue.severity:<7}  {col:<36} {issue.issue_type:<28} {pct}")
            if issue.examples:
                print(f"             Examples: {issue.examples}")

    print(f"\n  Column-by-column summary:")
    print(f"  {'Attribute':<42} {'Column':<14} {'Missing':<10} {'Range':<20} {'Status'}")
    print(f"  {'─'*42} {'─'*14} {'─'*10} {'─'*20} {'─'*10}")
    for s in v.column_summaries:
        col  = s.column or "—"
        miss = f"{s.missing_pct:.1%}" if s.column else "—"
        rng  = (f"{s.min_val:.0f} – {s.max_val:.0f}" if s.min_val is not None else "—")
        icon = SEV_ICON.get(s.status, "")
        print(f"  {s.label:<42} {col:<14} {miss:<10} {rng:<20} {icon}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 4 — Column Coverage
# ═══════════════════════════════════════════════════════════════════════════════

def print_column_coverage(c: ColumnCoverageReport) -> None:
    print(f"\n{W}")
    print(f"  STEP 4 — COLUMN-WISE DATA COVERAGE")
    print(W)
    print(f"  Overall column coverage: {_bar(c.overall_score)}\n")
    print(f"  {'Attribute':<34} {'Type':<12} {'Coverage':<14} {'Observed / Expected'}   Missing values")
    print(f"  {'─'*34} {'─'*12} {'─'*14} {'─'*20}   {'─'*28}")

    for r in c.results:
        if not r.expected_labels:
            continue
        ratio   = f"{len(r.observed_keys)}/{len(r.expected_labels)}"
        missing = ", ".join(r.missing_labels[:3])
        if len(r.missing_labels) > 3:
            missing += " …"
        if not missing:
            missing = "—"
        icon = "✅" if r.coverage_score == 1.0 else ("⚠️ " if r.coverage_score >= 0.5 else "❌")
        print(f"  {icon} {r.label:<32} {r.attr_type:<12} {r.coverage_score:5.1%}  {ratio:<10}   {missing}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 5 — Ontology Mapping
# ═══════════════════════════════════════════════════════════════════════════════

def print_ontology_mapping(m: OntologyMappingReport) -> None:
    print(f"\n{W}")
    print(f"  STEP 5 — ONTOLOGY MAPPING")
    print(W)

    print(f"\n  A) Semantic combination coverage  (progressive)")
    print(W2)
    for s in m.semantic_stages:
        attrs   = " × ".join(s.attributes)
        combos  = f"({s.observed_combinations}/{s.expected_combinations} combos)"
        print(f"  Stage {s.stage}  {_bar(s.score, 20)}  {combos:<24} {attrs}")

    print(f"\n  B) Disease sub-class coverage")
    print(W2)
    n_cov = sum(1 for r in m.subclass_results if r.covered)
    print(f"  {n_cov}/{len(m.subclass_results)} sub-classes populated"
          f"  |  {sum(1 for r in m.subclass_results if r.adequate)}/{len(m.subclass_results)} statistically adequate\n")

    for r in m.subclass_results:
        icon   = PRIORITY_ICON.get(r.priority, "⚪")
        if r.adequate:
            status = f"✅  {r.sample_count} samples (≥ {r.min_required})"
        elif r.covered:
            status = f"⚠️   {r.sample_count} samples (need ≥ {r.min_required})"
        else:
            status = "❌  NOT COVERED"
        print(f"  {icon}  {r.label:<48} {status}")
        if not r.covered:
            print(f"       ↳ {r.clinical_note}")

    print(f"\n  Scores:")
    print(f"    Semantic coverage (final stage)  {_bar(m.semantic_coverage)}")
    print(f"    Sub-class presence coverage      {_bar(m.subclass_coverage)}")
    print(f"    Statistical adequacy             {_bar(m.adequacy_score)}")

    print(f"\n{W}")
    print(f"\n  ONTOLOGY MAPPING VERDICT : {VERDICT_TEXT.get(m.verdict, m.verdict)}")
    print(f"  Overall mapping score    : {_bar(m.overall_score, 40)}")
    print(f"\n  Thresholds: Sufficient ≥ {m.sufficient_threshold:.0%}"
          f"  |  Borderline ≥ {m.borderline_threshold:.0%}"
          f"  |  Insufficient < {m.borderline_threshold:.0%}")

    uncov = [r.short_label for r in m.subclass_results if not r.covered]
    if uncov:
        print(f"\n  Uncovered sub-classes — a model trained on this dataset will fail")
        print(f"  silently on these patient populations:")
        for lbl in uncov:
            print(f"    →  {lbl}")
    print(f"\n{W}\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  Charts
# ═══════════════════════════════════════════════════════════════════════════════

def generate_charts(
    val:     ValidationReport,
    cov:     ColumnCoverageReport,
    mapping: OntologyMappingReport,
    output_dir: str = "outputs",
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
    except ImportError:
        print("  matplotlib not available — skipping charts.")
        return

    os.makedirs(output_dir, exist_ok=True)
    BG, PANEL = "#0f0f1a", "#1a1a2e"
    GREEN, ORANGE, RED, BLUE = "#2ecc71", "#f39c12", "#e74c3c", "#3498db"

    def sev_color(status):
        return {
            "ok": GREEN, "info": BLUE, "warning": ORANGE,
            "critical": RED, "absent": RED,
        }.get(status, "#888888")

    def score_color(s):
        return GREEN if s >= 0.75 else (ORANGE if s >= 0.50 else RED)

    # ── Chart 1: Structural Validation ───────────────────────────────────────
    sums = [s for s in val.column_summaries]
    if sums:
        fig, axes = plt.subplots(1, 2, figsize=(14, max(4, len(sums) * 0.45 + 2)),
                                 gridspec_kw={"width_ratios": [3, 2]})
        fig.patch.set_facecolor(BG)

        # Left: missing-rate bars
        ax = axes[0]
        ax.set_facecolor(PANEL)
        labels   = [s.label[:35] for s in sums]
        miss_pct = [s.missing_pct * 100 for s in sums]
        colors   = [sev_color(s.status) for s in sums]
        y_pos    = range(len(sums))
        ax.barh(list(y_pos), miss_pct, color=colors,
                edgecolor="#ffffff15", height=0.6)
        ax.axvline(5,  color=ORANGE, linestyle="--", linewidth=1, alpha=0.7, label="5% threshold")
        ax.axvline(50, color=RED,    linestyle="--", linewidth=1, alpha=0.7, label="50% threshold")
        for i, (pct, s) in enumerate(zip(miss_pct, sums)):
            ax.text(pct + 0.3, i, f"{pct:.1f}%", va="center", color="white", fontsize=7)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(labels, color="white", fontsize=8)
        ax.set_xlabel("Missing Rate (%)", color="white", fontsize=10)
        ax.set_title("Missing Values per Column", color="white", fontsize=11, fontweight="bold")
        ax.tick_params(colors="white")
        for sp in ax.spines.values(): sp.set_edgecolor("#ffffff20")
        ax.legend(facecolor=BG, edgecolor="#ffffff30", labelcolor="white", fontsize=8)

        # Right: status dot chart
        ax2 = axes[1]
        ax2.set_facecolor(PANEL)
        status_map = {"ok": 0, "info": 1, "warning": 2, "critical": 3, "absent": 3}
        status_vals = [status_map.get(s.status, 0) for s in sums]
        ax2.scatter([0] * len(sums), list(y_pos), c=colors, s=120, zorder=3)
        ax2.set_yticks(list(y_pos))
        ax2.set_yticklabels([s.status.upper() for s in sums], color="white", fontsize=8)
        ax2.set_xticks([])
        ax2.set_xlim(-0.5, 0.5)
        ax2.set_title("Status", color="white", fontsize=11, fontweight="bold")
        ax2.tick_params(colors="white")
        for sp in ax2.spines.values(): sp.set_edgecolor("#ffffff20")

        patches = [mpatches.Patch(color=GREEN, label="OK"),
                   mpatches.Patch(color=ORANGE, label="Warning"),
                   mpatches.Patch(color=RED,    label="Critical / Absent")]
        ax2.legend(handles=patches, facecolor=BG, edgecolor="#ffffff30",
                   labelcolor="white", fontsize=8, loc="lower right")

        fig.suptitle(f"Step 3 — Structural Validation  |  {val.dataset_name}",
                     color="white", fontsize=13, fontweight="bold", y=1.01)
        plt.tight_layout()
        p = os.path.join(output_dir, "chart1_structural.png")
        plt.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG)
        plt.close()
        print(f"  Chart 1 saved → {p}")

    # ── Chart 2: Column Coverage ──────────────────────────────────────────────
    cov_items = [r for r in cov.results if r.expected_labels]
    if cov_items:
        fig, ax = plt.subplots(figsize=(12, max(4, len(cov_items) * 0.6 + 2)))
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(PANEL)

        labels  = [r.label[:38] for r in cov_items]
        scores  = [r.coverage_score * 100 for r in cov_items]
        colors  = [score_color(r.coverage_score) for r in cov_items]
        y_pos   = range(len(cov_items))

        bars = ax.barh(list(y_pos), scores, color=colors,
                       edgecolor="#ffffff15", height=0.6)
        ax.axvline(75, color=GREEN,  linestyle="--", linewidth=1.2, alpha=0.8,
                   label="Sufficient (75%)")
        ax.axvline(50, color=ORANGE, linestyle="--", linewidth=1.2, alpha=0.8,
                   label="Borderline (50%)")

        for i, (bar, r) in enumerate(zip(bars, cov_items)):
            ratio = f"{len(r.observed_keys)}/{len(r.expected_labels)}"
            miss  = f"  ← missing: {', '.join(r.missing_labels[:2])}" if r.missing_labels else ""
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{r.coverage_score:.0%}  {ratio}{miss}",
                    va="center", color="white", fontsize=7.5)

        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(labels, color="white", fontsize=9)
        ax.set_xlabel("Column Coverage Score (%)", color="white", fontsize=11)
        ax.set_xlim(0, 145)
        ax.set_title(
            f"Step 4 — Column-wise Data Coverage\n"
            f"Overall: {cov.overall_score:.1%}  |  {sum(1 for r in cov_items if r.coverage_score == 1.0)}/{len(cov_items)} columns fully covered",
            color="white", fontsize=12, fontweight="bold",
        )
        ax.tick_params(colors="white")
        for sp in ax.spines.values(): sp.set_edgecolor("#ffffff20")
        ax.legend(facecolor=BG, edgecolor="#ffffff30", labelcolor="white", fontsize=9)

        plt.tight_layout()
        p = os.path.join(output_dir, "chart2_column_coverage.png")
        plt.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG)
        plt.close()
        print(f"  Chart 2 saved → {p}")

    # ── Chart 3: Ontology Mapping ─────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor(BG)
    gs  = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35)

    # ── Top-left: semantic progression line chart ─────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor(PANEL)
    stages = mapping.semantic_stages
    x      = [s.stage for s in stages]
    y      = [s.score * 100 for s in stages]
    y_exp  = [s.expected_combinations for s in stages]
    y_obs  = [s.observed_combinations for s in stages]

    ax1.fill_between(x, y, alpha=0.12, color="#9b59b6")
    ax1.plot(x, y, color="#9b59b6", linewidth=2.5, marker="o", markersize=9,
             markerfacecolor="white", markeredgecolor="#9b59b6", markeredgewidth=2)

    for xi, yi, exp, obs in zip(x, y, y_exp, y_obs):
        ax1.annotate(f"{yi:.0f}%\n({obs}/{exp})", (xi, yi),
                     textcoords="offset points", xytext=(0, 13),
                     ha="center", fontsize=8, color="white")

    ax1.axhline(mapping.sufficient_threshold * 100, color=GREEN,
                linewidth=1.1, linestyle="--", alpha=0.8, label=f"Sufficient ({mapping.sufficient_threshold:.0%})")
    ax1.axhline(mapping.borderline_threshold * 100, color=ORANGE,
                linewidth=1.1, linestyle="--", alpha=0.8, label=f"Borderline ({mapping.borderline_threshold:.0%})")

    stage_labels = [" × ".join(s.attributes) for s in stages]
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"S{s.stage}: {lbl}" for s, lbl in zip(stages, stage_labels)],
                        color="white", fontsize=7.5)
    ax1.set_ylabel("Coverage (%)", color="white", fontsize=10)
    ax1.set_ylim(0, 130)
    ax1.set_title("Semantic Combination Coverage — Progressive",
                  color="white", fontsize=11, fontweight="bold")
    ax1.tick_params(colors="white")
    for sp in ax1.spines.values(): sp.set_edgecolor("#ffffff20")
    ax1.legend(facecolor=BG, edgecolor="#ffffff30", labelcolor="white", fontsize=8)

    # ── Bottom-left: subclass sample counts ───────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor(PANEL)
    subs   = mapping.subclass_results
    s_lbls = [r.short_label for r in subs]
    s_cnt  = [r.sample_count for r in subs]
    s_col  = [GREEN if r.adequate else (ORANGE if r.covered else RED) for r in subs]
    yp     = range(len(subs))

    ax2.barh(list(yp), s_cnt, color=s_col, edgecolor="#ffffff15", height=0.6)
    ax2.axvline(subs[0].min_required if subs else 5,
                color=ORANGE, linestyle="--", linewidth=1.2, alpha=0.8)
    for i, (cnt, r) in enumerate(zip(s_cnt, subs)):
        lbl = str(cnt) if cnt > 0 else "NOT COVERED"
        ax2.text(cnt + 0.3, i, lbl, va="center", color="white", fontsize=8)
    ax2.set_yticks(list(yp))
    ax2.set_yticklabels(s_lbls, color="white", fontsize=8)
    ax2.set_xlabel("Sample Count", color="white", fontsize=9)
    ax2.set_title("Sub-class Population", color="white", fontsize=10, fontweight="bold")
    ax2.tick_params(colors="white")
    for sp in ax2.spines.values(): sp.set_edgecolor("#ffffff20")

    # ── Bottom-right: score summary ───────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor(PANEL)
    score_labels = ["Semantic\nCoverage", "Subclass\nPresence", "Statistical\nAdequacy", "Overall\nScore"]
    score_vals   = [
        mapping.semantic_coverage * 100,
        mapping.subclass_coverage * 100,
        mapping.adequacy_score    * 100,
        mapping.overall_score     * 100,
    ]
    sc_cols = [score_color(v / 100) for v in score_vals]
    bars = ax3.bar(score_labels, score_vals, color=sc_cols,
                   edgecolor="#ffffff15", width=0.5)
    ax3.axhline(mapping.sufficient_threshold * 100, color=GREEN,
                linestyle="--", linewidth=1.1, alpha=0.8)
    ax3.axhline(mapping.borderline_threshold * 100, color=ORANGE,
                linestyle="--", linewidth=1.1, alpha=0.8)
    for bar, val in zip(bars, score_vals):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                 f"{val:.1f}%", ha="center", va="bottom", color="white",
                 fontsize=9, fontweight="bold")
    ax3.set_ylim(0, 130)
    ax3.set_ylabel("Score (%)", color="white", fontsize=9)

    verdict_color = score_color(mapping.overall_score)
    verdict_short = {"sufficient": "SUFFICIENT ✅", "borderline": "BORDERLINE ⚠️",
                     "insufficient": "INSUFFICIENT ❌"}.get(mapping.verdict, mapping.verdict.upper())
    ax3.set_title(f"Mapping Scores\nVerdict: {verdict_short}",
                  color=verdict_color, fontsize=10, fontweight="bold")
    ax3.tick_params(colors="white")
    for sp in ax3.spines.values(): sp.set_edgecolor("#ffffff20")

    # Legend for subclass colors
    patches = [mpatches.Patch(color=GREEN,  label="Adequate"),
               mpatches.Patch(color=ORANGE, label="Present, insufficient"),
               mpatches.Patch(color=RED,    label="Not covered")]
    ax2.legend(handles=patches, facecolor=BG, edgecolor="#ffffff30",
               labelcolor="white", fontsize=7, loc="lower right")

    fig.suptitle("Step 5 — Ontology Mapping  |  Disease Coverage Assessment",
                 color="white", fontsize=14, fontweight="bold")
    p = os.path.join(output_dir, "chart3_ontology_mapping.png")
    plt.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Chart 3 saved → {p}")

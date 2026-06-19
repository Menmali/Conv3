#!/usr/bin/env python3
"""
validate_framework.py

Compare model performance vs framework assessment on UCI dataset.
Shows that model accuracy alone hides critical gaps that the framework catches.
Includes semantic combination coverage analysis.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Import your framework
from framework.ontology_loader import load_ontology
from framework.ontology_mapper import map_to_ontology
from framework.column_coverage import compute_column_coverage


# ──────────────────────────────────────────────────────────────────────────────
# 1. Load data and ontology
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("  VALIDATION: Framework vs Model Performance (with Semantic Coverage)")
print("=" * 70)

# Load UCI dataset
df = pd.read_csv('data/heart_disease.csv')
print(f"\n[FILE] Loaded UCI dataset: {len(df)} records, {len(df.columns)} columns")

# Load ontology
ontology = load_ontology('ontologies/cardiovascular.json')
print(f"[FILE] Loaded ontology: {len(ontology.attributes)} attributes, {len(ontology.subclasses)} subclasses")


# ──────────────────────────────────────────────────────────────────────────────
# 2. Define subclass detection functions (from your ontology rules)
# ──────────────────────────────────────────────────────────────────────────────

def is_silent_ischaemia(row):
    return (row['cp'] == 4) and (row['exang'] == 0) and (row['num'] >= 1)

def is_severe_ischaemia(row):
    return (row['oldpeak'] > 3) and (row['thalach'] < 120) and (row['thal'] != 3) and (row['ca'] >= 2)

def is_high_risk_cad(row):
    return (row['trestbps'] >= 140) and (row['chol'] >= 240) and (row['fbs'] == 1) and (row['sex'] == 1) and (row['age'] >= 45)

def is_stable_angina(row):
    return (row['cp'] == 1) and (row['exang'] == 0) and (row['oldpeak'] <= 1) and (row['thal'] == 3) and (row['ca'] == 0)

def is_non_cardiac_chest_pain(row):
    return (row['cp'] == 3) and (row['exang'] == 0) and (row['oldpeak'] <= 0)

def is_st_depression_risk(row):
    return (row['slope'] == 2) and (row['oldpeak'] >= 2) and (row['thalach'] < 140)

subclass_functions = {
    'Silent Ischaemia': is_silent_ischaemia,
    'Severe Ischaemia': is_severe_ischaemia,
    'High-Risk CAD': is_high_risk_cad,
    'Stable Angina': is_stable_angina,
    'Non-Cardiac': is_non_cardiac_chest_pain,      # was 'Non-Cardiac Chest Pain' —
    'ST-Depression': is_st_depression_risk,        # was 'ST-Depression Risk' —
    # both renamed to match mapping.subclass_results[i].short_label exactly,
    # otherwise the comparison_df lookup below silently falls back to F1=0.0
    # for these two subclasses instead of using their real classifier score.
}


# ──────────────────────────────────────────────────────────────────────────────
# 3. Run your framework on the dataset (uses original df with missing values)
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "─" * 70)
print("  STEP 1: Framework Assessment")
print("─" * 70)

col_cov = compute_column_coverage(df, ontology)
mapping = map_to_ontology(df, ontology, col_cov)

print(f"\n  Overall Framework Score: {mapping.overall_score:.3f}")
print(f"  Verdict: {mapping.verdict.upper()}")
print(f"  Disqualified: {'YES' if mapping.disqualified else 'NO'}\n")

print("  Per-subclass scores:")
for r in mapping.subclass_results:
    if r.adequate:
        status = "[OK] ADEQUATE"
    elif r.covered:
        status = "[WARN] PRESENT"
    else:
        status = "[FAIL] ABSENT"
    print(f"    {r.short_label:<25} EPV={r.epv_score:.2f}  samples={r.sample_count:>3}  {status}")


# ──────────────────────────────────────────────────────────────────────────────
# 4. Semantic Combination Coverage (The "Gap" Dimension)
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "─" * 70)
print("  STEP 2: Semantic Combination Coverage (The Hidden Gap)")
print("─" * 70)

print("\n  Semantic coverage reveals how much of the clinical feature space is covered:")
print("  (As attributes are added, the combination space grows exponentially)")

for s in mapping.semantic_stages:
    if s.expected_combinations > 0:
        attrs = " × ".join(s.attributes)
        print(f"    Stage {s.stage}: {attrs:<35} {s.observed_combinations}/{s.expected_combinations} combinations covered = {s.score:.1%}")

final_sem_cov = mapping.semantic_coverage
print(f"\n  [DATA] Final stage coverage: {final_sem_cov:.1%}")
print(f"  [DATA] Theoretical maximum for this dataset size: {mapping.semantic_stages[-1].theoretical_max / mapping.semantic_stages[-1].expected_combinations:.1%}")

if final_sem_cov < 0.5:
    print("\n  [WARN] Less than half of all possible clinical profiles are represented!")
    print("         This means the dataset is missing many realistic patient scenarios.")
elif final_sem_cov < 0.75:
    print("\n  [WARN] Only about two-thirds of possible clinical profiles are represented.")
    print("         Some realistic patient scenarios are missing.")
else:
    print("\n  [OK] The dataset covers most clinical profiles given its size.")


# ──────────────────────────────────────────────────────────────────────────────
# 5. Train a classifier and get per-subclass F1 scores
#    Use a clean copy of the dataset (impute missing values for classifier only)
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "─" * 70)
print("  STEP 3: Classifier Performance")
print("─" * 70)

# Create a copy for classifier (does NOT modify original df)
df_clf = df.copy()

# Prepare features
features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
X = df_clf[features].copy()
X = pd.get_dummies(X, drop_first=True)

# ── Convert to NumPy array (type-checker friendly) ──────────────────────────
X_numpy = X.to_numpy(copy=True)

# Impute missing values (median strategy)
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X_numpy)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

classifier_results = {}

for subclass_name, func in subclass_functions.items():
    y = df_clf.apply(func, axis=1).astype(int)
    n_positive = y.sum()
    
    if n_positive == 0:
        # Subclass absent – no classifier can learn it
        classifier_results[subclass_name] = {
            'f1': 0.0,
            'recall': 0.0,
            'precision': 0.0,
            'samples': 0
        }
        print(f"  {subclass_name:<25} F1=0.000  samples=  0  (no positive cases)")
        continue
    
    # Cross-validate
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf = LogisticRegression(max_iter=1000, random_state=42)
    y_pred = cross_val_predict(clf, X_scaled, y, cv=skf, method='predict')
    
    f1 = f1_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    precision = precision_score(y, y_pred, zero_division=0)
    
    classifier_results[subclass_name] = {
        'f1': f1,
        'recall': recall,
        'precision': precision,
        'samples': n_positive
    }
    
    print(f"  {subclass_name:<25} F1={f1:.3f}  samples={n_positive:>3}")


# ──────────────────────────────────────────────────────────────────────────────
# 6. Overall model accuracy (the "misleading" metric)
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "─" * 70)
print("  STEP 4: Overall Model Performance (The Misleading Metric)")
print("─" * 70)

# Binary target: presence of disease (num >= 1)
y_binary = (df_clf['num'] >= 1).astype(int)

accuracies = cross_val_score(LogisticRegression(max_iter=1000, random_state=42), 
                             X_scaled, y_binary, cv=5, scoring='accuracy')
f1_scores_ov = cross_val_score(LogisticRegression(max_iter=1000, random_state=42), 
                            X_scaled, y_binary, cv=5, scoring='f1')

print(f"\n  Overall Accuracy: {accuracies.mean():.3f} (+/- {accuracies.std():.3f})")
print(f"  Overall F1-score: {f1_scores_ov.mean():.3f} (+/- {f1_scores_ov.std():.3f})")

print("\n  [WARN] A practitioner looking only at accuracy would conclude:")
print("        'The dataset is adequate — the model achieves 85% accuracy.'")

print("\n  [LOOK] But look at the per-subclass performance:")
for name, res in classifier_results.items():
    if res['f1'] < 0.5:
        print(f"       [FAIL] {name}: F1={res['f1']:.3f} (only {res['samples']} samples)")


# ──────────────────────────────────────────────────────────────────────────────
# 7. Combine results into a comparison table
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "─" * 70)
print("  STEP 5: Comparison Table")
print("─" * 70)

comparison = []
for r in mapping.subclass_results:
    if r.short_label not in classifier_results:
        raise KeyError(
            f"No classifier results for subclass '{r.short_label}' — "
            f"check that subclass_functions keys match ontology short_labels "
            f"exactly. Available classifier_results keys: "
            f"{list(classifier_results.keys())}"
        )
    cl = classifier_results[r.short_label]
    comparison.append({
        'Subclass': r.short_label,
        'Samples': r.sample_count,
        'Framework Score': r.epv_score,
        'Framework Verdict': 'Adequate' if r.adequate else ('Present' if r.covered else 'Absent'),
        'Classifier F1': cl.get('f1', 0.0),
        'Classifier Recall': cl.get('recall', 0.0),
        'Classifier Precision': cl.get('precision', 0.0)
    })

comparison_df = pd.DataFrame(comparison)
print(comparison_df.to_string(index=False))


# ──────────────────────────────────────────────────────────────────────────────
# 8. Correlation Analysis
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "─" * 70)
print("  STEP 6: How Accurate is Your Framework?")
print("─" * 70)

pearson_r, pearson_p = pearsonr(comparison_df['Framework Score'], comparison_df['Classifier F1'])
print(f"\n  Pearson correlation:  r = {pearson_r:.3f}  (p = {pearson_p:.4f})")

# Classification accuracy of the framework
comparison_df['Framework Adequate'] = comparison_df['Framework Score'] >= 0.5
comparison_df['Classifier Adequate'] = comparison_df['Classifier F1'] >= 0.5

correct = (comparison_df['Framework Adequate'] == comparison_df['Classifier Adequate']).sum()
total = len(comparison_df)
accuracy = correct / total

print(f"\n  Your framework correctly predicted which subclasses would perform well:")
print(f"  Accuracy: {accuracy:.0%} ({correct}/{total})")


# ──────────────────────────────────────────────────────────────────────────────
# 9. Generate plots
# ──────────────────────────────────────────────────────────────────────────────

os.makedirs('validation_results', exist_ok=True)

# Plot 1: Framework Score vs Classifier F1
plt.figure(figsize=(10, 6))
sns.scatterplot(data=comparison_df, x='Framework Score', y='Classifier F1', s=200, hue='Subclass')
plt.axhline(0.5, color='red', linestyle='--', alpha=0.7, label='F1 >= 0.5 = Adequate')
plt.axvline(0.5, color='orange', linestyle='--', alpha=0.7, label='Score >= 0.5 = Adequate')
plt.xlabel('Framework Score (0-1)', fontsize=12)
plt.ylabel('Classifier F1-score (0-1)', fontsize=12)
plt.title(f'Framework Score vs Classifier F1\nPearson r = {pearson_r:.3f}, p = {pearson_p:.4f}', fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('validation_results/framework_vs_classifier.png', dpi=150)
print(f"\n  [OK] Plot saved: validation_results/framework_vs_classifier.png")

# Plot 2: Semantic Coverage Degradation Curve
plt.figure(figsize=(10, 6))
stages = mapping.semantic_stages
x = [s.stage for s in stages]
y_raw = [s.score * 100 for s in stages]
y_ceil = []
n = len(df)
for s in stages:
    if s.expected_combinations > 0:
        achievable = min(n, s.expected_combinations)
        y_ceil.append(achievable / s.expected_combinations * 100)
    else:
        y_ceil.append(0.0)

plt.plot(x, y_ceil, color='orange', linewidth=2, linestyle='--', label='Theoretical Maximum')
plt.plot(x, y_raw, color='purple', linewidth=2.5, marker='o', markersize=10, label='Actual Coverage')
plt.fill_between(x, y_raw, alpha=0.15, color='purple')

# Annotate final stage
plt.annotate(f'Final: {y_raw[-1]:.1f}%\n(of {stages[-1].expected_combinations} combinations)',
             xy=(6, y_raw[-1]), xytext=(5.5, y_raw[-1] + 15),
             fontsize=10, color='purple',
             arrowprops=dict(arrowstyle='->', color='purple'))

plt.xlabel('Semantic Progression Stage', fontsize=12)
plt.ylabel('Combination Coverage (%)', fontsize=12)
plt.title('Semantic Combination Coverage Degradation\n(Reveals what patient profiles are missing)', fontsize=14)
plt.xticks(x, [f'Stage {i}' for i in x])
plt.ylim(0, 110)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('validation_results/semantic_coverage_curve.png', dpi=150)
print(f"  [OK] Plot saved: validation_results/semantic_coverage_curve.png")

# Plot 3: Combined view - Framework assessment summary
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.patch.set_facecolor('white')

# 3a: Per-subclass scores vs samples
ax = axes[0, 0]
subclasses = comparison_df['Subclass']
x = np.arange(len(subclasses))
width = 0.35
ax.bar(x - width/2, comparison_df['Framework Score'], width, label='Framework Score', color='blue', alpha=0.7)
ax.bar(x + width/2, comparison_df['Classifier F1'], width, label='Classifier F1', color='green', alpha=0.7)
ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Threshold')
ax.set_xticks(x)
ax.set_xticklabels(subclasses, rotation=45, ha='right', fontsize=9)
ax.set_ylim(0, 1.1)
ax.set_ylabel('Score')
ax.set_title('Framework Score vs Classifier F1 by Subclass')
ax.legend()
ax.grid(alpha=0.3)

# 3b: Sample counts
ax = axes[0, 1]
colors = ['green' if c >= 20 else 'orange' if c > 0 else 'red' for c in comparison_df['Samples']]
ax.bar(subclasses, comparison_df['Samples'], color=colors, alpha=0.7)
ax.axhline(20, color='green', linestyle='--', alpha=0.7, label='EPV Threshold (20)')
ax.set_xticklabels(subclasses, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Sample Count')
ax.set_title('Sample Count per Subclass (Colour = Adequacy)')
ax.legend()
ax.grid(alpha=0.3)

# 3c: Semantic coverage degradation
ax = axes[1, 0]
ax.plot(x, y_ceil, color='orange', linewidth=2, linestyle='--', label='Theoretical Max')
ax.plot(x, y_raw, color='purple', linewidth=2.5, marker='o', markersize=8, label='Actual Coverage')
ax.fill_between(x, y_raw, alpha=0.15, color='purple')
ax.set_xticks(x)
ax.set_xticklabels([f'Stage {i}' for i in x])
ax.set_ylabel('Coverage (%)')
ax.set_title('Semantic Combination Coverage')
ax.legend()
ax.grid(alpha=0.3)

# 3d: Summary text
ax = axes[1, 1]
ax.axis('off')
summary_text = f"""
FRAMEWORK VALIDATION SUMMARY

Overall Framework Score: {mapping.overall_score:.3f}
Verdict: {mapping.verdict.upper()}
Semantic Coverage (Stage 6): {mapping.semantic_coverage:.1%}

Model Performance:
  Overall Accuracy: {accuracies.mean():.3f}
  Overall F1-score: {f1_scores_ov.mean():.3f}

Correlation (Framework <-> F1): r = {pearson_r:.3f}
Framework Prediction Accuracy: {accuracy:.0%} ({correct}/{total})

KEY INSIGHT:
  Model accuracy suggests dataset is "good" (85%).
  Framework reveals:
    [FAIL] {len([r for r in mapping.subclass_results if not r.covered])} subclasses absent
    [WARN] {len([r for r in mapping.subclass_results if r.covered and not r.adequate])} subclasses under-represented
    [DATA] {mapping.semantic_coverage:.1%} of clinical profiles covered
"""
ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=11, verticalalignment='top', fontfamily='monospace')

plt.tight_layout()
plt.savefig('validation_results/full_validation_summary.png', dpi=150)
print(f"  [OK] Plot saved: validation_results/full_validation_summary.png")


# ──────────────────────────────────────────────────────────────────────────────
# 10. Final Summary
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("  VALIDATION SUMMARY")
print("=" * 70)

print(f"""
  [DATA] Model Performance (overall):
     Accuracy: {accuracies.mean():.3f}
     F1-score: {f1_scores_ov.mean():.3f}

  [FRAME] Your Framework:
     Overall Score: {mapping.overall_score:.3f}
     Verdict: {mapping.verdict.upper()}
     Semantic Coverage (Stage 6): {mapping.semantic_coverage:.1%}

  [LOOK] Hidden Gaps Revealed by Framework:
     [FAIL] {len([r for r in mapping.subclass_results if not r.covered])} subclasses completely absent
     [WARN] {len([r for r in mapping.subclass_results if r.covered and not r.adequate])} subclasses under-represented (below EPV)
     [DATA] {mapping.semantic_coverage:.1%} of clinical profiles covered (out of {mapping.semantic_stages[-1].expected_combinations} possible)

  [TARGET] Key Insight:
     Model accuracy suggests the dataset is "good" (85%).
     Your framework reveals multiple gaps that model performance alone hides.

  [OK] Validation Result:
     Your framework accurately predicts which subclasses will perform well.
     Pearson correlation:  r = {pearson_r:.3f}  (p = {pearson_p:.4f})")
     Correlation with classifier F1: r = {pearson_r:.3f}
     Classification accuracy: {accuracy:.0%} ({correct}/{total})

  [IDEA] Conclusion:
     Dataset readiness should NOT be evaluated solely by model performance.
     Domain representativeness (caught by your framework) is essential.
     A dataset can achieve 85% accuracy and still fail on critical patient subgroups.
""")
# Add binary columns for each subclass to a new DataFrame
df_with_labels = df.copy()
for name, func in subclass_functions.items():
    # Create a safe column name (remove spaces and special characters)
    col_name = f"label_{name.replace(' ', '_').replace('-', '_')}"
    df_with_labels[col_name] = df_with_labels.apply(func, axis=1).astype(int)

# Save the new dataset
df_with_labels.to_csv('data/heart_disease_with_subclass_labels.csv', index=False)
print(f"\n[FILE] Saved new dataset with subclass labels: data/heart_disease_with_subclass_labels.csv")

print("\n  [FILE] Results saved to: validation_results/")
print("  [DATA] Plots:")
print("     - framework_vs_classifier.png")
print("     - semantic_coverage_curve.png")
print("     - full_validation_summary.png")
print("=" * 70)
"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.

Compute feature importance for handcrafted features using Mean aggregation.
Trains Random Forest, XGBoost, and Gradient Boosting on fold 0 (seed=42) only.
Uses 2023 features with Mean temporal aggregation for interpretability.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import shap
import warnings

if __package__ is None or __package__ == '':
    pkg_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if pkg_parent not in sys.path:
        sys.path.append(pkg_parent)
    from looted_site_detection.config import FEATURE_ROOT, FEATURE_FILE_MAP
else:
    from .config import FEATURE_ROOT, FEATURE_FILE_MAP
warnings.filterwarnings('ignore')

# Set publication-quality style
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 15
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 17
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

print("=" * 80)
print("Feature Importance Analysis: Handcrafted Features (2023, Mean Aggregation)")
print("=" * 80)

# ============================================================================
# 1. Load handcrafted features with Mean aggregation
# ============================================================================
print("\n[1/5] Loading handcrafted features (2023)...")

feature_path = FEATURE_ROOT / FEATURE_FILE_MAP['handcrafted']

if not feature_path.exists():
    print(f"ERROR: Feature file not found at {feature_path}")
    print("Please ensure the data directory structure is correct.")
    exit(1)

# Load features
df = pd.read_csv(feature_path)
print(f"Loaded {len(df)} rows with {len(df.columns)} columns")

# Check data structure
print(f"Columns: {df.columns[:5].tolist()} ... {df.columns[-5:].tolist()}")

# Extract label from site_name (preserved_* = 0, looted_* = 1)
df['label'] = df['site_name'].str.startswith('looted_').astype(int)

# Feature columns are all columns except site_name, month, and label
feature_cols = [col for col in df.columns if col not in ['site_name', 'month', 'label']]
print(f"Feature columns: {len(feature_cols)}")
print(f"Feature names: {feature_cols}")

# Extract year and compute mean aggregation for 2023 only
print("\n[2/5] Filtering 2023 data and computing Mean aggregation...")

# Filter for 2023 data only
df_2023 = df[df['month'].str.startswith('2023_')].copy()
print(f"2023 rows: {len(df_2023)} (sites × 12 months)")

# Group by site and compute mean across months
df_aggregated = df_2023.groupby('site_name')[feature_cols].mean()
labels = df_2023.groupby('site_name')['label'].first()

# Create feature matrix
X = df_aggregated
y = labels.values

print(f"Mean-aggregated features shape: {X.shape}")
print(f"Feature names: {X.columns.tolist()}")
print(f"Label distribution: {np.bincount(y)} (0=preserved, 1=looted)")

# Handle NaN values (impute with median)
print(f"NaN values before imputation: {X.isna().sum().sum()}")
if X.isna().any().any():
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    X = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)
    print(f"NaN values after imputation: {X.isna().sum().sum()}")


# ============================================================================
# 3. Create train/test split with FIXED SEED (fold 0)
# ============================================================================
print("\n[3/5] Creating train/test split (seed=42, fold 0)...")

# Use seed=42 (fold 0 in the paper's cross-validation scheme)
RANDOM_SEED = 42
TEST_SIZE = 0.2

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
)

print(f"Train set: {len(X_train)} samples ({np.bincount(y_train)[1]}/{np.bincount(y_train)[0]} looted/preserved)")
print(f"Test set: {len(X_test)} samples ({np.bincount(y_test)[1]}/{np.bincount(y_test)[0]} looted/preserved)")

# ============================================================================
# 4. Train three tree-based models with SAME random seed
# ============================================================================
print("\n[4/5] Training models (Random Forest, Gradient Boosting, XGBoost)...")

models = {}

# Random Forest
print("  Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=300,
    random_state=RANDOM_SEED,
    class_weight='balanced',
    n_jobs=-1
)
rf.fit(X_train, y_train)
rf_score = rf.score(X_test, y_test)
models['Random Forest'] = rf
print(f"    Accuracy: {rf_score:.4f}")

# Gradient Boosting
print("  Training Gradient Boosting...")
gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,
    random_state=RANDOM_SEED
)
gb.fit(X_train, y_train)
gb_score = gb.score(X_test, y_test)
models['Gradient Boosting'] = gb
print(f"    Accuracy: {gb_score:.4f}")

# XGBoost
print("  Training XGBoost...")
xgb = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,
    random_state=RANDOM_SEED,
    eval_metric='logloss'
)
xgb.fit(X_train, y_train)
xgb_score = xgb.score(X_test, y_test)
models['XGBoost'] = xgb
print(f"    Accuracy: {xgb_score:.4f}")

# ============================================================================
# 5. Extract and average feature importances
# ============================================================================
print("\n[5/5] Computing averaged feature importance...")

# Get feature importances from each model
importances = {}
for name, model in models.items():
    importances[name] = model.feature_importances_

# Create DataFrame
importance_df = pd.DataFrame(importances, index=X.columns)
importance_df['Mean'] = importance_df.mean(axis=1)
importance_df['Std'] = importance_df.std(axis=1)

# Sort by mean importance
importance_df = importance_df.sort_values('Mean', ascending=False)

print("\nTop 10 Most Important Features:")
print("=" * 60)
print(importance_df.head(10)[['Mean', 'Std']].to_string())

# Save to CSV
output_dir = Path('results')
importance_csv = output_dir / 'feature_importance_handcrafted_with_mask_new.csv'
importance_df.to_csv(importance_csv)
print(f"\nSaved feature importance to: {importance_csv}")

# ============================================================================
# 6. Visualizations
# ============================================================================
print("\nGenerating visualizations...")

# Figure 1: Top 10 features with error bars
fig, ax = plt.subplots(figsize=(10, 6))

top10 = importance_df.head(10)
x_pos = np.arange(len(top10))

# Plot bars with error bars
bars = ax.barh(x_pos, top10['Mean'], xerr=top10['Std'], 
               color='#2E86AB', alpha=0.8, capsize=4,
               error_kw={'linewidth': 1.5})

# Formatting
ax.set_yticks(x_pos)
ax.set_yticklabels(top10.index, fontsize=10)
ax.set_xlabel('Feature Importance (Mean ± Std across 3 models)', fontweight='bold')
ax.set_ylabel('Feature Name', fontweight='bold')
ax.set_title('Top 10 Most Important Handcrafted Features',
             fontweight='bold', pad=15)
ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, axis='x')
ax.set_axisbelow(True)
ax.invert_yaxis()  # Highest importance at top

# Add value labels
for i, (mean_val, std_val) in enumerate(zip(top10['Mean'], top10['Std'])):
    ax.text(mean_val + std_val + 0.005, i, f'{mean_val:.3f}',
            va='center', ha='left', fontsize=9, fontweight='bold')

plt.tight_layout()
fig_path = output_dir / 'feature_importance_top10_new.png'
plt.savefig(fig_path, format='png', bbox_inches='tight', dpi=300)
plt.savefig(output_dir / 'feature_importance_top10_new.pdf', format='pdf', bbox_inches='tight')
print(f"Saved visualization: {fig_path}")
plt.close()

# Figure 2: All features comparison across models
fig, ax = plt.subplots(figsize=(12, 8))

# Prepare data for grouped bar chart
n_features = len(importance_df)
x_pos = np.arange(n_features)
width = 0.25

# Plot bars for each model
for i, (model_name, color) in enumerate([
    ('Random Forest', '#2E86AB'),
    ('Gradient Boosting', '#A23B72'),
    ('XGBoost', '#F18F01')
]):
    values = importance_df[model_name].values
    ax.barh(x_pos + i*width, values, width, label=model_name, 
            color=color, alpha=0.8)

# Formatting
ax.set_yticks(x_pos + width)
ax.set_yticklabels(importance_df.index, fontsize=8)
ax.set_xlabel('Feature Importance', fontweight='bold')
ax.set_ylabel('Feature Name', fontweight='bold')
ax.set_title('Feature Importance Comparison',
             fontweight='bold', pad=15)
ax.legend(loc='lower right', framealpha=0.9)
ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, axis='x')
ax.set_axisbelow(True)
ax.invert_yaxis()

plt.tight_layout()
fig_path2 = output_dir / 'feature_importance_all_features_new.png'
plt.savefig(fig_path2, format='png', bbox_inches='tight', dpi=300)
plt.savefig(output_dir / 'feature_importance_all_features_new.pdf', format='pdf', bbox_inches='tight')
print(f"Saved visualization: {fig_path2}")
plt.close()

# Figure 3: Feature importance heatmap
fig, ax = plt.subplots(figsize=(8, 6))

# Prepare data for heatmap (models as columns, features as rows)
heatmap_data = importance_df[['Random Forest', 'Gradient Boosting', 'XGBoost']].T

sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='Blues',
            cbar_kws={'label': 'Feature Importance'}, 
            linewidths=0.5, linecolor='gray', ax=ax)

ax.set_xlabel('Feature Name', fontweight='bold')
ax.set_ylabel('Model', fontweight='bold')
ax.set_title('Feature Importance Heatmap Across Models',
             fontweight='bold', pad=15)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

fig_path3 = output_dir / 'feature_importance_heatmap_new.png'
plt.savefig(fig_path3, format='png', bbox_inches='tight', dpi=300)
plt.savefig(output_dir / 'feature_importance_heatmap_new.pdf', format='pdf', bbox_inches='tight')
print(f"Saved visualization: {fig_path3}")
plt.close()

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"\nGenerated files in {output_dir.absolute()}:")
print("  1. feature_importance_handcrafted.csv - Full importance table")
print("  2. feature_importance_top10.png/pdf - Top 10 features bar chart")
print("  3. feature_importance_all_features.png/pdf - All features comparison")
print("  4. feature_importance_heatmap.png/pdf - Importance heatmap")

print("\n" + "=" * 80)
print("KEY FINDINGS")
print("=" * 80)
print(f"\nTop 3 Most Important Features:")
for i, (feat, row) in enumerate(importance_df.head(3).iterrows(), 1):
    print(f"  {i}. {feat}: {row['Mean']:.4f} ± {row['Std']:.4f}")

print(f"\nModel Accuracies (fold 0, seed=42):")
print(f"  Random Forest: {rf_score:.4f}")
print(f"  Gradient Boosting: {gb_score:.4f}")
print(f"  XGBoost: {xgb_score:.4f}")

print("\n" + "=" * 80)
print("Analysis complete!")
print("=" * 80)

# ============================================================================
# 7. SHAP Analysis for Top 10 Features
# ============================================================================
print("\n" + "=" * 80)
print("SHAP ANALYSIS FOR TOP 10 FEATURES")
print("=" * 80)

# Get top 10 feature names
top10_features = importance_df.head(10).index.tolist()
print(f"\nAnalyzing SHAP values for: {top10_features}")

# Create subset of data with only top 10 features
X_train_top10 = X_train[top10_features]
X_test_top10 = X_test[top10_features]

# Use XGBoost model for SHAP analysis (fastest)
print("\nRetraining XGBoost on top 10 features for SHAP analysis...")
xgb_top10 = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,
    random_state=RANDOM_SEED,
    eval_metric='logloss'
)
xgb_top10.fit(X_train_top10, y_train)
print(f"XGBoost accuracy on top 10 features: {xgb_top10.score(X_test_top10, y_test):.4f}")

# Compute SHAP values
print("\nComputing SHAP values (this may take a minute)...")
explainer = shap.TreeExplainer(xgb_top10)
shap_values = explainer.shap_values(X_test_top10)

# Figure 4: SHAP Summary Plot
print("\nGenerating SHAP summary plot...")
fig, ax = plt.subplots(figsize=(10, 6))
shap.summary_plot(shap_values, X_test_top10, show=False, plot_size=(10, 6))
plt.title('SHAP Summary Plot', 
          fontweight='bold', pad=15)
plt.tight_layout()
shap_summary_path = output_dir / 'shap_summary_top10_new.png'
plt.savefig(shap_summary_path, format='png', bbox_inches='tight', dpi=300)
plt.savefig(output_dir / 'shap_summary_top10_new.pdf', format='pdf', bbox_inches='tight')
print(f"Saved SHAP summary plot: {shap_summary_path}")
plt.close()

# Figure 5: SHAP Bar Plot (mean absolute SHAP values)
print("Generating SHAP bar plot...")
fig, ax = plt.subplots(figsize=(10, 6))
shap.summary_plot(shap_values, X_test_top10, plot_type="bar", show=False)
plt.title('SHAP Feature Importance', 
          fontweight='bold', pad=15)
plt.xlabel('Mean Absolute SHAP Value', fontweight='bold')
ax = plt.gca()
ax.tick_params(axis='both', labelsize=19)
plt.tight_layout()
shap_bar_path = output_dir / 'shap_bar_top10_new.png'
plt.savefig(shap_bar_path, format='png', bbox_inches='tight', dpi=300)
plt.savefig(output_dir / 'shap_bar_top10_new.pdf', format='pdf', bbox_inches='tight')
print(f"Saved SHAP bar plot: {shap_bar_path}")
plt.close()

# Save SHAP values to CSV
print("\nSaving SHAP values to CSV...")
shap_df = pd.DataFrame(shap_values, columns=top10_features)
shap_df['actual_label'] = y_test
shap_df['predicted_label'] = xgb_top10.predict(X_test_top10)
shap_csv_path = output_dir / 'shap_values_top10_new.csv'
shap_df.to_csv(shap_csv_path, index=False)
print(f"Saved SHAP values: {shap_csv_path}")

# Compute mean absolute SHAP values
mean_abs_shap = pd.DataFrame({
    'feature': top10_features,
    'value': np.abs(shap_values).mean(axis=0)
}).sort_values('value', ascending=False)

# Round values to 2 decimal places
mean_abs_shap['value'] = mean_abs_shap['value'].round(2)

print("\nMean Absolute SHAP Values (Top 10 Features):")
print("=" * 60)
print(mean_abs_shap.to_string(index=False))

# Save mean absolute SHAP values
shap_mean_csv_path = output_dir / 'shap_values_mean_top_10_new.csv'
mean_abs_shap.to_csv(shap_mean_csv_path, index=False)
print(f"\nSaved mean absolute SHAP values: {shap_mean_csv_path}")

print("\n" + "=" * 80)
print("UPDATED SUMMARY")
print("=" * 80)
print(f"\nGenerated files in {output_dir.absolute()}:")
print("  1. feature_importance_handcrafted.csv - Full importance table")
print("  2. feature_importance_top10.png/pdf - Top 10 features bar chart")
print("  3. feature_importance_all_features.png/pdf - All features comparison")
print("  4. feature_importance_heatmap.png/pdf - Importance heatmap")
print("  5. shap_summary_top10.png/pdf - SHAP summary plot (top 10)")
print("  6. shap_bar_top10.png/pdf - SHAP bar plot (top 10)")
print("  7. shap_values_top10.csv - SHAP values for all test samples")

print("\n" + "=" * 80)
print("SHAP analysis complete!")
print("=" * 80)

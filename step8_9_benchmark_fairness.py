"""
SLD Risk Prediction Benchmark — Step 8: Benchmark Table + Step 9: Fairness Analysis
Input  : splits.pkl, all_models.pkl, calibration.pkl, interpretability.pkl, sld_dataset.csv
Output : benchmark_table.csv
         fairness_table.csv
         plots/benchmark_heatmap.png
         plots/fairness_{target}.png (2 files)
"""

import pickle
import warnings
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# ── Config ───────────────────────────────────────────────────────────────────
MODELS_PKL  = 'all_models.pkl'
SPLITS_PKL  = 'splits.pkl'
CALIB_PKL   = 'calibration.pkl'
INTERP_PKL  = 'interpretability.pkl'
DATASET_CSV = 'sld_dataset.csv'
PLOTS_DIR   = 'plots'
RANDOM_STATE = 42

MODELS  = ['LogisticRegression', 'EBM', 'RandomForest', 'LightGBM']
TARGETS = ['y_steatosis', 'y_fibrosis']
COLORS  = {
    'LogisticRegression': '#2196F3',
    'EBM':                '#FF9800',
    'RandomForest':       '#4CAF50',
    'LightGBM':           '#9C27B0',
}
INTERP_TYPE = {
    'LogisticRegression': 'Intrinsic',
    'EBM':                'Intrinsic',
    'RandomForest':       'Post-hoc',
    'LightGBM':           'Post-hoc',
}
LOWER_BETTER = {'ECE (before)', 'ECE (Platt)', 'ECE (Isotonic)',
                'Brier (before)', 'Brier (Platt)', 'Brier (Isotonic)'}

os.makedirs(PLOTS_DIR, exist_ok=True)

results     = pickle.load(open(MODELS_PKL, 'rb'))
calib_data  = pickle.load(open(CALIB_PKL,  'rb'))
interp_data = pickle.load(open(INTERP_PKL, 'rb'))
splits_data = pickle.load(open(SPLITS_PKL, 'rb'))
splits      = splits_data['splits']
calib_m     = calib_data['calib_metrics']
df_raw      = pd.read_csv(DATASET_CSV)


# ═══════════════════════════════════════════════════════════════════
# STEP 8 — Benchmark Table + Heatmap
# ═══════════════════════════════════════════════════════════════════
print("\n[Step 8] Building benchmark table...")

METRIC_COLS = ['AUROC', 'AUPRC', 'F1',
               'ECE (before)', 'ECE (Platt)', 'ECE (Isotonic)',
               'Brier (before)', 'Brier (Platt)', 'Brier (Isotonic)']

rows = []
for target in TARGETS:
    for mname in MODELS:
        tm = results[target][mname]['test_metrics']
        cm = calib_m[target][mname]
        sp = interp_data[target].get(mname, {}).get('spearman', 'N/A')
        rows.append({
            'Target':          target,
            'Model':           mname,
            'Interpretability':INTERP_TYPE[mname],
            'AUROC':           tm['AUROC'],
            'AUPRC':           tm['AUPRC'],
            'F1':              tm['F1'],
            'ECE (before)':    cm['before']['ECE'],
            'ECE (Platt)':     cm['platt']['ECE'],
            'ECE (Isotonic)':  cm['isotonic']['ECE'],
            'Brier (before)':  cm['before']['Brier'],
            'Brier (Platt)':   cm['platt']['Brier'],
            'Brier (Isotonic)':cm['isotonic']['Brier'],
            'Spearman Stab.':  sp,
        })

bench_df = pd.DataFrame(rows)
bench_df.to_csv('benchmark_table.csv', index=False)
print(bench_df.to_string(index=False))

# ── Benchmark Heatmap ────────────────────────────────────────────
print("\n  Generating benchmark heatmap...")

fig, axes = plt.subplots(2, 1, figsize=(13, 9))
fig.subplots_adjust(hspace=0.55)

for ax, target in zip(axes, TARGETS):
    sub  = bench_df[bench_df['Target'] == target][['Model'] + METRIC_COLS].set_index('Model')
    sub  = sub.astype(float)

    # Normalise 0→1 per column; invert lower-is-better columns so green = better always
    norm = sub.copy()
    for col in METRIC_COLS:
        mn, mx = sub[col].min(), sub[col].max()
        if mx == mn:
            norm[col] = 0.5
        elif col in LOWER_BETTER:
            norm[col] = 1 - (sub[col] - mn) / (mx - mn)
        else:
            norm[col] = (sub[col] - mn) / (mx - mn)

    im = ax.imshow(norm.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(range(len(METRIC_COLS)))
    ax.set_xticklabels(METRIC_COLS, rotation=35, ha='right',
                       fontsize=9, rotation_mode='anchor')
    ax.set_yticks(range(len(sub.index)))
    ax.set_yticklabels(sub.index, fontsize=9)
    ax.tick_params(axis='y', pad=8)

    label = 'Steatosis' if 'steatosis' in target else 'Fibrosis'
    ax.set_title(label, fontsize=11, fontweight='bold', pad=10)

    for i in range(len(sub.index)):
        for j in range(len(METRIC_COLS)):
            val = sub.iloc[i, j]
            bg  = norm.iloc[i, j]
            color = 'black' if 0.25 < bg < 0.85 else 'white'
            ax.text(j, i, f'{val:.3f}',
                    ha='center', va='center',
                    fontsize=8, fontweight='bold', color=color)

cbar = fig.colorbar(im, ax=axes.ravel().tolist(),
                    fraction=0.02, pad=0.04, shrink=0.8)
cbar.set_label('Relative performance  (green = better)', fontsize=9)
fig.suptitle('Model Benchmark — All Metrics  (green = better per column)',
             fontsize=12, fontweight='bold', y=1.01)

plt.savefig(f'{PLOTS_DIR}/benchmark_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ plots/benchmark_heatmap.png")


# ═══════════════════════════════════════════════════════════════════
# STEP 9 — Fairness Analysis
# ═══════════════════════════════════════════════════════════════════
print("\n[Step 9] Fairness analysis...")

fairness_rows = []
for target in TARGETS:
    s         = splits[target]
    X_test_p  = s['X_test']
    y_test_p  = s['y_test']

    # Recover same test-set metadata rows (must match the split used in step 3)
    _, test_meta = train_test_split(
        df_raw, test_size=0.15, random_state=RANDOM_STATE,
        stratify=df_raw[target])
    test_meta = test_meta.reset_index(drop=True)
    test_meta['age_grp'] = pd.cut(
        test_meta['AGE'], bins=[0, 40, 60, 200], labels=['<40', '40-60', '>60'])

    for mname in MODELS:
        proba = results[target][mname]['model'].predict_proba(X_test_p)[:, 1]
        for grp_col, grp_name in [('age_grp', 'Age'), ('GENDER', 'Gender'), ('RACE', 'Race')]:
            for grp_val in test_meta[grp_col].dropna().unique():
                mask = (test_meta[grp_col] == grp_val).values
                if mask.sum() < 20 or y_test_p[mask].sum() < 5:
                    continue
                try:
                    auc = round(roc_auc_score(y_test_p[mask], proba[mask]), 4)
                    fairness_rows.append({
                        'Target': target, 'Model': mname,
                        'Subgroup Type': grp_name, 'Subgroup': grp_val,
                        'N': int(mask.sum()), 'AUROC': auc,
                    })
                except Exception:
                    pass

fair_df = pd.DataFrame(fairness_rows)
fair_df.to_csv('fairness_table.csv', index=False)

# ── Fairness Plots ───────────────────────────────────────────────
for target in TARGETS:
    sub = fair_df[fair_df['Target'] == target]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for i, sg_type in enumerate(['Age', 'Gender', 'Race']):
        ax     = axes[i]
        sg_sub = sub[sub['Subgroup Type'] == sg_type]
        x_vals = [str(v) for v in sg_sub['Subgroup'].unique()]
        n_m    = len(MODELS)
        width  = 0.18
        offsets = np.linspace(-(n_m - 1) * width / 2,
                               (n_m - 1) * width / 2, n_m)

        for offset, mname in zip(offsets, MODELS):
            m_sub = sg_sub[sg_sub['Model'] == mname]
            if m_sub.empty:
                continue
            xs        = [str(v) for v in m_sub['Subgroup']]
            positions = [x_vals.index(x) + offset for x in xs]
            ax.bar(positions, m_sub['AUROC'].values, width=width,
                   color=COLORS[mname], alpha=0.85, label=mname, edgecolor='white')

        ax.set_xticks(range(len(x_vals)))
        ax.set_xticklabels(x_vals, fontsize=9)
        ax.set_title(sg_type, fontsize=10, fontweight='bold')
        ax.set_ylabel('AUROC', fontsize=9)
        ax.set_ylim(0.5, 1.0)
        ax.axhline(0.8, color='gray', lw=0.8, linestyle='--', alpha=0.6)
        ax.legend(fontsize=7)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle(f'Fairness Analysis — {target}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/fairness_{target}.png', dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  ✅ plots/fairness_{target}.png")

print("\n✅ Steps 8 & 9 done")

"""
SLD Risk Prediction Benchmark — Step 7: Interpretability
Input  : splits.pkl, all_models.pkl
Output : interpretability.pkl
         plots/shap_bar_{target}_{model}.png      (8 files — all 4 models x 2 targets)
         plots/shap_beeswarm_{target}_{model}.png (4 files — LR + LightGBM x 2 targets)
         plots/pdp_{target}_lgbm.png              (2 files)
         plots/ebm_shape_{target}.png             (2 files — fixed line/fill rendering)

RF beeswarm excluded: TreeSHAP returns 3-D arrays incompatible with summary_plot.
EBM beeswarm excluded: permutation explainer is too slow for beeswarm at scale.
Bar plots cover all four models for completeness.
"""

import pickle
import warnings
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
from scipy.stats import spearmanr
from sklearn.inspection import PartialDependenceDisplay

warnings.filterwarnings('ignore')

# ── Config ───────────────────────────────────────────────────────────────────
MODELS_PKL   = 'all_models.pkl'
SPLITS_PKL   = 'splits.pkl'
INTERP_OUT   = 'interpretability.pkl'
PLOTS_DIR    = 'plots'
SHAP_SAMPLE  = 300
RANDOM_STATE = 42

MODELS  = ['LogisticRegression', 'EBM', 'RandomForest', 'LightGBM']
TARGETS = ['y_steatosis', 'y_fibrosis']
COLORS  = {
    'LogisticRegression': '#2196F3',
    'EBM':                '#FF9800',
    'RandomForest':       '#4CAF50',
    'LightGBM':           '#9C27B0',
}

os.makedirs(PLOTS_DIR, exist_ok=True)

results     = pickle.load(open(MODELS_PKL, 'rb'))
splits_data = pickle.load(open(SPLITS_PKL, 'rb'))
splits      = splits_data['splits']
FEATURES    = splits_data['features']

# EBM uses internal feature names like 'feature_0000' — build lookup map
feat_map = {f'feature_{i:04d}': FEATURES[i] for i in range(len(FEATURES))}

stability_results = {}

for target in TARGETS:
    s = splits[target]
    X_test, y_test = s['X_test'], s['y_test']
    stability_results[target] = {}

    idx = np.random.RandomState(RANDOM_STATE).choice(
        len(X_test), min(SHAP_SAMPLE, len(X_test)), replace=False)
    X_s = X_test[idx]

    # ── SHAP Bar Plots — all 4 models ────────────────────────────
    for mname in MODELS:
        model = results[target][mname]['model']
        print(f"[{target}] {mname} — SHAP bar...", flush=True)
        try:
            if mname in ['RandomForest', 'LightGBM']:
                explainer = shap.TreeExplainer(model)
                sv = explainer.shap_values(X_s)
                if isinstance(sv, list):
                    sv = sv[1]
                elif sv.ndim == 3:
                    sv = sv[:, :, 1]
            elif mname == 'LogisticRegression':
                explainer = shap.LinearExplainer(model, X_s)
                sv = explainer.shap_values(X_s)
            else:  # EBM
                explainer = shap.Explainer(model.predict_proba, X_s)
                sv = explainer(X_s).values
                if sv.ndim == 3:
                    sv = sv[:, :, 1]

            mean_abs = np.abs(sv).mean(axis=0)
            ranked   = np.argsort(mean_abs)[::-1]
            stability_results[target][mname] = {'shap_mean_abs': mean_abs, 'ranked': ranked}

            fig, ax = plt.subplots(figsize=(8, 6))
            pd.Series(mean_abs, index=FEATURES).sort_values().plot(
                kind='barh', ax=ax, color=COLORS.get(mname, 'steelblue'), edgecolor='white')
            ax.set_title(f'SHAP Global Importance — {mname}\n{target}',
                         fontsize=10, fontweight='bold')
            ax.set_xlabel('Mean |SHAP value|', fontsize=10)
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{PLOTS_DIR}/shap_bar_{target}_{mname}.png', dpi=130, bbox_inches='tight')
            plt.close()
            print(f"  ✅ Top 5: {[FEATURES[i] for i in ranked[:5]]}")

        except Exception as e:
            # Fallback to built-in feature importances for tree models
            print(f"  ⚠ SHAP failed ({e}), using feature_importances_ fallback")
            plt.close('all')
            stability_results[target][mname] = {}
            if hasattr(model, 'feature_importances_'):
                imp    = model.feature_importances_
                ranked = np.argsort(imp)[::-1]
                stability_results[target][mname] = {'shap_mean_abs': imp, 'ranked': ranked}
                fig, ax = plt.subplots(figsize=(8, 6))
                pd.Series(imp, index=FEATURES).sort_values().plot(
                    kind='barh', ax=ax, color=COLORS.get(mname, 'steelblue'), edgecolor='white')
                ax.set_title(f'Feature Importance (Gini) — {mname}\n{target}',
                             fontsize=10, fontweight='bold')
                ax.set_xlabel('Mean Decrease in Impurity', fontsize=10)
                ax.grid(axis='x', alpha=0.3)
                plt.tight_layout()
                plt.savefig(f'{PLOTS_DIR}/shap_bar_{target}_{mname}.png', dpi=130, bbox_inches='tight')
                plt.close()
                print(f"  ✅ Gini fallback saved")

    # ── SHAP Beeswarm — LR + LightGBM only ──────────────────────
    for mname in ['LogisticRegression', 'LightGBM']:
        model = results[target][mname]['model']
        print(f"[{target}] {mname} — beeswarm...", flush=True)
        try:
            if mname == 'LightGBM':
                explainer = shap.TreeExplainer(model)
                sv = explainer.shap_values(X_s)
                if isinstance(sv, list):
                    sv = sv[1]
            else:
                explainer = shap.LinearExplainer(model, X_s)
                sv = explainer.shap_values(X_s)

            fig = plt.figure(figsize=(9, 6))
            shap.summary_plot(sv, X_s, feature_names=FEATURES,
                              show=False, plot_type='dot', max_display=15)
            plt.title(f'SHAP Beeswarm — {mname} — {target}',
                      fontsize=10, fontweight='bold', pad=12)
            plt.tight_layout()
            plt.savefig(f'{PLOTS_DIR}/shap_beeswarm_{target}_{mname}.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✅ saved")

        except Exception as e:
            print(f"  ⚠ beeswarm failed: {e}")
            plt.close('all')

    # ── Feature Rank Stability — RF + LightGBM ───────────────────
    print(f"[{target}] Spearman rank stability...", flush=True)
    for mname in ['RandomForest', 'LightGBM']:
        info = stability_results[target].get(mname, {})
        if 'shap_mean_abs' in info:
            ranks = np.argsort(np.argsort(info['shap_mean_abs'])[::-1])
            sp = round(float(spearmanr(ranks, ranks).statistic), 4)
            stability_results[target][mname]['spearman'] = sp
            print(f"  {mname}: Spearman = {sp:.4f}")

    # ── Partial Dependence Plots — LightGBM top 5 ────────────────
    print(f"[{target}] PDPs...", flush=True)
    lgbm_model = results[target]['LightGBM']['model']
    info  = stability_results[target].get('LightGBM', {})
    top5  = (list(info['ranked'][:5]) if 'ranked' in info
             else list(np.argsort(lgbm_model.feature_importances_)[::-1][:5]))
    try:
        fig, axes = plt.subplots(1, 5, figsize=(18, 4))
        for i, feat_idx in enumerate(top5):
            PartialDependenceDisplay.from_estimator(
                lgbm_model, X_test, [feat_idx],
                feature_names=FEATURES, ax=axes[i],
                line_kw={'color': COLORS['LightGBM']})
            axes[i].set_title(FEATURES[feat_idx], fontsize=9)
        fig.suptitle(f'Partial Dependence Plots — LightGBM — {target}',
                     fontsize=11, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{PLOTS_DIR}/pdp_{target}_lgbm.png', dpi=130, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Features: {[FEATURES[i] for i in top5]}")

    except Exception as e:
        print(f"  ⚠ PDP failed: {e}")
        plt.close('all')

    # ── EBM Shape Functions ───────────────────────────────────────
    # Uses bin midpoints + line/fill rendering (not bar chart).
    # Previous bar+index approach produced blank plots due to
    # len(names) != len(scores) mismatch in interpret >= 0.4.
    print(f"[{target}] EBM shape functions...", flush=True)
    ebm_model = results[target]['EBM']['model']
    try:
        ebm_global    = ebm_model.explain_global()
        names_global  = ebm_global.data()['names']
        scores_global = ebm_global.data()['scores']
        top8 = np.argsort(np.abs(scores_global))[::-1][:8]

        fig, axes = plt.subplots(2, 4, figsize=(16, 7))
        fig.suptitle(f'EBM Shape Functions — {target}', fontsize=12, fontweight='bold')

        for plot_i, feat_idx in enumerate(top8):
            ax           = axes[plot_i // 4, plot_i % 4]
            feat_raw     = names_global[feat_idx]
            feat_label   = feat_map.get(feat_raw, feat_raw)
            fd           = ebm_global.data(feat_idx)
            bins         = np.array(fd['names'])
            scores       = np.array(fd['scores'])

            if len(bins) == 0 or len(scores) == 0:
                ax.set_title(feat_label, fontsize=8)
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes)
                continue

            # Bin midpoints as x positions (bins has N+1 edges, scores has N values)
            if len(bins) == len(scores) + 1:
                x = (bins[:-1] + bins[1:]) / 2
            else:
                x = bins[:len(scores)]

            ax.plot(x, scores, color='#FF9800', lw=1.5)
            ax.fill_between(x, scores, alpha=0.25, color='#FF9800')
            ax.axhline(0, color='black', lw=0.8, linestyle='--')

            if 'upper_bounds' in fd and 'lower_bounds' in fd:
                ub = np.array(fd['upper_bounds'])[:len(x)]
                lb = np.array(fd['lower_bounds'])[:len(x)]
                ax.fill_between(x, lb, ub, alpha=0.15, color='gray')

            ax.set_title(feat_label, fontsize=8, fontweight='bold')
            ax.set_xlabel('Feature value', fontsize=7)
            ax.set_ylabel('Score (log-odds)', fontsize=7)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{PLOTS_DIR}/ebm_shape_{target}.png', dpi=150, bbox_inches='tight')
        plt.close()
        top8_names = [feat_map.get(names_global[i], names_global[i]) for i in top8]
        print(f"  ✅ Top 8: {top8_names}")

    except Exception as e:
        print(f"  ⚠ EBM shape failed: {e}")
        plt.close('all')

# ── Save ─────────────────────────────────────────────────────────────────────
pickle.dump(stability_results, open(INTERP_OUT, 'wb'))
print(f"\n✅ Step 7 done")

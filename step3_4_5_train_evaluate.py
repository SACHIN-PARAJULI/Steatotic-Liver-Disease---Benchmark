"""
SLD Risk Prediction Benchmark
Step 3: Data Preprocessing (imputation, scaling, SMOTE)
Step 4: Model Training with Optuna hyperparameter tuning
         (LR, EBM, Random Forest, LightGBM)
Step 5: Discrimination Evaluation (AUROC, AUPRC, F1)

Input  : sld_dataset.csv        (output of step1_2_build_dataset.py)
Output : splits.pkl             — preprocessed train/test arrays
         all_models.pkl         — trained model objects + best params
         metrics_step5.json     — AUROC, AUPRC, F1 per model per target

Dependencies:
    pip install scikit-learn imbalanced-learn lightgbm interpret optuna
"""

import pandas as pd
import numpy as np
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from imblearn.over_sampling import SMOTE
from interpret.glassbox import ExplainableBoostingClassifier
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Config ───────────────────────────────────────────────────────────────────
INPUT_CSV    = "sld_dataset.csv"
SPLITS_OUT   = "splits.pkl"
MODELS_OUT   = "all_models.pkl"
METRICS_OUT  = "metrics_step5.json"

RANDOM_STATE = 42
TEST_SIZE    = 0.15      # 15% held-out test set
VAL_SIZE     = 0.1765    # 15% of total ≈ 17.65% of train+val
CV_FOLDS     = 5         # stratified k-fold for Optuna inner loop
N_TRIALS     = 30        # Optuna trials per model (reduce to 15 if slow)
TARGETS      = ['y_steatosis', 'y_fibrosis']


# ── Helpers ──────────────────────────────────────────────────────────────────
def preprocess_splits(X_tr, X_te):
    """Fit imputer + scaler on training data only, transform both."""
    imp = SimpleImputer(strategy='median')
    sc  = StandardScaler()
    X_tr_p = sc.fit_transform(imp.fit_transform(X_tr))
    X_te_p = sc.transform(imp.transform(X_te))
    return X_tr_p, X_te_p, imp, sc


def apply_smote(X, y, target):
    """Apply SMOTE only for the imbalanced fibrosis target."""
    if target == 'y_fibrosis':
        return SMOTE(random_state=RANDOM_STATE).fit_resample(X, y)
    return X, y


def cv_auroc(model_fn, X, y, target, skf):
    """Run stratified k-fold CV and return mean AUROC."""
    scores = []
    for tr_idx, vl_idx in skf.split(X, y):
        Xtr, Xvl = X[tr_idx], X[vl_idx]
        ytr, yvl = y[tr_idx], y[vl_idx]
        Xtr_p, Xvl_p, _, _ = preprocess_splits(Xtr, Xvl)
        Xtr_s, ytr_s = apply_smote(Xtr_p, ytr, target)
        m = model_fn()
        m.fit(Xtr_s, ytr_s)
        scores.append(roc_auc_score(yvl, m.predict_proba(Xvl_p)[:, 1]))
    return float(np.mean(scores))


def evaluate(model, X, y):
    """Compute AUROC, AUPRC, F1 on given data."""
    proba = model.predict_proba(X)[:, 1]
    pred  = (proba >= 0.5).astype(int)
    return {
        'AUROC': round(roc_auc_score(y, proba), 4),
        'AUPRC': round(average_precision_score(y, proba), 4),
        'F1':    round(f1_score(y, pred, zero_division=0), 4),
    }


# ── Step 3: Load & Preprocess ────────────────────────────────────────────────
print("\n[Step 3] Loading dataset and building splits...")

df = pd.read_csv(INPUT_CSV)
FEATURES = [c for c in df.columns if c not in ['SEQN', 'y_steatosis', 'y_fibrosis']]
print(f"  Features ({len(FEATURES)}): {FEATURES}")

splits = {}
for target in TARGETS:
    X = df[FEATURES].values
    y = df[target].values

    print(f"\n  [{target}]  positive: {y.mean()*100:.1f}%")

    # 70 / 15 / 15 stratified split
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=y_tv)

    # Merge train + val → full training set for final model fitting
    X_ft = np.vstack([X_train, X_val])
    y_ft = np.concatenate([y_train, y_val])

    print(f"    Train+Val: {len(X_ft)}  |  Test: {len(X_test)}")

    # Preprocess (fit on train only — no leakage)
    X_ft_p, X_test_p, imp, sc = preprocess_splits(X_ft, X_test)

    # SMOTE on training set only
    X_ft_s, y_ft_s = apply_smote(X_ft_p, y_ft, target)
    print(f"    After SMOTE: {len(X_ft_s)} samples")

    splits[target] = {
        'X_ft_raw': X_ft,       # raw, unprocessed (for CV inside Optuna)
        'y_ft':     y_ft,
        'X_ft_p':   X_ft_p,     # preprocessed, no SMOTE
        'X_ft_s':   X_ft_s,     # preprocessed + SMOTE → for final training
        'y_ft_s':   y_ft_s,
        'X_test':   X_test_p,
        'y_test':   y_test,
        'imp':      imp,
        'sc':       sc,
    }

pickle.dump({'splits': splits, 'features': FEATURES}, open(SPLITS_OUT, 'wb'))
print(f"\n  ✅ Splits saved → {SPLITS_OUT}")


# ── Step 4: Train with Optuna ────────────────────────────────────────────────
print("\n[Step 4] Training models with Optuna hyperparameter tuning...")

all_results = {}

for target in TARGETS:
    s   = splits[target]
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    pos = int(s['y_ft'].sum())
    neg = int((s['y_ft'] == 0).sum())
    spw = neg / pos

    print(f"\n  {'='*58}")
    print(f"  TARGET: {target}  (pos_weight={spw:.2f})")
    print(f"  {'='*58}")
    all_results[target] = {}

    # ── 1. Logistic Regression ────────────────────────────────────
    print("  [1/4] Logistic Regression (Optuna)...", flush=True)

    def lr_objective(trial):
        C = trial.suggest_float('C', 1e-3, 10.0, log=True)
        solver = trial.suggest_categorical('solver', ['lbfgs', 'saga'])
        return cv_auroc(
            lambda: LogisticRegression(C=C, solver=solver,
                                       max_iter=1000, random_state=RANDOM_STATE),
            s['X_ft_raw'], s['y_ft'], target, skf)

    study_lr = optuna.create_study(direction='maximize')
    study_lr.optimize(lr_objective, n_trials=N_TRIALS, show_progress_bar=False)
    best = study_lr.best_params
    print(f"    Best params: {best}  |  CV-AUROC: {study_lr.best_value:.4f}")

    lr_final = LogisticRegression(**best, max_iter=1000, random_state=RANDOM_STATE)
    lr_final.fit(s['X_ft_s'], s['y_ft_s'])
    all_results[target]['LogisticRegression'] = {
        'model': lr_final, 'best_params': best,
        'cv_auroc': round(study_lr.best_value, 4),
        'test_metrics': evaluate(lr_final, s['X_test'], s['y_test']),
    }
    print(f"    Test: {all_results[target]['LogisticRegression']['test_metrics']}")

    # ── 2. Explainable Boosting Machine ───────────────────────────
    print("  [2/4] EBM (Optuna)...", flush=True)

    def ebm_objective(trial):
        params = dict(
            max_bins      = trial.suggest_int('max_bins', 128, 512),
            interactions  = trial.suggest_int('interactions', 5, 20),
            learning_rate = trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            max_leaves    = trial.suggest_int('max_leaves', 2, 5),
        )
        return cv_auroc(
            lambda: ExplainableBoostingClassifier(
                **params, random_state=RANDOM_STATE, n_jobs=-1),
            s['X_ft_raw'], s['y_ft'], target, skf)

    study_ebm = optuna.create_study(direction='maximize')
    study_ebm.optimize(ebm_objective, n_trials=N_TRIALS, show_progress_bar=False)
    best = study_ebm.best_params
    print(f"    Best params: {best}  |  CV-AUROC: {study_ebm.best_value:.4f}")

    ebm_final = ExplainableBoostingClassifier(
        **best, random_state=RANDOM_STATE, n_jobs=-1)
    ebm_final.fit(s['X_ft_s'], s['y_ft_s'])
    all_results[target]['EBM'] = {
        'model': ebm_final, 'best_params': best,
        'cv_auroc': round(study_ebm.best_value, 4),
        'test_metrics': evaluate(ebm_final, s['X_test'], s['y_test']),
    }
    print(f"    Test: {all_results[target]['EBM']['test_metrics']}")

    # ── 3. Random Forest ──────────────────────────────────────────
    print("  [3/4] Random Forest (Optuna)...", flush=True)

    def rf_objective(trial):
        params = dict(
            n_estimators      = trial.suggest_int('n_estimators', 100, 500),
            max_depth         = trial.suggest_int('max_depth', 3, 15),
            min_samples_split = trial.suggest_int('min_samples_split', 2, 20),
            max_features      = trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        )
        return cv_auroc(
            lambda: RandomForestClassifier(
                **params, random_state=RANDOM_STATE, n_jobs=-1),
            s['X_ft_raw'], s['y_ft'], target, skf)

    study_rf = optuna.create_study(direction='maximize')
    study_rf.optimize(rf_objective, n_trials=N_TRIALS, show_progress_bar=False)
    best = study_rf.best_params
    print(f"    Best params: {best}  |  CV-AUROC: {study_rf.best_value:.4f}")

    rf_final = RandomForestClassifier(
        **best, random_state=RANDOM_STATE, n_jobs=-1)
    rf_final.fit(s['X_ft_s'], s['y_ft_s'])
    all_results[target]['RandomForest'] = {
        'model': rf_final, 'best_params': best,
        'cv_auroc': round(study_rf.best_value, 4),
        'test_metrics': evaluate(rf_final, s['X_test'], s['y_test']),
    }
    print(f"    Test: {all_results[target]['RandomForest']['test_metrics']}")

    # ── 4. LightGBM ───────────────────────────────────────────────
    print("  [4/4] LightGBM (Optuna)...", flush=True)

    def lgbm_objective(trial):
        params = dict(
            n_estimators      = trial.suggest_int('n_estimators', 100, 500),
            max_depth         = trial.suggest_int('max_depth', 3, 10),
            learning_rate     = trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            num_leaves        = trial.suggest_int('num_leaves', 20, 100),
            subsample         = trial.suggest_float('subsample', 0.6, 1.0),
            colsample_bytree  = trial.suggest_float('colsample_bytree', 0.6, 1.0),
            reg_alpha         = trial.suggest_float('reg_alpha', 1e-4, 1.0, log=True),
            reg_lambda        = trial.suggest_float('reg_lambda', 1e-4, 1.0, log=True),
        )
        return cv_auroc(
            lambda: lgb.LGBMClassifier(
                **params, scale_pos_weight=spw,
                random_state=RANDOM_STATE, n_jobs=-1, verbose=-1),
            s['X_ft_raw'], s['y_ft'], target, skf)

    study_lgbm = optuna.create_study(direction='maximize')
    study_lgbm.optimize(lgbm_objective, n_trials=N_TRIALS, show_progress_bar=False)
    best = study_lgbm.best_params
    print(f"    Best params: {best}  |  CV-AUROC: {study_lgbm.best_value:.4f}")

    lgbm_final = lgb.LGBMClassifier(
        **best, scale_pos_weight=spw,
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    lgbm_final.fit(s['X_ft_s'], s['y_ft_s'])
    all_results[target]['LightGBM'] = {
        'model': lgbm_final, 'best_params': best,
        'cv_auroc': round(study_lgbm.best_value, 4),
        'test_metrics': evaluate(lgbm_final, s['X_test'], s['y_test']),
    }
    print(f"    Test: {all_results[target]['LightGBM']['test_metrics']}")

# ── Save models ───────────────────────────────────────────────────────────────
pickle.dump(all_results, open(MODELS_OUT, 'wb'))
print(f"\n  ✅ Models saved → {MODELS_OUT}")


# ── Step 5: Discrimination Summary ───────────────────────────────────────────
print("\n\n" + "=" * 68)
print("STEP 5 — DISCRIMINATION RESULTS (Test Set, after Optuna tuning)")
print("=" * 68)

metrics_only = {}
for t in TARGETS:
    print(f"\n{t.upper()}")
    print(f"  {'Model':<22} {'CV-AUROC':>10} {'AUROC':>8} {'AUPRC':>8} {'F1':>8}")
    print(f"  {'-'*58}")
    metrics_only[t] = {}
    for mn in ['LogisticRegression', 'EBM', 'RandomForest', 'LightGBM']:
        d  = all_results[t][mn]
        m  = d['test_metrics']
        cv = d['cv_auroc']
        metrics_only[t][mn] = {**m, 'cv_auroc': cv}
        print(f"  {mn:<22} {cv:>10} {m['AUROC']:>8} {m['AUPRC']:>8} {m['F1']:>8}")

json.dump(metrics_only, open(METRICS_OUT, 'w'), indent=2)
print(f"\n  ✅ Metrics saved → {METRICS_OUT}")
print("\nNext: run step6_calibration.py")

"""Model training, hyperparameter search, and evaluation for PCOS tabular dataset.

Implements:
- ACGCF baseline (simple logistic regression baseline named ACGCF for compatibility)
- XGBoost with Optuna tuning (train CV)
- LightGBM with Optuna tuning (train CV)

Saves models and parameters to results/models/ and plots to results/models/figures/
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    log_loss,
    brier_score_loss,
    matthews_corrcoef,
    cohen_kappa_score,
    balanced_accuracy_score,
    precision_recall_curve,
    roc_curve,
    
)
from sklearn.calibration import calibration_curve

from sklearn.model_selection import StratifiedKFold, cross_val_score

import xgboost as xgb
import lightgbm as lgb
import optuna

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_OUT = REPO_ROOT / "results" / "models"
FIG_OUT = MODELS_OUT / "figures"
MODELS_OUT.mkdir(parents=True, exist_ok=True)
FIG_OUT.mkdir(parents=True, exist_ok=True)

from pw_imp.preprocessing import build_preprocessing_pipeline


def eval_metrics(y_true, y_pred_proba, threshold=0.5):
    y_pred = (y_pred_proba >= threshold).astype(int)
    metrics = {}
    metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
    metrics['precision'] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics['recall'] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics['sensitivity'] = metrics['recall']
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    metrics['f1'] = float(f1_score(y_true, y_pred, zero_division=0))
    metrics['roc_auc'] = float(roc_auc_score(y_true, y_pred_proba))
    metrics['pr_auc'] = float(average_precision_score(y_true, y_pred_proba))
    metrics['mcc'] = float(matthews_corrcoef(y_true, y_pred))
    metrics['cohen_kappa'] = float(cohen_kappa_score(y_true, y_pred))
    metrics['balanced_accuracy'] = float(balanced_accuracy_score(y_true, y_pred))
    metrics['log_loss'] = float(log_loss(y_true, y_pred_proba, labels=[0,1]))
    metrics['brier_score'] = float(brier_score_loss(y_true, y_pred_proba))
    return metrics


def plot_and_save_curves(y_true, y_score, prefix):
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc_score(y_true, y_score):.3f}')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(FIG_OUT / f"{prefix}_roc.png", bbox_inches='tight')
    plt.close()

    # PR curve
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    plt.figure()
    plt.plot(recall, precision, label=f'PR AUC = {average_precision_score(y_true, y_score):.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.savefig(FIG_OUT / f"{prefix}_pr.png", bbox_inches='tight')
    plt.close()

    # Calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=10)
    plt.figure()
    plt.plot(prob_pred, prob_true, marker='o')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration curve')
    plt.savefig(FIG_OUT / f"{prefix}_calibration.png", bbox_inches='tight')
    plt.close()


def plot_confusion(y_true, y_score, prefix, threshold=0.5):
    y_pred = (y_score >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(FIG_OUT / f"{prefix}_confusion.png", bbox_inches='tight')
    plt.close()


def train_acgcf_baseline(X_train, y_train):
    # Implement a simple logistic regression baseline and call it ACGCF (baseline)
    # This is a placeholder baseline to compare against tree models.
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    return clf


def optuna_xgboost(X, y, n_trials=30, random_state=42):
    dtrain = xgb.DMatrix(X, label=y)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)

    def objective(trial):
        param = {
            'verbosity': 0,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'booster': 'gbtree',
            'eta': trial.suggest_loguniform('eta', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'lambda': trial.suggest_loguniform('lambda', 1e-8, 10.0),
            'alpha': trial.suggest_loguniform('alpha', 1e-8, 10.0),
        }
        aucs = []
        for train_idx, val_idx in skf.split(X, y):
            dtr = xgb.DMatrix(X.iloc[train_idx], label=y.iloc[train_idx])
            dval = xgb.DMatrix(X.iloc[val_idx], label=y.iloc[val_idx])
            bst = xgb.train(param, dtr, num_boost_round=200, evals=[(dval, 'val')], early_stopping_rounds=20, verbose_eval=False)
            preds = bst.predict(dval)
            aucs.append(roc_auc_score(y.iloc[val_idx], preds))
        return np.mean(aucs)

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials)
    best = study.best_params
    # finalize param
    best_param = {
        'verbosity': 0,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'eta': best['eta'],
        'max_depth': int(best['max_depth']),
        'subsample': best['subsample'],
        'colsample_bytree': best['colsample_bytree'],
        'lambda': best['lambda'],
        'alpha': best['alpha'],
    }
    return best_param, study


def optuna_lightgbm(X, y, n_trials=30, random_state=42):
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)

    def objective(trial):
        # Use sklearn-compatible LGBMClassifier to avoid API inconsistencies
        param = {
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
            'num_leaves': trial.suggest_int('num_leaves', 15, 127),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 10.0),
            'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 10.0),
        }
        aucs = []
        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]
            clf = lgb.LGBMClassifier(n_estimators=200, random_state=random_state, **param)
            clf.fit(X_tr, y_tr)
            preds = clf.predict_proba(X_va)[:, 1]
            aucs.append(roc_auc_score(y_va, preds))
        return float(np.mean(aucs))

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials)
    best = study.best_params
    best_param = {
        'learning_rate': best['learning_rate'],
        'num_leaves': int(best['num_leaves']),
        'feature_fraction': best['feature_fraction'],
        'bagging_fraction': best['bagging_fraction'],
        'reg_alpha': best['reg_alpha'],
        'reg_lambda': best['reg_lambda'],
    }
    return best_param, study


def run_experiment(save_models=True):
    # build preprocessing and get datasets
    out = build_preprocessing_pipeline()
    X_train = out['X_train']
    y_train = out['y_train']
    X_val = out['X_val']
    y_val = out['y_val']
    X_test = out['X_test']
    y_test = out['y_test']

    results = {}

    # baseline ACGCF (logistic)
    baseline = train_acgcf_baseline(X_train, y_train)
    baseline_proba_test = baseline.predict_proba(X_test)[:,1]
    baseline_metrics = eval_metrics(y_test, baseline_proba_test)
    results['acgcf'] = {'metrics_test': baseline_metrics}
    # save baseline
    if save_models:
        joblib.dump(baseline, MODELS_OUT / 'acgcf_baseline.joblib')

    # XGBoost optuna on train CV
    xgb_params, xgb_study = optuna_xgboost(X_train, y_train, n_trials=30)
    # train final xgboost on train set with best params
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    bst = xgb.train(xgb_params, dtrain, num_boost_round=500, evals=[(dval, 'val')], early_stopping_rounds=30, verbose_eval=False)
    # save booster
    if save_models:
        bst.save_model(str(MODELS_OUT / 'xgboost_model.json'))
    # predictions
    xgb_proba_test = bst.predict(xgb.DMatrix(X_test))
    xgb_metrics = eval_metrics(y_test, xgb_proba_test)
    results['xgboost'] = {'metrics_test': xgb_metrics, 'best_params': xgb_params}

    # LightGBM optuna on train CV
    lgb_params, lgb_study = optuna_lightgbm(X_train, y_train, n_trials=30)
    # train final LightGBM using sklearn API for compatibility
    clf_l = lgb.LGBMClassifier(n_estimators=500, random_state=42, **lgb_params)
    clf_l.fit(X_train, y_train)
    if save_models:
        joblib.dump(clf_l, MODELS_OUT / 'lightgbm_model.joblib')
    lgb_proba_test = clf_l.predict_proba(X_test)[:, 1]
    lgb_metrics = eval_metrics(y_test, lgb_proba_test)
    results['lightgbm'] = {'metrics_test': lgb_metrics, 'best_params': lgb_params}

    # Save model parameters
    params_out = {
        'xgboost': xgb_params,
        'lightgbm': lgb_params,
    }
    with open(MODELS_OUT / 'model_parameters.json', 'w') as f:
        json.dump(params_out, f, indent=2)

    # generate plots for best model (lightgbm and xgboost)
    plot_and_save_curves(y_test, baseline_proba_test, 'acgcf')
    plot_confusion(y_test, baseline_proba_test, 'acgcf')
    plot_and_save_curves(y_test, xgb_proba_test, 'xgboost')
    plot_confusion(y_test, xgb_proba_test, 'xgboost')
    plot_and_save_curves(y_test, lgb_proba_test, 'lightgbm')
    plot_confusion(y_test, lgb_proba_test, 'lightgbm')

    # return results dict and store as json
    with open(MODELS_OUT / 'results_summary.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == '__main__':
    res = run_experiment()
    print('Experiment completed. Summary:')
    print(json.dumps(res, indent=2))

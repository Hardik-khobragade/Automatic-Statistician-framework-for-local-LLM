"""LAYER 3: Statistical Analysis Engine.

A library of well-tested, documented wrapper functions over
scipy / statsmodels / pingouin. The ReAct agent is instructed (via the
system prompt cheat-sheet built in react_agent.py) to call these functions
instead of hand-writing raw statistics code -- this is the single biggest
reliability lever for a small 4B local model, since it only has to get the
*function call* right, not the underlying statistical procedure.

Every public function returns a plain dict of JSON-serializable values
(numpy scalars are cast to native Python types) so results can be printed,
inspected by the LLM, and recorded into the report.

Call `record_result(name, result_dict)` (injected into the sandbox
namespace, see sandbox.py) to attach any result to the final report's
"Statistical Results" tables -- this happens independently of whatever
narrative text the LLM writes, so the report's numbers stay correct even
if the model's prose is weak.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats as spstats


def _native(x: Any) -> Any:
    """Cast numpy/pandas scalars to native python types for clean printing/JSON."""
    if isinstance(x, (np.generic,)):
        return x.item()
    if isinstance(x, (np.ndarray,)):
        return x.tolist()
    if isinstance(x, pd.Series):
        return x.to_dict()
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x


def _round(x, nd=4):
    x = _native(x)
    if x is None:
        return None
    try:
        return round(float(x), nd)
    except (TypeError, ValueError):
        return x


# --------------------------------------------------------------------------- #
# Descriptive statistics
# --------------------------------------------------------------------------- #

def descriptive_stats(df: pd.DataFrame, columns: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Mean/std/median/quartiles/skew/kurtosis/missing for numeric columns."""
    cols = list(columns) if columns else df.select_dtypes(include=[np.number]).columns.tolist()
    out = {}
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(s) == 0:
            out[c] = {"error": "no numeric data"}
            continue
        out[c] = {
            "n": int(len(s)),
            "mean": _round(s.mean()),
            "std": _round(s.std()),
            "min": _round(s.min()),
            "p25": _round(s.quantile(0.25)),
            "median": _round(s.median()),
            "p75": _round(s.quantile(0.75)),
            "max": _round(s.max()),
            "skewness": _round(spstats.skew(s)) if len(s) > 2 else None,
            "kurtosis": _round(spstats.kurtosis(s)) if len(s) > 2 else None,
        }
    return {"test": "descriptive_stats", "columns": out}


def correlation_matrix(df: pd.DataFrame, columns: Optional[Sequence[str]] = None,
                        method: str = "pearson") -> Dict[str, Any]:
    """Correlation matrix (pearson/spearman/kendall) + the most significant pairs."""
    cols = list(columns) if columns else df.select_dtypes(include=[np.number]).columns.tolist()
    sub = df[cols].apply(pd.to_numeric, errors="coerce")
    corr = sub.corr(method=method)
    pairs = []
    for i, c1 in enumerate(cols):
        for c2 in cols[i + 1:]:
            x, y = sub[c1], sub[c2]
            valid = x.notna() & y.notna()
            if valid.sum() < 3:
                continue
            if method == "pearson":
                r, p = spstats.pearsonr(x[valid], y[valid])
            elif method == "spearman":
                r, p = spstats.spearmanr(x[valid], y[valid])
            else:
                r, p = spstats.kendalltau(x[valid], y[valid])
            pairs.append({"col_a": c1, "col_b": c2, "r": _round(r), "p_value": _round(p, 5)})
    pairs.sort(key=lambda d: abs(d["r"]) if d["r"] is not None else 0, reverse=True)
    return {
        "test": "correlation_matrix", "method": method,
        "matrix": {c: {c2: _round(corr.loc[c, c2]) for c2 in cols} for c in cols},
        "pairs_ranked": pairs,
    }


def test_normality(series: pd.Series, method: str = "shapiro") -> Dict[str, Any]:
    """Shapiro-Wilk (n<=5000) or D'Agostino K^2 normality test."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 3:
        return {"test": "normality", "error": "need at least 3 observations"}
    if method == "shapiro" and len(s) <= 5000:
        stat, p = spstats.shapiro(s)
        used = "shapiro"
    else:
        stat, p = spstats.normaltest(s)
        used = "dagostino_k2"
    return {
        "test": "normality", "method": used, "n": int(len(s)),
        "statistic": _round(stat), "p_value": _round(p, 5),
        "is_normal_at_0.05": bool(p > 0.05),
    }


# --------------------------------------------------------------------------- #
# Two-group comparisons
# --------------------------------------------------------------------------- #

def _cohend(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    pooled_sd = math.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if pooled_sd == 0:
        return float("nan")
    return (a.mean() - b.mean()) / pooled_sd


def ttest_ind(df: pd.DataFrame, value: str, group: str, groups: Optional[Sequence] = None,
              equal_var: Optional[bool] = None) -> Dict[str, Any]:
    """Independent-samples t-test between exactly two groups of `group` column.

    Automatically runs Shapiro normality checks and Levene's test for equal
    variances; if `equal_var` is not given, uses Welch's t-test when
    variances differ significantly (safer default), Student's otherwise.
    Always also reports the Mann-Whitney U nonparametric fallback so the
    agent/report can cite whichever is appropriate for the data.
    """
    sub = df[[value, group]].dropna()
    levels = groups if groups else sub[group].unique().tolist()
    if len(levels) != 2:
        return {"test": "ttest_ind", "error": f"expected exactly 2 groups, found {len(levels)}: {levels}"}
    g1, g2 = levels
    a = pd.to_numeric(sub.loc[sub[group] == g1, value], errors="coerce").dropna().to_numpy()
    b = pd.to_numeric(sub.loc[sub[group] == g2, value], errors="coerce").dropna().to_numpy()

    norm_a = spstats.shapiro(a) if 3 <= len(a) <= 5000 else (np.nan, np.nan)
    norm_b = spstats.shapiro(b) if 3 <= len(b) <= 5000 else (np.nan, np.nan)
    lev_stat, lev_p = spstats.levene(a, b)

    if equal_var is None:
        equal_var = bool(lev_p > 0.05)

    t_stat, p_val = spstats.ttest_ind(a, b, equal_var=equal_var)
    mw_stat, mw_p = spstats.mannwhitneyu(a, b, alternative="two-sided")

    return {
        "test": "ttest_ind",
        "value_col": value, "group_col": group, "groups": [str(g1), str(g2)],
        "n": [int(len(a)), int(len(b))],
        "mean": [_round(a.mean()), _round(b.mean())],
        "std": [_round(a.std(ddof=1)), _round(b.std(ddof=1))],
        "variant_used": "Welch" if not equal_var else "Student",
        "t_statistic": _round(t_stat), "p_value": _round(p_val, 5),
        "cohens_d": _round(_cohend(a, b)),
        "assumptions": {
            "levene_equal_variance_p": _round(lev_p, 5),
            "shapiro_normality_p": [_round(norm_a[1], 5), _round(norm_b[1], 5)],
        },
        "nonparametric_fallback": {"test": "mann_whitney_u", "statistic": _round(mw_stat), "p_value": _round(mw_p, 5)},
    }


def ttest_paired(df: pd.DataFrame, col1: str, col2: str) -> Dict[str, Any]:
    """Paired-samples t-test (e.g. before/after on the same subjects)."""
    sub = df[[col1, col2]].dropna()
    a = pd.to_numeric(sub[col1], errors="coerce").to_numpy()
    b = pd.to_numeric(sub[col2], errors="coerce").to_numpy()
    diff = a - b
    t_stat, p_val = spstats.ttest_rel(a, b)
    w_stat, w_p = spstats.wilcoxon(a, b) if len(diff) > 0 and np.any(diff != 0) else (np.nan, np.nan)
    sd = diff.std(ddof=1)
    return {
        "test": "ttest_paired", "col1": col1, "col2": col2, "n": int(len(sub)),
        "mean_diff": _round(diff.mean()), "t_statistic": _round(t_stat), "p_value": _round(p_val, 5),
        "cohens_d_paired": _round(diff.mean() / sd) if sd else None,
        "nonparametric_fallback": {"test": "wilcoxon_signed_rank", "statistic": _round(w_stat), "p_value": _round(w_p, 5)},
    }


def mann_whitney(df: pd.DataFrame, value: str, group: str) -> Dict[str, Any]:
    """Standalone Mann-Whitney U test (nonparametric two-group comparison)."""
    sub = df[[value, group]].dropna()
    levels = sub[group].unique().tolist()
    if len(levels) != 2:
        return {"test": "mann_whitney", "error": f"expected 2 groups, found {len(levels)}"}
    g1, g2 = levels
    a = pd.to_numeric(sub.loc[sub[group] == g1, value], errors="coerce").dropna()
    b = pd.to_numeric(sub.loc[sub[group] == g2, value], errors="coerce").dropna()
    stat, p = spstats.mannwhitneyu(a, b, alternative="two-sided")
    return {"test": "mann_whitney", "groups": [str(g1), str(g2)], "n": [int(len(a)), int(len(b))],
            "statistic": _round(stat), "p_value": _round(p, 5)}


def wilcoxon_signed_rank(df: pd.DataFrame, col1: str, col2: str) -> Dict[str, Any]:
    """Standalone Wilcoxon signed-rank test for paired non-normal data."""
    sub = df[[col1, col2]].dropna()
    a = pd.to_numeric(sub[col1], errors="coerce")
    b = pd.to_numeric(sub[col2], errors="coerce")
    stat, p = spstats.wilcoxon(a, b)
    return {"test": "wilcoxon_signed_rank", "col1": col1, "col2": col2, "n": int(len(sub)),
            "statistic": _round(stat), "p_value": _round(p, 5)}


# --------------------------------------------------------------------------- #
# Multi-group comparisons
# --------------------------------------------------------------------------- #

def anova_oneway(df: pd.DataFrame, value: str, group: str) -> Dict[str, Any]:
    """One-way ANOVA with eta-squared effect size, normality/variance checks,
    a Kruskal-Wallis nonparametric fallback, and Tukey HSD post-hoc pairwise
    comparisons if the omnibus test is significant (p < 0.05)."""
    sub = df[[value, group]].dropna()
    groups_list = [pd.to_numeric(g[value], errors="coerce").dropna().to_numpy()
                   for _, g in sub.groupby(group)]
    labels = list(sub.groupby(group).groups.keys())
    if len(groups_list) < 2:
        return {"test": "anova_oneway", "error": "need at least 2 groups"}

    f_stat, p_val = spstats.f_oneway(*groups_list)
    grand_mean = float(sub[value].astype(float).mean())
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups_list)
    ss_total = sum(((np.concatenate(groups_list) - grand_mean) ** 2))
    eta_sq = ss_between / ss_total if ss_total else None

    shapiro_ps = [_round(spstats.shapiro(g)[1], 5) if 3 <= len(g) <= 5000 else None for g in groups_list]
    lev_stat, lev_p = spstats.levene(*groups_list)
    kw_stat, kw_p = spstats.kruskal(*groups_list)

    posthoc = None
    if p_val < 0.05:
        try:
            from statsmodels.stats.multicomp import pairwise_tukeyhsd
            res = pairwise_tukeyhsd(sub[value].astype(float), sub[group].astype(str))
            tbl = res.summary().data[1:]  # skip header row
            posthoc = [
                {"group1": str(row[0]), "group2": str(row[1]), "meandiff": _round(row[2]),
                 "p_adj": _round(row[3], 5), "reject_null": bool(row[6])}
                for row in tbl
            ]
        except Exception as e:
            posthoc = {"error": f"tukey posthoc failed: {e}"}

    return {
        "test": "anova_oneway", "value_col": value, "group_col": group,
        "groups": [str(l) for l in labels], "n_per_group": [int(len(g)) for g in groups_list],
        "f_statistic": _round(f_stat), "p_value": _round(p_val, 5), "eta_squared": _round(eta_sq),
        "assumptions": {"shapiro_normality_p_per_group": shapiro_ps, "levene_equal_variance_p": _round(lev_p, 5)},
        "nonparametric_fallback": {"test": "kruskal_wallis", "statistic": _round(kw_stat), "p_value": _round(kw_p, 5)},
        "tukey_posthoc": posthoc,
    }


def kruskal_wallis(df: pd.DataFrame, value: str, group: str) -> Dict[str, Any]:
    """Standalone Kruskal-Wallis H-test (nonparametric one-way ANOVA)."""
    sub = df[[value, group]].dropna()
    groups_list = [pd.to_numeric(g[value], errors="coerce").dropna().to_numpy() for _, g in sub.groupby(group)]
    labels = list(sub.groupby(group).groups.keys())
    stat, p = spstats.kruskal(*groups_list)
    return {"test": "kruskal_wallis", "groups": [str(l) for l in labels],
            "n_per_group": [int(len(g)) for g in groups_list], "statistic": _round(stat), "p_value": _round(p, 5)}


# --------------------------------------------------------------------------- #
# Categorical association
# --------------------------------------------------------------------------- #

def chi_square_test(df: pd.DataFrame, col1: str, col2: str) -> Dict[str, Any]:
    """Chi-square test of independence + Cramer's V effect size."""
    sub = df[[col1, col2]].dropna()
    table = pd.crosstab(sub[col1], sub[col2])
    chi2, p, dof, expected = spstats.chi2_contingency(table)
    n = table.values.sum()
    r, k = table.shape
    cramers_v = math.sqrt((chi2 / n) / (min(r - 1, k - 1))) if min(r - 1, k - 1) > 0 else None
    low_expected_pct = float((expected < 5).mean() * 100)
    return {
        "test": "chi_square", "col1": col1, "col2": col2,
        "chi2_statistic": _round(chi2), "p_value": _round(p, 5), "dof": int(dof),
        "cramers_v": _round(cramers_v),
        "contingency_table": {str(i): {str(c): int(v) for c, v in row.items()} for i, row in table.iterrows()},
        "pct_cells_expected_lt_5": _round(low_expected_pct, 1),
        "caveat": "Chi-square assumptions may be violated: >20% of cells have expected count < 5"
                  if low_expected_pct > 20 else None,
    }


# --------------------------------------------------------------------------- #
# Regression
# --------------------------------------------------------------------------- #

def ols_regression(df: pd.DataFrame, formula: str) -> Dict[str, Any]:
    """OLS linear regression via a statsmodels R-style formula,
    e.g. 'income ~ age + C(group)'. Includes residual diagnostics:
    Shapiro normality of residuals, Breusch-Pagan heteroscedasticity test,
    Durbin-Watson autocorrelation, and VIF for numeric predictors."""
    import statsmodels.formula.api as smf
    from statsmodels.stats.diagnostic import het_breuschpagan
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.stats.stattools import durbin_watson

    work = df.copy()
    bool_cols = work.select_dtypes(include="bool").columns
    work[bool_cols] = work[bool_cols].astype(int)

    model = smf.ols(formula=formula, data=work).fit()
    resid = model.resid

    bp_stat, bp_p, _, _ = het_breuschpagan(resid, model.model.exog)
    shapiro_stat, shapiro_p = spstats.shapiro(resid) if 3 <= len(resid) <= 5000 else (np.nan, np.nan)
    dw = durbin_watson(resid)

    vif_data = {}
    try:
        exog = model.model.exog
        names = model.model.exog_names
        for i, name in enumerate(names):
            if name == "Intercept":
                continue
            vif_data[name] = _round(variance_inflation_factor(exog, i))
    except Exception:
        pass

    coefs = {
        name: {
            "coef": _round(model.params[name]),
            "std_err": _round(model.bse[name]),
            "t": _round(model.tvalues[name]),
            "p_value": _round(model.pvalues[name], 5),
            "ci_low": _round(model.conf_int().loc[name, 0]),
            "ci_high": _round(model.conf_int().loc[name, 1]),
        } for name in model.params.index
    }

    return {
        "test": "ols_regression", "formula": formula, "n": int(model.nobs),
        "r_squared": _round(model.rsquared), "adj_r_squared": _round(model.rsquared_adj),
        "f_statistic": _round(model.fvalue), "f_p_value": _round(model.f_pvalue, 5),
        "coefficients": coefs,
        "diagnostics": {
            "residual_normality_shapiro_p": _round(shapiro_p, 5),
            "heteroscedasticity_breusch_pagan_p": _round(bp_p, 5),
            "homoscedastic_at_0.05": bool(bp_p > 0.05) if not np.isnan(bp_p) else None,
            "durbin_watson": _round(dw),
            "vif": vif_data,
        },
    }


def logistic_regression(df: pd.DataFrame, formula: str) -> Dict[str, Any]:
    """Binary logistic regression via a statsmodels R-style formula,
    e.g. 'passed ~ age + income'. Reports odds ratios and pseudo-R^2,
    plus a confusion matrix / accuracy at a 0.5 cutoff. For a target with
    more than 2 classes, use multinomial_logistic_regression() instead."""
    import statsmodels.formula.api as smf

    target_col = formula.split("~")[0].strip()
    if target_col in df.columns and df[target_col].nunique(dropna=True) > 2:
        return {
            "test": "logistic_regression",
            "error": f"'{target_col}' has {df[target_col].nunique()} classes -- logistic_regression "
                     f"only supports binary targets. Use multinomial_logistic_regression(df, formula) "
                     f"or classification_test(df, target='{target_col}') instead.",
        }

    # statsmodels' formula endog parser chokes on bool dtype columns
    # (it tries to one-hot encode them); cast bool -> int first.
    work = df.copy()
    bool_cols = work.select_dtypes(include="bool").columns
    work[bool_cols] = work[bool_cols].astype(int)

    model = smf.logit(formula=formula, data=work).fit(disp=0)
    pred = (model.predict() >= 0.5).astype(int)
    y = model.model.endog.astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    accuracy = (tp + tn) / len(y) if len(y) else None
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None

    coefs = {
        name: {
            "coef": _round(model.params[name]),
            "odds_ratio": _round(math.exp(model.params[name])),
            "p_value": _round(model.pvalues[name], 5),
        } for name in model.params.index
    }
    return {
        "test": "logistic_regression", "formula": formula, "n": int(model.nobs),
        "pseudo_r_squared": _round(model.prsquared), "log_likelihood": _round(model.llf),
        "coefficients": coefs,
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "accuracy": _round(accuracy), "precision": _round(precision), "recall": _round(recall),
    }


def multinomial_logistic_regression(df: pd.DataFrame, formula: str) -> Dict[str, Any]:
    """Multinomial logistic regression for a categorical target with MORE
    THAN 2 classes, via a statsmodels R-style formula, e.g.
    'target ~ alcohol + malic_acid'. Reports coefficients/odds ratios/
    p-values for each non-reference class vs. a reference class, plus
    training-set accuracy and a confusion matrix. This gives you
    statistical inference (which predictors matter, and how much). For a
    held-out predictive accuracy estimate instead, use classification_test()."""
    import statsmodels.formula.api as smf

    target_col = formula.split("~")[0].strip()
    work = df.copy()
    bool_cols = work.select_dtypes(include="bool").columns
    work[bool_cols] = work[bool_cols].astype(int)
    work = work.dropna(subset=[c.strip() for c in [target_col] + formula.split("~")[1].split("+")])

    model = smf.mnlogit(formula=formula, data=work).fit(disp=0, maxiter=200)

    y_raw = work[target_col]
    categories = sorted(pd.unique(y_raw), key=str)
    reference, non_ref = categories[0], categories[1:]

    coefs: Dict[str, Any] = {}
    for col_idx, level in zip(model.params.columns, non_ref):
        for pred_name in model.params.index:
            coef = model.params.loc[pred_name, col_idx]
            p = model.pvalues.loc[pred_name, col_idx]
            coefs[f"{level} vs {reference}: {pred_name}"] = {
                "coef": _round(coef), "odds_ratio": _round(math.exp(coef)), "p_value": _round(p, 5),
            }

    probs = model.predict()
    probs_arr = probs.values if hasattr(probs, "values") else np.asarray(probs)
    pred_idx = probs_arr.argmax(axis=1)
    pred_labels = [categories[i] for i in pred_idx]
    actual_labels = list(y_raw)
    n = len(actual_labels)
    accuracy = sum(p == a for p, a in zip(pred_labels, actual_labels)) / n if n else None

    confusion = {str(a): {str(c): 0 for c in categories} for a in categories}
    for a, p in zip(actual_labels, pred_labels):
        confusion[str(a)][str(p)] += 1

    precisions, recalls, f1s = [], [], []
    for c in categories:
        tp = confusion[str(c)][str(c)]
        fp = sum(confusion[str(a)][str(c)] for a in categories if a != c)
        fn = sum(confusion[str(c)][str(p)] for p in categories if p != c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        precisions.append(prec); recalls.append(rec)
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)

    return {
        "test": "multinomial_logistic_regression", "formula": formula, "n": int(model.nobs),
        "classes": [str(c) for c in categories], "reference_class": str(reference),
        "training_accuracy": _round(accuracy),
        "macro_precision": _round(sum(precisions) / len(precisions)) if precisions else None,
        "macro_recall": _round(sum(recalls) / len(recalls)) if recalls else None,
        "macro_f1": _round(sum(f1s) / len(f1s)) if f1s else None,
        "coefficients": coefs,
        "confusion_matrix_training": confusion,
        "caveat": "Accuracy/confusion matrix here are training-set fit, not held-out predictive "
                  "performance -- use classification_test() for a train/test-split estimate.",
    }


def classification_test(df: pd.DataFrame, target: str, features: Optional[Sequence[str]] = None,
                         model: str = "random_forest", test_size: float = 0.25,
                         random_state: int = 42) -> Dict[str, Any]:
    """Predictive classification test: trains a classifier on a held-out
    train/test split and reports test-set accuracy, macro precision/
    recall/f1, a confusion matrix, and either feature importances
    (tree-based models) or coefficients (logistic). Works for a binary OR
    multi-class target, unlike logistic_regression (binary-only). Gives a
    predictive-accuracy estimate, in contrast to logistic_regression /
    multinomial_logistic_regression which give statistical inference
    (p-values per predictor) instead.

    model: "random_forest" (default -- no scaling needed, handles
    non-linear relationships, gives feature importances) | "logistic" |
    "decision_tree". features: defaults to all numeric columns except target.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier

    if target not in df.columns:
        return {"test": "classification_test", "error": f"target column '{target}' not found"}
    if features is None:
        features = [c for c in df.select_dtypes(include=[np.number]).columns if c != target]
    features = list(features)
    if not features:
        return {"test": "classification_test", "error": "no usable numeric feature columns found"}

    work = df[[target] + features].copy()
    work[features] = work[features].apply(pd.to_numeric, errors="coerce")
    work = work.dropna()
    if work.empty:
        return {"test": "classification_test", "error": "no complete rows after dropping missing values"}

    X, y = work[features], work[target]
    can_stratify = y.nunique() > 1 and y.value_counts().min() >= 2
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y if can_stratify else None)
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state)

    if model == "logistic":
        scaler = StandardScaler()
        X_train_s, X_test_s = scaler.fit_transform(X_train), scaler.transform(X_test)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train_s, y_train)
        y_pred = clf.predict(X_test_s)
        importance_label = "coefficients"
        if clf.coef_.shape[0] == 1:
            importances = {f: _round(c) for f, c in zip(features, clf.coef_[0])}
        else:
            importances = {
                str(cls): {f: _round(c) for f, c in zip(features, row)}
                for cls, row in zip(clf.classes_, clf.coef_)
            }
    elif model == "decision_tree":
        clf = DecisionTreeClassifier(random_state=random_state, max_depth=5)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        importance_label = "feature_importances"
        importances = {f: _round(c) for f, c in zip(features, clf.feature_importances_)}
    else:
        clf = RandomForestClassifier(n_estimators=200, random_state=random_state)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        importance_label = "feature_importances"
        importances = {f: _round(c) for f, c in zip(features, clf.feature_importances_)}

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
    labels_sorted = sorted(y.unique(), key=str)
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
    confusion = {str(a): {str(p): int(cm[i, j]) for j, p in enumerate(labels_sorted)}
                 for i, a in enumerate(labels_sorted)}

    return {
        "test": "classification_test", "model": model, "target": target,
        "features_used": features, "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "n_classes": int(y.nunique()),
        "accuracy": _round(acc), "macro_precision": _round(precision),
        "macro_recall": _round(recall), "macro_f1": _round(f1),
        "confusion_matrix_test": confusion,
        importance_label: importances,
    }


def mixed_effects(df: pd.DataFrame, formula: str, groups: str) -> Dict[str, Any]:
    """Linear mixed-effects model (random intercept by `groups`),
    e.g. formula='score ~ time', groups='subject_id'."""
    import statsmodels.formula.api as smf
    # Drop rows with any missing values up front -- statsmodels' formula
    # machinery drops NA rows internally when building the design matrix,
    # which can desync row counts against `groups` if not pre-aligned.
    work = df.dropna().reset_index(drop=True)
    model = smf.mixedlm(formula, data=work, groups=work[groups]).fit()
    coefs = {
        name: {"coef": _round(model.params[name]), "p_value": _round(model.pvalues[name], 5)}
        for name in model.params.index if name != "Group Var"
    }
    return {
        "test": "mixed_effects", "formula": formula, "groups_col": groups, "n": int(model.nobs),
        "coefficients": coefs, "log_likelihood": _round(model.llf),
        "group_variance": _round(model.cov_re.iloc[0, 0]) if hasattr(model, "cov_re") else None,
    }


# --------------------------------------------------------------------------- #
# Standalone effect sizes
# --------------------------------------------------------------------------- #

def cohens_d(series_a: pd.Series, series_b: pd.Series) -> Dict[str, Any]:
    """Standalone Cohen's d effect size between two numeric series."""
    a = pd.to_numeric(series_a, errors="coerce").dropna().to_numpy()
    b = pd.to_numeric(series_b, errors="coerce").dropna().to_numpy()
    return {"effect_size": "cohens_d", "value": _round(_cohend(a, b))}


def cramers_v_from_table(table: pd.DataFrame) -> Dict[str, Any]:
    """Standalone Cramer's V from an existing contingency table (e.g. pd.crosstab output)."""
    chi2, p, dof, _ = spstats.chi2_contingency(table)
    n = table.values.sum()
    r, k = table.shape
    v = math.sqrt((chi2 / n) / (min(r - 1, k - 1))) if min(r - 1, k - 1) > 0 else None
    return {"effect_size": "cramers_v", "value": _round(v), "chi2_p_value": _round(p, 5)}
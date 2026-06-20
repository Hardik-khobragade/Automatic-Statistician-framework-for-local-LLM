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
    plus a confusion matrix / accuracy at a 0.5 cutoff."""
    import statsmodels.formula.api as smf

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

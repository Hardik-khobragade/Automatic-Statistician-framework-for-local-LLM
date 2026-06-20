"""LAYER 5: Evaluation & Validation.

Three things, scoped realistically for a small local model:

1. Code execution safety -- already handled by sandbox.py's AST whitelist
   and process isolation; this module doesn't re-implement that.
2. Statistical-correctness sanity checks: schema/range validation over
   every recorded result (p in [0,1], dof > 0, n agrees with group sizes,
   etc). This catches corrupted/garbled results, not subtle statistical
   errors -- it is not a substitute for a human review of decision-critical
   conclusions.
3. Ground-truth comparison for benchmark datasets where expected values
   are known (e.g. simulated data, textbook examples, a prior analysis).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


def _is_prob(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and 0.0 <= x <= 1.0


def validate_result(result: Dict[str, Any]) -> List[str]:
    """Return a list of human-readable warnings for one recorded result."""
    warnings: List[str] = []
    name = result.get("name") or result.get("test", "result")

    if "error" in result:
        warnings.append(f"'{name}': the toolkit reported an error and the result may be incomplete: {result['error']}")
        return warnings

    p = result.get("p_value")
    if p is not None and not _is_prob(p):
        warnings.append(f"'{name}': p_value={p} is outside the valid [0, 1] range -- treat this result with suspicion.")

    if "dof" in result and isinstance(result["dof"], (int, float)) and result["dof"] <= 0:
        warnings.append(f"'{name}': degrees of freedom is non-positive ({result['dof']}).")

    n = result.get("n")
    if isinstance(n, list) and any(isinstance(x, (int, float)) and x < 5 for x in n):
        warnings.append(f"'{name}': at least one group has fewer than 5 observations (n={n}) -- "
                         f"results may be unstable regardless of the p-value.")
    elif isinstance(n, (int, float)) and n < 5:
        warnings.append(f"'{name}': sample size is very small (n={n}).")

    assumptions = result.get("assumptions")
    if isinstance(assumptions, dict):
        shapiro_p = assumptions.get("shapiro_normality_p")
        levene_p = assumptions.get("levene_equal_variance_p")
        flat_shapiro = shapiro_p if isinstance(shapiro_p, list) else [shapiro_p]
        if any(isinstance(sp, (int, float)) and sp < 0.05 for sp in flat_shapiro if sp is not None):
            warnings.append(f"'{name}': normality assumption may be violated (Shapiro-Wilk p<0.05) -- "
                             f"consider the nonparametric fallback result alongside the parametric one.")
        if isinstance(levene_p, (int, float)) and levene_p < 0.05 and result.get("variant_used") == "Student":
            warnings.append(f"'{name}': Levene's test suggests unequal variances, but the Student "
                             f"(equal-variance) t-test variant was used.")

    diagnostics = result.get("diagnostics")
    if isinstance(diagnostics, dict):
        if diagnostics.get("homoscedastic_at_0.05") is False:
            warnings.append(f"'{name}': heteroscedasticity detected (Breusch-Pagan p<0.05) -- "
                             f"standard errors may be unreliable; consider robust standard errors.")
        vif = diagnostics.get("vif") or {}
        high_vif = {k: v for k, v in vif.items() if isinstance(v, (int, float)) and v > 10}
        if high_vif:
            warnings.append(f"'{name}': high multicollinearity detected (VIF>10) for: {list(high_vif.keys())}.")

    cv = result.get("pct_cells_expected_lt_5")
    if isinstance(cv, (int, float)) and cv > 20:
        warnings.append(f"'{name}': {cv}% of contingency-table cells have expected count < 5 -- "
                         f"the chi-square approximation may not be reliable.")

    return warnings


def validate_all(recorded_results: List[Dict[str, Any]]) -> List[str]:
    """Run validate_result over every recorded result and flatten warnings."""
    all_warnings: List[str] = []
    for r in recorded_results:
        all_warnings.extend(validate_result(r))
    return all_warnings


# --------------------------------------------------------------------------- #
# Ground-truth comparison
# --------------------------------------------------------------------------- #

def compare_to_ground_truth(recorded_results: List[Dict[str, Any]],
                             ground_truth: Dict[str, Dict[str, Any]],
                             tolerance: float = 0.05) -> Dict[str, Dict[str, Any]]:
    """Compare recorded results against known expected values.

    `ground_truth` maps a result `name` (as passed to record_result) to a
    dict of {field: expected_value}, e.g.:
        {"Income by group ANOVA": {"p_value": 0.30, "eta_squared": 0.008}}

    Returns, for every matched name/field, the observed value, expected
    value, absolute difference, and whether it falls within `tolerance`
    (relative, or absolute for very small expected values).
    """
    by_name = {r.get("name"): r for r in recorded_results}
    comparison: Dict[str, Dict[str, Any]] = {}

    for name, expected_fields in ground_truth.items():
        observed = by_name.get(name)
        if observed is None:
            comparison[name] = {"status": "no matching recorded result found"}
            continue
        field_results = {}
        for field, expected_value in expected_fields.items():
            actual_value = observed.get(field)
            if isinstance(expected_value, (int, float)) and isinstance(actual_value, (int, float)):
                diff = abs(actual_value - expected_value)
                denom = max(abs(expected_value), 1e-9)
                rel_diff = diff / denom
                within_tol = rel_diff <= tolerance or diff <= tolerance
                field_results[field] = {
                    "observed": actual_value, "expected": expected_value,
                    "abs_diff": round(diff, 6), "within_tolerance": within_tol,
                }
            else:
                field_results[field] = {
                    "observed": actual_value, "expected": expected_value,
                    "within_tolerance": actual_value == expected_value,
                }
        comparison[name] = field_results
    return comparison

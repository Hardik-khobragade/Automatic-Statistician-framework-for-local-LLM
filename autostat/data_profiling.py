"""LAYER 1: Data Ingestion & Profiling.

Loads CSV/Excel/JSON, infers a semantic type for every column
(numeric / categorical / ordinal-candidate / datetime / boolean / text-id),
computes descriptive statistics, and emits a compact JSON profile that is
small enough to fit in a 4B model's limited context window.
"""
from __future__ import annotations

import json
import os
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as spstats

warnings.filterwarnings("ignore", message="Could not infer format")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_data(path: str) -> pd.DataFrame:
    """Load a CSV / Excel / JSON file into a DataFrame."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".csv", ".tsv", ".txt"):
        sep = "\t" if ext == ".tsv" else None
        df = pd.read_csv(path, sep=sep, engine="python" if sep is None else "c")
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif ext == ".json":
        df = pd.read_json(path)
    elif ext == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
    # Normalize column names: strip whitespace, keep original otherwise.
    df.columns = [str(c).strip() for c in df.columns]
    return df


# --------------------------------------------------------------------------- #
# Type inference
# --------------------------------------------------------------------------- #

def _try_datetime(series: pd.Series) -> Optional[pd.Series]:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    sample = series.dropna().astype(str).head(50)
    if sample.empty:
        return None
    try:
        parsed = pd.to_datetime(sample, errors="coerce")
    except Exception:
        return None
    success_rate = parsed.notna().mean()
    if success_rate >= 0.9:
        try:
            return pd.to_datetime(series, errors="coerce")
        except Exception:
            return None
    return None


def infer_column_type(series: pd.Series) -> str:
    """Heuristic semantic type: numeric | categorical | ordinal_candidate |
    datetime | boolean | text_or_id."""
    n = len(series)
    nunique = series.nunique(dropna=True)

    if pd.api.types.is_bool_dtype(series):
        return "boolean"

    if _try_datetime(series) is not None and not pd.api.types.is_numeric_dtype(series):
        return "datetime"

    if pd.api.types.is_numeric_dtype(series):
        non_null = series.dropna()
        if non_null.empty:
            return "numeric"
        all_int_like = np.all(np.equal(np.mod(non_null, 1), 0))
        if nunique <= 2:
            return "boolean"
        if all_int_like and nunique <= 12:
            return "ordinal_candidate"  # small integer range -> could be a Likert/ordinal/category
        return "numeric"

    # object / category dtype
    if nunique <= 1:
        return "categorical"
    if n > 0 and (nunique / n) > 0.5 and nunique > 50:
        return "text_or_id"
    return "categorical"


# --------------------------------------------------------------------------- #
# Per-column profiling
# --------------------------------------------------------------------------- #

def _iqr_outlier_count(series: pd.Series) -> int:
    s = series.dropna()
    if len(s) < 4:
        return 0
    q1, q3 = np.percentile(s, [25, 75])
    iqr = q3 - q1
    if iqr == 0:
        return 0
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return int(((s < lower) | (s > upper)).sum())


def _safe_round(x, nd=4):
    try:
        if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
            return None
        return round(float(x), nd)
    except Exception:
        return None


def profile_column(series: pd.Series, dtype_label: str) -> Dict[str, Any]:
    n = len(series)
    n_missing = int(series.isna().sum())
    info: Dict[str, Any] = {
        "detected_type": dtype_label,
        "pandas_dtype": str(series.dtype),
        "n_missing": n_missing,
        "pct_missing": _safe_round(100 * n_missing / n if n else 0, 2),
        "n_unique": int(series.nunique(dropna=True)),
    }

    if dtype_label in ("numeric", "ordinal_candidate"):
        s = pd.to_numeric(series, errors="coerce").dropna()
        if len(s) > 0:
            info.update({
                "mean": _safe_round(s.mean()),
                "std": _safe_round(s.std()),
                "min": _safe_round(s.min()),
                "p25": _safe_round(s.quantile(0.25)),
                "median": _safe_round(s.median()),
                "p75": _safe_round(s.quantile(0.75)),
                "max": _safe_round(s.max()),
                "skewness": _safe_round(spstats.skew(s)) if len(s) > 2 else None,
                "kurtosis": _safe_round(spstats.kurtosis(s)) if len(s) > 2 else None,
                "n_outliers_iqr": _iqr_outlier_count(s),
            })
            if dtype_label == "ordinal_candidate":
                vc = series.value_counts(dropna=True).sort_index()
                info["levels"] = {str(k): int(v) for k, v in vc.head(12).items()}

    elif dtype_label in ("categorical", "boolean", "text_or_id"):
        vc = series.value_counts(dropna=True)
        info["top_values"] = {str(k): int(v) for k, v in vc.head(8).items()}
        if len(vc) > 0:
            info["mode"] = str(vc.index[0])

    elif dtype_label == "datetime":
        dt = _try_datetime(series)
        if dt is not None:
            valid = dt.dropna()
            if len(valid) > 0:
                info["min"] = str(valid.min())
                info["max"] = str(valid.max())
                info["range_days"] = _safe_round((valid.max() - valid.min()).days, 1)

    return info


# --------------------------------------------------------------------------- #
# Full profile
# --------------------------------------------------------------------------- #

def profile_data(df: pd.DataFrame, max_columns_detailed: int = 60) -> Dict[str, Any]:
    """Build the full profile dict (saved as JSON, and also used to build
    the compact prompt summary for the LLM)."""
    n_rows, n_cols = df.shape
    columns: Dict[str, Any] = {}
    for col in df.columns[:max_columns_detailed]:
        dtype_label = infer_column_type(df[col])
        columns[col] = profile_column(df[col], dtype_label)

    numeric_cols = [c for c, info in columns.items() if info["detected_type"] == "numeric"]
    correlations = []
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr(numeric_only=True)
        seen = set()
        pairs = []
        for c1 in numeric_cols:
            for c2 in numeric_cols:
                if c1 == c2 or (c2, c1) in seen:
                    continue
                seen.add((c1, c2))
                val = corr.loc[c1, c2]
                if pd.notna(val):
                    pairs.append((c1, c2, float(val)))
        pairs.sort(key=lambda t: abs(t[2]), reverse=True)
        correlations = [
            {"col_a": a, "col_b": b, "pearson_r": _safe_round(r)} for a, b, r in pairs[:15]
        ]

    profile = {
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "columns_profiled": len(columns),
        "columns_truncated": n_cols > max_columns_detailed,
        "total_missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "columns": columns,
        "top_correlations": correlations,
    }
    return profile


def save_profile(profile: Dict[str, Any], path: str) -> None:
    with open(path, "w") as f:
        json.dump(profile, f, indent=2, default=str)


# --------------------------------------------------------------------------- #
# Compact prompt rendering (token-budget aware, for small local LLMs)
# --------------------------------------------------------------------------- #

def profile_to_prompt_string(profile: Dict[str, Any], max_cols: int = 25) -> str:
    lines: List[str] = []
    lines.append(f"Rows: {profile['n_rows']}  Columns: {profile['n_cols']}  "
                  f"Missing cells: {profile['total_missing_cells']}  "
                  f"Duplicate rows: {profile['duplicate_rows']}")
    lines.append("")
    lines.append("Columns (name | type | key stats):")
    cols = list(profile["columns"].items())
    shown = cols[:max_cols]
    for name, info in shown:
        t = info["detected_type"]
        if t in ("numeric", "ordinal_candidate"):
            lines.append(
                f"- {name} | {t} | mean={info.get('mean')} std={info.get('std')} "
                f"min={info.get('min')} max={info.get('max')} skew={info.get('skewness')} "
                f"missing={info.get('pct_missing')}% outliers={info.get('n_outliers_iqr')}"
            )
        elif t in ("categorical", "boolean", "text_or_id"):
            top = info.get("top_values", {})
            top_str = ", ".join(f"{k}={v}" for k, v in list(top.items())[:4])
            lines.append(
                f"- {name} | {t} | n_unique={info['n_unique']} top=[{top_str}] "
                f"missing={info.get('pct_missing')}%"
            )
        elif t == "datetime":
            lines.append(f"- {name} | datetime | range={info.get('min')} to {info.get('max')}")
        else:
            lines.append(f"- {name} | {t}")
    if len(cols) > max_cols:
        lines.append(f"... ({len(cols) - max_cols} more columns omitted, use query_data to inspect)")

    if profile.get("top_correlations"):
        lines.append("")
        lines.append("Strongest numeric correlations:")
        for c in profile["top_correlations"][:8]:
            lines.append(f"- {c['col_a']} vs {c['col_b']}: r={c['pearson_r']}")

    return "\n".join(lines)

import numpy as np
import pandas as pd


def infer_bulk_log_status(values):
    x = np.asarray(values, dtype=np.float64)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return {
            "is_log": False,
            "reason": "no finite values; defaulting to log1p",
            "max": np.nan,
            "q95": np.nan,
            "q99": np.nan,
            "negative_fraction": np.nan,
            "integer_fraction": np.nan,
        }

    negative_fraction = float(np.mean(finite < 0))
    nonnegative = finite[finite >= 0]
    if nonnegative.size == 0:
        return {
            "is_log": True,
            "reason": "all finite values are negative; treating as already transformed",
            "max": float(np.max(finite)),
            "q95": float(np.percentile(finite, 95)),
            "q99": float(np.percentile(finite, 99)),
            "negative_fraction": negative_fraction,
            "integer_fraction": 0.0,
        }

    max_value = float(np.max(nonnegative))
    q95 = float(np.percentile(nonnegative, 95))
    q99 = float(np.percentile(nonnegative, 99))
    integer_fraction = float(
        np.mean(np.isclose(nonnegative, np.round(nonnegative), atol=1e-6))
    )

    if negative_fraction > 0:
        return {
            "is_log": True,
            "reason": "contains negative values, which is more consistent with pre-transformed data",
            "max": max_value,
            "q95": q95,
            "q99": q99,
            "negative_fraction": negative_fraction,
            "integer_fraction": integer_fraction,
        }

    if q99 > 50.0 or max_value > 100.0:
        return {
            "is_log": False,
            "reason": "large upper expression range suggests raw/count-like data",
            "max": max_value,
            "q95": q95,
            "q99": q99,
            "negative_fraction": negative_fraction,
            "integer_fraction": integer_fraction,
        }

    if q95 > 30.0:
        return {
            "is_log": False,
            "reason": "high 95th percentile suggests raw/count-like data",
            "max": max_value,
            "q95": q95,
            "q99": q99,
            "negative_fraction": negative_fraction,
            "integer_fraction": integer_fraction,
        }

    if integer_fraction > 0.8 and max_value > 20.0:
        return {
            "is_log": False,
            "reason": "integer-heavy matrix with a moderate range suggests counts",
            "max": max_value,
            "q95": q95,
            "q99": q99,
            "negative_fraction": negative_fraction,
            "integer_fraction": integer_fraction,
        }

    return {
        "is_log": True,
        "reason": "compact nonnegative range suggests log-transformed or normalized data",
        "max": max_value,
        "q95": q95,
        "q99": q99,
        "negative_fraction": negative_fraction,
        "integer_fraction": integer_fraction,
    }


def resolve_bulk_transform_info(values):
    """
    Resolve SABER's single transform decision for a bulk/profile matrix.

    The decision is always automatic:
    raw/count-like input -> clip negatives, log1p, sample-wise min-max
    log-like input       -> clip negatives, sample-wise min-max
    """
    infer_info = infer_bulk_log_status(values)
    info = dict(infer_info)
    info["input_scale"] = "log-like" if infer_info["is_log"] else "raw/count-like"
    info["applied_log1p"] = not infer_info["is_log"]
    info["transform"] = "auto"
    info["summary"] = (
        f"input_scale={info['input_scale']}, applied_log1p={info['applied_log1p']}, "
        f"q95={info['q95']:.4g}, q99={info['q99']:.4g}, max={info['max']:.4g}, "
        f"integer_fraction={info['integer_fraction']:.3f}, "
        f"negative_fraction={info['negative_fraction']:.3f}; {info['reason']}"
    )
    return info


def transform_bulk_matrix_to_analysis_scale(values, transform_info=None, return_info=False):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)

    if transform_info is None:
        transform_info = resolve_bulk_transform_info(x)

    z = np.maximum(x, 0.0)
    if transform_info["applied_log1p"]:
        z = np.log1p(z)

    row_min = z.min(axis=1, keepdims=True)
    row_max = z.max(axis=1, keepdims=True)
    denominator = row_max - row_min
    denominator[denominator == 0] = 1.0
    transformed = (z - row_min) / denominator

    if return_info:
        return transformed, dict(transform_info)
    return transformed


def rank_percentile_rows(values):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    n_genes = x.shape[1]

    ranks = pd.DataFrame(x).rank(axis=1, method="average", ascending=True).values
    return (ranks - 1.0) / float(n_genes - 1)


def transform_bulk_dataframe_to_analysis_scale(df, transform_info=None, return_info=False):
    result = transform_bulk_matrix_to_analysis_scale(
        df.values,
        transform_info=transform_info,
        return_info=return_info,
    )
    if return_info:
        transformed, info = result
        return pd.DataFrame(transformed, columns=df.columns, index=df.index), info
    return pd.DataFrame(result, columns=df.columns, index=df.index)

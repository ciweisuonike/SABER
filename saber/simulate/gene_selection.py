import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from saber.data.transform import (
    rank_percentile_rows,
    transform_bulk_matrix_to_analysis_scale,
)


GENE_SELECTION_DEFAULTS = {
    "top_genes": 5000,
    "enable_gene_selection": True,
    "marker_genes_per_celltype": 150,
    "min_pct_in_celltype": 0.05,
    "min_logfc": 0.25,
    "enable_target_stability_filter": True,
    "source_zero_threshold": 0.9,
    "target_zero_threshold": 0.9,
    "target_min_variance_quantile": 0.05,
    "source_target_shift_threshold": 3.0,
    "target_high_expression_quantile": 0.75,
    "stability_soft_threshold": 0.5,
    "stability_gamma": 1.0,
    "stability_pseudobulk_samples": 512,
    "stability_zero_value_threshold": 1e-8,
    "reference_col": "reference",
    "enable_reference_stability_filter": True,
    "reference_min_cells_per_group": 10,
    "reference_celltype_quantile": 0.75,
    "reference_hard_filter_quantile": 0.95,
    "reference_stability_soft_threshold": 1.0,
    "reference_stability_gamma": 1.0,
}


def select_genes_for_saber(adata, bulk, sim_config, return_info=False):
    """
    Select SABER genes before source simulation.

    The fixed path is marker-balanced + HVG supplement + source-target
    stability filtering, with reference-stability filtering when multiple real
    references are available. The returned genes are sorted by gene name so
    that training, finetuning, checkpointing, and prediction use a stable order.
    """
    cfg = GENE_SELECTION_DEFAULTS.copy()
    cfg.update(sim_config or {})

    common_genes = sorted(set(map(str, adata.var_names)) & set(map(str, bulk.columns)))
    if not common_genes:
        raise ValueError("No common genes found between scRNA-seq and target bulk data.")

    top_genes = int(cfg.get("top_genes", 5000))
    if top_genes <= 0:
        raise ValueError("top_genes must be positive.")
    target_n = min(top_genes, len(common_genes))

    if len(common_genes) <= target_n:
        print(
            f"[SABER] Only {len(common_genes)} common genes are available; "
            f"using all of them instead of top_genes={top_genes}."
        )
        info = _empty_selection_info(common_genes)
        return (common_genes, info) if return_info else common_genes

    adata_common = adata[:, common_genes].copy()
    bulk_common = bulk.loc[:, common_genes]
    if not cfg.get("enable_gene_selection", True):
        selected = _select_top_hvg(adata_common, target_n)
        info = _empty_selection_info(selected)
        print(
            "[SABER] Gene selection disabled; selected top "
            f"{len(selected)} HVGs from {len(common_genes)} common genes."
        )
    else:
        selected, info = _select_marker_hvg_stable(
            adata_common,
            bulk_common,
            cfg,
            target_n,
            return_info=True,
        )

    selected = sorted(selected)
    info["selected_genes"] = selected
    info["marker_genes"] = sorted(set(info.get("marker_genes", [])) & set(selected))
    if cfg.get("enable_gene_selection", True):
        info["gene_confidence"] = _gene_confidence_from_selection_weights(
            selected,
            info.get("selection_weights_by_gene"),
        )
    else:
        info["gene_confidence"] = [1.0] * len(selected)
    if cfg.get("enable_gene_selection", True):
        print(
            "[SABER] Marker-HVG-target-stable-reference-stable gene selection selected "
            f"{len(selected)} genes from {len(common_genes)} common genes."
        )
    return (selected, info) if return_info else selected


def _empty_selection_info(genes):
    return {
        "selected_genes": list(genes),
        "marker_genes": [],
        "gene_confidence": [1.0] * len(genes),
        "stability_stats": None,
        "reference_stability_stats": None,
    }


def _select_top_hvg(adata, target_n):
    x_norm = _normalized_log_expression(adata)
    genes = np.asarray(adata.var_names.astype(str))
    hvg_scores = _hvg_scores(x_norm)
    selected_idx = _rank_desc(hvg_scores)[:target_n]
    return list(genes[selected_idx])


def _select_marker_hvg_stable(adata, bulk, cfg, target_n, return_info=False):
    celltype_col = cfg.get("celltype_col", "celltype")
    if celltype_col not in adata.obs.columns:
        raise KeyError(f"celltype_col '{celltype_col}' not found in adata.obs.")

    x_norm = _normalized_log_expression(adata)
    labels = adata.obs[celltype_col].astype(str).to_numpy()
    celltypes = np.unique(labels)
    genes = np.asarray(adata.var_names.astype(str))

    stability_weight, stable_mask, stability_stats = _compute_stability_weights(
        x_norm=x_norm,
        labels=labels,
        celltypes=celltypes,
        bulk=bulk,
        cfg=cfg,
    )
    _print_stability_summary(stability_stats)
    reference_weight, reference_mask, reference_stability_stats = _compute_reference_stability_weights(
        adata=adata,
        x_norm=x_norm,
        labels=labels,
        celltypes=celltypes,
        cfg=cfg,
    )
    _print_reference_stability_summary(reference_stability_stats)
    selection_weight = stability_weight * reference_weight
    selection_mask = stable_mask & reference_mask

    hvg_scores = _hvg_scores(x_norm)
    hvg_weighted = hvg_scores * selection_weight

    marker_k = max(1, int(cfg.get("marker_genes_per_celltype", 150)))
    min_pct = float(cfg.get("min_pct_in_celltype", 0.05))
    min_logfc = float(cfg.get("min_logfc", 0.25))
    eps = 1e-8

    n_celltypes = len(celltypes)
    n_genes = x_norm.shape[1]
    celltype_counts = np.zeros(n_celltypes, dtype=np.int64)
    celltype_sums = np.zeros((n_celltypes, n_genes), dtype=np.float64)
    celltype_detects = np.zeros((n_celltypes, n_genes), dtype=np.float64)

    for ct_idx, ct in enumerate(celltypes):
        x_in = x_norm[labels == ct]
        celltype_counts[ct_idx] = x_in.shape[0]
        celltype_sums[ct_idx] = x_in.sum(axis=0)
        celltype_detects[ct_idx] = (x_in > 0).sum(axis=0)

    celltype_means = celltype_sums / celltype_counts[:, None]
    celltype_pcts = celltype_detects / celltype_counts[:, None]
    total_sum = celltype_sums.sum(axis=0)
    total_detect = celltype_detects.sum(axis=0)
    total_cells = int(celltype_counts.sum())

    selected_idx = set()
    per_type_rankings = []
    per_type_counts = {}
    aggregate_marker = np.zeros(x_norm.shape[1], dtype=np.float64)

    for ct_idx, ct in enumerate(celltypes):
        n_in = int(celltype_counts[ct_idx])
        n_rest = total_cells - n_in
        if n_rest <= 0:
            continue

        mean_in = celltype_means[ct_idx]
        pct_in = celltype_pcts[ct_idx]
        mean_rest = (total_sum - celltype_sums[ct_idx]) / float(n_rest)
        pct_rest = (total_detect - celltype_detects[ct_idx]) / float(n_rest)
        other_celltypes = np.arange(n_celltypes) != ct_idx
        max_other_mean = celltype_means[other_celltypes].max(axis=0)
        max_other_pct = celltype_pcts[other_celltypes].max(axis=0)

        logfc = np.log((mean_in + eps) / (mean_rest + eps))
        competitor_logfc = np.log((mean_in + eps) / (max_other_mean + eps))
        det_diff = pct_in - pct_rest
        marker_score = (
            np.maximum(logfc, 0.0)
            * np.maximum(det_diff, 0.0)
            * np.sqrt(pct_in + eps)
        )
        quality_mask = (
            (pct_in >= min_pct)
            & (logfc >= min_logfc)
            & (competitor_logfc >= min_logfc)
            & (pct_in > max_other_pct)
        )
        marker_score = np.where(quality_mask, marker_score * selection_weight, 0.0)

        aggregate_marker = np.maximum(aggregate_marker, marker_score)
        ranking = _rank_desc(marker_score)
        ranking = ranking[marker_score[ranking] > 0]
        per_type_rankings.append(ranking)
        selected_for_type = ranking[:marker_k]
        selected_idx.update(selected_for_type.tolist())
        per_type_counts[str(ct)] = {
            "selected": int(len(selected_for_type)),
            "eligible": int(len(ranking)),
        }

    print("[SABER] Marker selection per cell type:")
    for ct in sorted(per_type_counts):
        stats = per_type_counts[ct]
        print(
            f"  - {ct}: selected {stats['selected']} marker genes "
            f"from {stats['eligible']} eligible candidates"
        )

    marker_union_count = len(selected_idx)
    print(f"[SABER] Marker union genes before trimming: {marker_union_count}")

    if len(selected_idx) > target_n:
        selected_idx = _trim_marker_union(
            per_type_rankings=per_type_rankings,
            aggregate_marker=aggregate_marker,
            marker_k=marker_k,
            target_n=target_n,
        )
        print(f"[SABER] Marker union trimmed to top_genes limit: {len(selected_idx)}")
    marker_idx = set(selected_idx)

    before_hvg = len(selected_idx)
    _add_ranked_by_score(selected_idx, hvg_weighted, target_n, min_score=0.0)
    after_hvg = len(selected_idx)
    print(f"[SABER] HVG supplement genes added: {after_hvg - before_hvg}")

    if len(selected_idx) < target_n:
        before_stable_fallback = len(selected_idx)
        stable_scores = np.where(selection_mask, selection_weight, 0.0)
        _add_ranked_by_score(selected_idx, stable_scores, target_n, min_score=0.0)
        after_stable_fallback = len(selected_idx)
        print(
            f"[SABER] Stable fallback genes added: "
            f"{after_stable_fallback - before_stable_fallback}"
        )

    if len(selected_idx) < target_n:
        before_hvg_fallback = len(selected_idx)
        _add_ranked_by_score(selected_idx, hvg_scores, target_n, min_score=None)
        after_hvg_fallback = len(selected_idx)
        print(
            f"[SABER] Raw HVG fallback genes added: "
            f"{after_hvg_fallback - before_hvg_fallback}"
        )

    if len(selected_idx) < target_n:
        before_final_fallback = len(selected_idx)
        for idx in range(len(genes)):
            selected_idx.add(idx)
            if len(selected_idx) >= target_n:
                break
        print(f"[SABER] Final fallback genes added: {len(selected_idx) - before_final_fallback}")

    print(f"[SABER] Final selected genes: {len(selected_idx)}")

    selected_genes = list(genes[list(selected_idx)])
    info = {
        "selected_genes": selected_genes,
        "marker_genes": list(genes[list(marker_idx & selected_idx)]),
        "selection_weights_by_gene": {
            str(genes[idx]): float(selection_weight[idx])
            for idx in selected_idx
        },
        "stability_stats": stability_stats,
        "reference_stability_stats": reference_stability_stats,
    }
    return (selected_genes, info) if return_info else selected_genes


def _gene_confidence_from_selection_weights(selected_genes, weights_by_gene):
    n_genes = len(selected_genes)
    if n_genes == 0:
        return []
    if n_genes == 1 or not weights_by_gene:
        return [1.0] * n_genes

    weights = np.asarray(
        [weights_by_gene.get(str(gene), np.nan) for gene in selected_genes],
        dtype=np.float64,
    )
    if (
        weights.shape[0] != n_genes
        or not np.all(np.isfinite(weights))
        or np.allclose(weights, weights[0])
    ):
        return [1.0] * n_genes

    ranks = _average_rank_ascending(weights)
    confidence = ranks / (float(n_genes - 1) + 1e-8)
    confidence = np.clip(confidence, 0.0, 1.0)
    return confidence.astype(float).tolist()


def _average_rank_ascending(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    sorted_values = values[order]

    start = 0
    while start < sorted_values.shape[0]:
        end = start + 1
        while end < sorted_values.shape[0] and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = (start + end - 1) / 2.0
        ranks[order[start:end]] = average_rank
        start = end

    return ranks


def _trim_marker_union(per_type_rankings, aggregate_marker, marker_k, target_n):
    selected = set()
    n_types = max(len(per_type_rankings), 1)
    if target_n >= n_types:
        per_type_floor = max(1, min(marker_k, target_n // (2 * n_types)))
    else:
        per_type_floor = 0

    for ranking in per_type_rankings:
        for idx in ranking[:per_type_floor]:
            selected.add(int(idx))
            if len(selected) >= target_n:
                return selected

    _add_ranked_by_score(selected, aggregate_marker, target_n, min_score=0.0)
    return selected


def _compute_stability_weights(x_norm, labels, celltypes, bulk, cfg):
    n_genes = x_norm.shape[1]
    if not cfg.get("enable_target_stability_filter", True):
        stats = {
            "enabled": False,
            "total_genes": int(n_genes),
            "hard_filtered": 0,
            "target_zero": 0,
            "low_target_variance": 0,
            "source_zero_target_high": 0,
            "source_target_shift": 0,
            "stable": int(n_genes),
        }
        return np.ones(n_genes, dtype=np.float64), np.ones(n_genes, dtype=bool), stats

    source_pre_raw = _make_source_pseudobulk(
        x_norm=x_norm,
        labels=labels,
        celltypes=celltypes,
        n_samples=int(cfg.get("stability_pseudobulk_samples", 512)),
        random_state=int(cfg.get("random_state", 42)),
    )
    source_zero_scale = transform_bulk_matrix_to_analysis_scale(source_pre_raw)
    target_values = bulk.values if isinstance(bulk, pd.DataFrame) else np.asarray(bulk)
    target_zero_scale, target_transform_info = transform_bulk_matrix_to_analysis_scale(
        target_values,
        transform_info=cfg.get("real_bulk_transform_info"),
        return_info=True,
    )

    zero_threshold = float(cfg.get("stability_zero_value_threshold", 1e-8))
    source_zero = (source_zero_scale <= zero_threshold).mean(axis=0)
    target_zero = (target_zero_scale <= zero_threshold).mean(axis=0)

    source_pre = rank_percentile_rows(source_zero_scale)
    target_pre = rank_percentile_rows(target_zero_scale)

    target_var = target_pre.var(axis=0) if target_pre.shape[0] > 1 else np.ones(n_genes)
    if target_pre.shape[0] > 1:
        var_threshold = np.quantile(
            target_var,
            float(cfg.get("target_min_variance_quantile", 0.05)),
        )
        low_target_var = target_var <= var_threshold
    else:
        low_target_var = np.zeros(n_genes, dtype=bool)

    source_median = np.median(source_pre, axis=0)
    target_median = np.median(target_pre, axis=0)
    source_iqr = np.percentile(source_pre, 75, axis=0) - np.percentile(source_pre, 25, axis=0)
    target_iqr = np.percentile(target_pre, 75, axis=0) - np.percentile(target_pre, 25, axis=0)
    shift = np.abs(source_median - target_median) / (source_iqr + target_iqr + 1e-8)
    shift = np.nan_to_num(shift, nan=np.inf, posinf=np.inf, neginf=np.inf)

    target_high = np.quantile(
        target_median,
        float(cfg.get("target_high_expression_quantile", 0.75)),
    )
    source_zero_target_high = (
        source_zero >= float(cfg.get("source_zero_threshold", 0.9))
    ) & (target_median >= target_high)

    target_zero_filter = target_zero >= float(cfg.get("target_zero_threshold", 0.9))
    shift_filter = shift >= float(cfg.get("source_target_shift_threshold", 3.0))
    hard_filter = target_zero_filter | low_target_var | source_zero_target_high | shift_filter

    gamma = float(cfg.get("stability_gamma", 1.0))
    soft_threshold = float(cfg.get("stability_soft_threshold", 0.5))
    soft_shift = np.clip(shift - soft_threshold, 0.0, 50.0)
    weights = np.exp(-gamma * soft_shift)
    weights[hard_filter] = 0.0
    stable_mask = ~hard_filter
    stats = {
        "enabled": True,
        "total_genes": int(n_genes),
        "hard_filtered": int(hard_filter.sum()),
        "target_zero": int(target_zero_filter.sum()),
        "low_target_variance": int(low_target_var.sum()),
        "source_zero_target_high": int(source_zero_target_high.sum()),
        "source_target_shift": int(shift_filter.sum()),
        "stable": int(stable_mask.sum()),
        "target_transform_info": target_transform_info,
    }
    return weights.astype(np.float64), stable_mask, stats


def _compute_reference_stability_weights(adata, x_norm, labels, celltypes, cfg):
    n_genes = x_norm.shape[1]
    ones = np.ones(n_genes, dtype=np.float64)
    all_stable = np.ones(n_genes, dtype=bool)
    reference_col = cfg.get("reference_col", "reference")
    min_cells = None
    celltype_quantile = None
    hard_filter_quantile = None
    gamma = None

    def skipped_stats(reason, valid_references=0):
        return {
            "enabled": False,
            "reason": reason,
            "reference_col": reference_col,
            "valid_references": int(valid_references),
            "comparable_celltypes": 0,
            "hard_filtered": 0,
            "stable": int(n_genes),
            "total_genes": int(n_genes),
            "min_cells_per_group": min_cells,
            "celltype_quantile": celltype_quantile,
            "hard_filter_quantile": hard_filter_quantile,
            "gamma": gamma,
        }

    if not cfg.get("enable_reference_stability_filter", True):
        return ones, all_stable, skipped_stats("disabled via config")
    if reference_col is None:
        return ones, all_stable, skipped_stats("reference_col is not configured")
    if reference_col not in adata.obs.columns:
        return ones, all_stable, skipped_stats(
            f"reference_col '{reference_col}' not found in adata.obs"
        )

    raw_references = adata.obs[reference_col].to_numpy()
    valid_reference_mask = np.asarray(
        [
            not pd.isna(reference) and str(reference).strip() != ""
            for reference in raw_references
        ],
        dtype=bool,
    )
    references = np.asarray(
        [
            str(reference).strip() if is_valid else ""
            for reference, is_valid in zip(raw_references, valid_reference_mask)
        ],
        dtype=object,
    )
    unique_references = np.unique(references[valid_reference_mask])
    if len(unique_references) < 2:
        return ones, all_stable, skipped_stats(
            "fewer than two valid references",
            valid_references=len(unique_references),
        )

    min_cells = max(2, int(cfg.get("reference_min_cells_per_group", 10)))
    celltype_quantile = _validate_quantile(
        cfg.get("reference_celltype_quantile", 0.75),
        "reference_celltype_quantile",
    )
    hard_filter_quantile = _validate_quantile(
        cfg.get("reference_hard_filter_quantile", 0.95),
        "reference_hard_filter_quantile",
    )
    gamma = float(cfg.get("reference_stability_gamma", 1.0))
    if not np.isfinite(gamma) or gamma < 0:
        raise ValueError("reference_stability_gamma must be finite and nonnegative.")

    per_celltype_dispersions = []
    comparable_celltypes = []
    for celltype in celltypes:
        reference_means = []
        reference_variances = []
        for reference in unique_references:
            group_mask = (
                valid_reference_mask
                & (references == reference)
                & (labels == celltype)
            )
            if int(group_mask.sum()) < min_cells:
                continue
            x_group = x_norm[group_mask]
            reference_means.append(x_group.mean(axis=0))
            reference_variances.append(x_group.var(axis=0, ddof=1))

        if len(reference_means) < 2:
            continue

        reference_means = np.vstack(reference_means)
        reference_variances = np.vstack(reference_variances)
        mean_across_references = reference_means.mean(axis=0)
        between_reference_std = np.sqrt(
            np.mean((reference_means - mean_across_references) ** 2, axis=0)
        )
        within_reference_std = np.sqrt(np.mean(reference_variances, axis=0))
        dispersion = between_reference_std / (within_reference_std + 1e-8)
        dispersion = np.nan_to_num(
            dispersion,
            nan=1e12,
            posinf=1e12,
            neginf=1e12,
        )
        per_celltype_dispersions.append(dispersion)
        comparable_celltypes.append(str(celltype))

    if not per_celltype_dispersions:
        return ones, all_stable, skipped_stats(
            "no cell type has at least two sufficiently populated references",
            valid_references=len(unique_references),
        )

    reference_dispersion = np.quantile(
        np.vstack(per_celltype_dispersions),
        celltype_quantile,
        axis=0,
    )
    hard_threshold = float(np.quantile(reference_dispersion, hard_filter_quantile))
    reference_mask = reference_dispersion <= hard_threshold
    soft_threshold = float(cfg.get("reference_stability_soft_threshold", 1.0))
    soft_dispersion = np.clip(reference_dispersion - soft_threshold, 0.0, 50.0)
    weights = np.exp(-gamma * soft_dispersion)
    weights[~reference_mask] = 0.0
    stats = {
        "enabled": True,
        "reason": None,
        "reference_col": reference_col,
        "valid_references": int(len(unique_references)),
        "comparable_celltypes": int(len(comparable_celltypes)),
        "comparable_celltype_names": comparable_celltypes,
        "hard_filtered": int((~reference_mask).sum()),
        "stable": int(reference_mask.sum()),
        "total_genes": int(n_genes),
        "min_cells_per_group": int(min_cells),
        "celltype_quantile": float(celltype_quantile),
        "hard_filter_quantile": float(hard_filter_quantile),
        "hard_threshold": hard_threshold,
        "gamma": float(gamma),
    }
    return weights.astype(np.float64), reference_mask, stats


def _validate_quantile(value, name):
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1.")
    return value


def _print_reference_stability_summary(stats):
    if not stats.get("enabled", False):
        print(
            "[SABER] Reference stability filter skipped: "
            f"{stats['reason']}; using neutral reference weights."
        )
        return

    print("[SABER] Reference stability filter summary:")
    print(f"  - reference column: {stats['reference_col']}")
    print(f"  - valid references: {stats['valid_references']}")
    print(f"  - comparable cell types: {stats['comparable_celltypes']}")
    print(f"  - hard-filtered genes: {stats['hard_filtered']}")
    print(f"  - reference-stable genes: {stats['stable']}")


def _print_stability_summary(stats):
    if not stats.get("enabled", False):
        print(
            "[SABER] Target stability filter disabled; "
            f"all {stats['total_genes']} candidate genes remain stability-eligible."
        )
        return

    print("[SABER] Target stability filter summary:")
    target_transform_info = stats.get("target_transform_info")
    if target_transform_info:
        print(f"  - target transform decision: {target_transform_info['summary']}")
    print(f"  - total candidate genes: {stats['total_genes']}")
    print(f"  - hard-filtered genes: {stats['hard_filtered']}")
    print(f"  - stable genes: {stats['stable']}")
    print(f"  - target near-zero genes: {stats['target_zero']}")
    print(f"  - target low-variance genes: {stats['low_target_variance']}")
    print(f"  - source-zero target-high genes: {stats['source_zero_target_high']}")
    print(f"  - source-target shifted genes: {stats['source_target_shift']}")


def _make_source_pseudobulk(x_norm, labels, celltypes, n_samples, random_state):
    n_samples = max(1, int(n_samples))
    signatures = []
    for ct in celltypes:
        mask = labels == ct
        if mask.any():
            signatures.append(x_norm[mask].mean(axis=0))
    if not signatures:
        return np.maximum(x_norm[: min(n_samples, x_norm.shape[0])], 0.0)

    signatures = np.vstack(signatures)
    rng = np.random.default_rng(random_state)
    proportions = rng.dirichlet(np.ones(signatures.shape[0]), size=n_samples)
    pseudo = proportions @ signatures
    return np.maximum(pseudo, 0.0)


def _normalized_log_expression(adata):
    adata_norm = adata.copy()
    adata_norm.X = _nonnegative_matrix(adata_norm.X)
    sc.pp.normalize_total(adata_norm, target_sum=1e4)
    sc.pp.log1p(adata_norm)
    return _to_dense(adata_norm.X).astype(np.float64, copy=False)


def _hvg_scores(x_norm):
    scores = np.var(x_norm, axis=0)
    return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)


def _rank_desc(scores):
    scores = np.asarray(scores, dtype=np.float64)
    return np.argsort(-scores, kind="mergesort")


def _add_ranked_by_score(selected_idx, scores, target_n, min_score):
    if len(selected_idx) >= target_n:
        return
    ranked = _rank_desc(scores)
    for idx in ranked:
        score = scores[idx]
        if min_score is not None and score <= min_score:
            continue
        selected_idx.add(int(idx))
        if len(selected_idx) >= target_n:
            return


def _to_dense(x):
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)


def _nonnegative_matrix(x):
    if sparse.issparse(x):
        x = x.copy()
        x.data = np.maximum(x.data, 0.0)
        return x
    return np.maximum(np.asarray(x), 0.0)

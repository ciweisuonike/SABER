from saber.simulate.bulk import simulate_bulk_with_batch_effect


PROP_TRAIN_ONLY_SIMULATION_KEYS = (
    "layer",
    "top_genes",
    "n_target_sample_to_train",
    "real_bulk_transform_info",
    "enable_gene_selection",
    "marker_genes_per_celltype",
    "min_pct_in_celltype",
    "min_logfc",
    "enable_target_stability_filter",
    "source_zero_threshold",
    "target_zero_threshold",
    "target_min_variance_quantile",
    "source_target_shift_threshold",
    "target_high_expression_quantile",
    "stability_soft_threshold",
    "stability_gamma",
    "stability_pseudobulk_samples",
    "enable_reference_stability_filter",
    "reference_min_cells_per_group",
    "reference_celltype_quantile",
    "reference_hard_filter_quantile",
    "reference_stability_soft_threshold",
    "reference_stability_gamma",
)


def run_simulation(
    adata,
    sim_config: dict,
):
    """
    PROP training only:
    - generate simulated bulk data with ground-truth proportions
    - expose paired clean/observed source views for consistency training
    """

    sim_config = sim_config.copy()
    enable_gene_selection = bool(sim_config.get("enable_gene_selection", True))
    for prop_train_key in PROP_TRAIN_ONLY_SIMULATION_KEYS:
        sim_config.pop(prop_train_key, None)
    enable_twoview = bool(sim_config.get("enable_twoview", True))
    enable_confidence_gene_dropout = bool(
        sim_config.get("enable_confidence_gene_dropout", True)
    )
    sim_config["enable_confidence_gene_dropout"] = (
        enable_gene_selection and enable_twoview and enable_confidence_gene_dropout
    )

    results = simulate_bulk_with_batch_effect(
        adata=adata,
        **sim_config,
    )

    (
        X_obs_df,
        X_clean_scaled_df,
        X_proc_df,
        X_clean_counts_df,
        P_df,
        B_df,
        celltypes,
    ) = results

    return {
        "X_obs": X_obs_df,
        "X_clean": X_clean_scaled_df,
        "X_proc": X_proc_df,
        "X_clean_counts": X_clean_counts_df,
        "P": P_df,
        "B": B_df,
        "celltypes": celltypes,
        "enable_twoview": enable_twoview,
    }

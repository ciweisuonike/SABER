def preprocess_simulated_data(sim_data: dict):
    """
    PROP training preprocessing:
    - gene alignment
    - paired source view sanity checks
    - proportion-only preprocessing
    """

    X_obs = sim_data["X_obs"]
    X_clean = sim_data.get("X_clean", X_obs)
    X_clean_counts = sim_data.get("X_clean_counts", X_clean)
    enable_twoview = bool(sim_data.get("enable_twoview", True))
    P = sim_data["P"]
    B = sim_data["B"]

    if X_obs.shape[0] != P.shape[0]:
        raise ValueError(
            "Sample number mismatch between observed expression matrix X_obs "
            "and proportion matrix P"
        )
    if X_clean.shape[0] != P.shape[0]:
        raise ValueError(
            "Sample number mismatch between clean expression matrix X_clean "
            "and proportion matrix P"
        )
    if list(X_clean.columns) != list(X_obs.columns):
        raise ValueError("Gene order mismatch between clean and observed source views")
    if list(X_clean_counts.columns) != list(X_obs.columns):
        raise ValueError(
            "Gene order mismatch between clean counts and observed source views"
        )

    gene_names = X_obs.columns.tolist()

    return {
        "X_obs": X_obs,
        "X_clean": X_clean,
        "X_clean_counts": X_clean_counts,
        "P": P,
        "B": B,
        "gene_names": gene_names,
        "celltypes": sim_data["celltypes"],
        "enable_twoview": enable_twoview,
    }

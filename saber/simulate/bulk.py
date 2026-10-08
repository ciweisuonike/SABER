import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import sparse
import random
from saber.data.transform import transform_bulk_matrix_to_analysis_scale


def _validate_count_matrix(X, name="adata.X"):
    dtype = getattr(X, "dtype", None)
    if (
        dtype is None
        or not np.issubdtype(dtype, np.number)
        or np.issubdtype(dtype, np.complexfloating)
    ):
        raise ValueError(f"{name} must be a numeric count matrix.")

    values = X.data if sparse.issparse(X) else np.asarray(X)
    if values.size == 0:
        return
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} count matrix must contain only finite values.")
    if np.any(values < 0):
        raise ValueError(f"{name} count matrix must be nonnegative.")
    if not np.issubdtype(values.dtype, np.integer) and not np.all(
        values == np.floor(values)
    ):
        raise ValueError(f"{name} count matrix must contain integer-valued counts.")


def _build_balanced_reference_schedule(reference_pools, n_banks, rng):
    """Assign reference pools to banks with balanced shuffled rounds."""
    reference_pools = list(reference_pools)
    if n_banks < 1:
        raise ValueError("n_banks must be at least 1.")
    if not reference_pools:
        return []

    if len(reference_pools) >= n_banks:
        return rng.sample(reference_pools, n_banks)

    schedule = []
    while len(schedule) < n_banks:
        shuffled_round = reference_pools.copy()
        rng.shuffle(shuffled_round)
        remaining = n_banks - len(schedule)
        schedule.extend(shuffled_round[:remaining])
    return schedule


def _build_bank_ids(K, n_pseudo_banks):
    """Split K samples across non-empty banks whose sizes differ by at most one."""
    n_banks = min(int(K), max(1, int(n_pseudo_banks)))
    bank_sizes = np.full(n_banks, int(K) // n_banks, dtype=int)
    bank_sizes[: int(K) % n_banks] += 1
    return np.repeat(np.arange(n_banks, dtype=int), bank_sizes)


# =========================================================================
# Reference-style augmentation with ordered operations and marker support
# =========================================================================
def _apply_reference_style(
    subE,
    marker_indices=None,
    scale_sigma=0.1,
    drop_alpha=0.5,
    drop_beta=0.5,
    marker_prob=0.1,
    marker_factor=0.7,
    rng=None,
):
    """
    Apply a reference-style shift to a single-cell expression matrix.
    """
    rng = np.random if rng is None else rng
    subE_aug = subE.copy().astype(np.float64)
    N, G = subE_aug.shape

    # 1. Expression-dependent logistic dropout
    logits = drop_alpha - drop_beta * subE_aug
    drop_p = 0.5 / (1.0 + np.exp(-logits))
    drop_mask = rng.binomial(1, drop_p, size=subE_aug.shape).astype(bool)
    subE_aug[drop_mask] = 0.0

    # 2. Gene- and cell-level log-normal scaling
    gene_scale = rng.lognormal(mean=0.0, sigma=scale_sigma, size=(1, G))
    cell_scale = rng.lognormal(mean=0.0, sigma=scale_sigma, size=(N, 1))
    subE_aug *= gene_scale * cell_scale

    # 3. Random marker-gene attenuation
    if marker_indices is not None and len(marker_indices) > 0:
        num_markers = len(marker_indices)
        attenuation_mask = rng.random((N, num_markers)) < marker_prob
        gamma = rng.uniform(marker_factor, 1.0, size=(N, num_markers))
        subE_aug[:, marker_indices] = np.where(
            attenuation_mask,
            subE_aug[:, marker_indices] * gamma,
            subE_aug[:, marker_indices]
        )
    else:
        attenuation_mask = rng.random((N, G)) < marker_prob
        gamma = rng.uniform(marker_factor, 1.0, size=(N, G))
        subE_aug[attenuation_mask] *= gamma[attenuation_mask]

    return subE_aug


# =========================================================================
# Main simulation function with cycle-level reference generation
# =========================================================================
def simulate_bulk_with_batch_effect(
    adata,
    K=10000,
    N=500,
    n_batches=1000,
    celltype_col="celltype",
    random_state=42,
    n_conditions=1000,
    condition_col=None,

    # Parameters controlling within-condition similarity and composition
    concentration_factor=1.0,
    missing_celltype_fraction=0.2,
    min_celltypes_per_sample=2,
    dominant_sample_fraction=0.2,
    min_dominant_fraction=0.4,
    max_dominant_fraction=0.95,
    enable_missing_celltypes=True,
    enable_dominant_celltypes=True,

    # Reference-related parameters
    reference_col="reference",

    # Split K samples into this many cycles and generate one reference per cycle.
    n_pseudo_banks=10,

    # Noise and batch-effect parameters
    n_batch_factors=3,
    affected_gene_range=(500, 3000),
    log_uk_range=(-10, 10),
    sigma_k_range=(0.05, 0.2),
    individual_gene_variance=0.1,
    gaussian_noise_level=0.1,
    batch_dropout_range=(0.01,0.05),
    sample_variance_dropout=0.005,
    dropout_expression_threshold=[3.0,5.0,10.0,20.0],
    dropout_enhancement_factor=[25.0, 12.5,6.25,3],

    # === Paired source view technical perturbation switch ===
    enable_twoview=True,

    # Probability of reference augmentation at the cycle level
    prob_ref_aug=0.5,

    prob_batch_inj=0.5,
    enable_confidence_gene_dropout=True,
    confidence_gene_dropout_max=0.8,
    confidence_gene_dropout_kappa=1.5,
    confidence_gene_dropout_retention=0.25,
    gene_confidence=None,
):
    """
    Simulate bulk RNA-seq data with batch effects.

    Samples are divided among non-empty pseudo-banks. Each bank shares one
    reference pool and one generated reference view.
    """
    np.random.seed(random_state)
    reference_rng = random.Random(random_state)
    refaug_seed = None if random_state is None else int(random_state) + 104729
    refaug_rng = np.random.default_rng(refaug_seed)
    confidence_dropout_seed = None if random_state is None else int(random_state) + 209759
    confidence_dropout_rng = np.random.default_rng(confidence_dropout_seed)
    enable_twoview = bool(enable_twoview)
    if not enable_twoview:
        enable_confidence_gene_dropout = False
        print(
            " -> enable_twoview=False: using clean simulated source as the single "
            "source view; reference augmentation, batch injection, dropout, "
            "and noise are disabled."
        )

    # Step 0: use the genes prepared before simulation.
    adata_filtered = adata.copy()
    print(f"Single-cell dataset cells: {adata_filtered.shape[0]}")
    print(f"Single-cell dataset genes: {adata_filtered.shape[1]}")

    if celltype_col in adata_filtered.obs.columns:
        celltype_counts = adata_filtered.obs[celltype_col].value_counts()
        print(f"\nCell type distribution ({celltype_col}):")
        print("-" * 40)
        for celltype, count in celltype_counts.items():
            percentage = (count / adata_filtered.shape[0]) * 100
            print(f"{celltype:>20}: {count:>5} cells ({percentage:>5.1f}%)")
        print("-" * 40)

    print("\nValidating the adata.X count matrix...")
    _validate_count_matrix(adata_filtered.X)
    E = adata_filtered.X
    if not isinstance(E, np.ndarray):
        E = E.toarray()

    _, G = E.shape
    confidence_gene_dropout_max = float(confidence_gene_dropout_max)
    confidence_gene_dropout_kappa = float(confidence_gene_dropout_kappa)
    confidence_gene_dropout_retention = float(confidence_gene_dropout_retention)
    gene_confidence_vector = None
    enable_confidence_gene_dropout = bool(enable_confidence_gene_dropout)
    if enable_confidence_gene_dropout:
        if gene_confidence is None:
            raise ValueError(
                "gene_confidence is required when enable_confidence_gene_dropout=True."
            )
        gene_confidence_vector = np.asarray(gene_confidence, dtype=np.float64)
        if gene_confidence_vector.shape != (G,):
            raise ValueError(
                "gene_confidence must have one value per simulated gene "
                f"({G}); got shape {gene_confidence_vector.shape}."
            )
        if not np.all(np.isfinite(gene_confidence_vector)):
            raise ValueError("gene_confidence must contain only finite values.")
        gene_confidence_vector = np.clip(gene_confidence_vector, 0.0, 1.0)
        print(
            " -> confidence-guided gene dropout enabled "
            f"(max={confidence_gene_dropout_max:.3f}, "
            f"kappa={confidence_gene_dropout_kappa:.3f}, "
            f"retention={confidence_gene_dropout_retention:.3f})."
        )
    celltypes = np.unique(adata_filtered.obs[celltype_col])
    n = len(celltypes)
    idx_dict = {ct: np.where(adata_filtered.obs[celltype_col] == ct)[0] for ct in celltypes}

    # Prepare reference index pools and balanced bank assignments.
    print("\nInitializing reference index pools...")
    ref_idx_pools = []

    if reference_col is not None and reference_col in adata_filtered.obs.columns:
        reference_values = adata_filtered.obs[reference_col]
        valid_reference_mask = reference_values.notna() & (
            reference_values.astype(str).str.strip() != ""
        )
        unique_refs = reference_values.loc[valid_reference_mask].unique()
        if len(unique_refs) > 1:
            print(
                f" -> Detected {len(unique_refs)} reference groups; "
                "assigning reference pools across balanced pseudo-banks."
            )
            for ref in unique_refs:
                ref_idx_dict = {}
                for ct in celltypes:
                    ct_idx = np.where(
                        (adata_filtered.obs[celltype_col] == ct) &
                        (adata_filtered.obs[reference_col] == ref)
                    )[0]
                    if len(ct_idx) > 0:
                        ref_idx_dict[ct] = ct_idx
                    else:
                        ref_idx_dict[ct] = idx_dict[ct]
                ref_idx_pools.append({
                    "name": str(ref),
                    "idx_dict": ref_idx_dict
                })
        else:
            print(
                f" -> Reference column '{reference_col}' contains one category; "
                "using a cycle-level pseudo-reference."
            )
    else:
        print(" -> No valid reference column provided; using a cycle-level pseudo-reference.")

    bank_ids = _build_bank_ids(K, n_pseudo_banks)
    n_cycles = int(bank_ids.max()) + 1
    print(
        f" -> n_pseudo_banks = {n_pseudo_banks}; using {n_cycles} non-empty banks "
        f"for {K} samples"
    )

    reference_schedule = _build_balanced_reference_schedule(
        ref_idx_pools,
        n_cycles,
        reference_rng,
    )
    if reference_schedule:
        schedule_counts = {}
        for pool in reference_schedule:
            pool_name = pool["name"]
            schedule_counts[pool_name] = schedule_counts.get(pool_name, 0) + 1
        schedule_summary = ", ".join(
            f"{name}={count}" for name, count in schedule_counts.items()
        )
        print(f" -> Reference bank schedule: {schedule_summary}")

    # Decide once per bank whether to generate an augmented reference.
    # Reference augmentation is now controlled only by enable_twoview.
    use_ref_cycle_mask = (
        refaug_rng.random(n_cycles) < prob_ref_aug
        if enable_twoview
        else np.zeros(n_cycles, dtype=bool)
    )

    # Assign condition labels.
    print("\nAssigning condition labels...")
    if condition_col is not None and condition_col in adata_filtered.obs.columns:
        unique_conditions = adata_filtered.obs[condition_col].unique()
        condition_labels = np.random.choice(unique_conditions, size=K, replace=True)
        n_conditions = len(unique_conditions)
    else:
        condition_labels = np.random.choice(n_conditions, size=K, replace=True)
        unique_conditions = np.arange(n_conditions)

    # Generate baseline proportions without condition-level gene effects.
    condition_base_proportions = {}
    for cond in unique_conditions:
        condition_base_proportions[cond] = np.random.dirichlet(np.ones(n))

    # Keep only the observed reference view and index pool for the active cycle.
    current_cycle_id = None
    current_ref_E_obs = None
    current_ref_idx_dict = None
    current_ref_name = None

    def _prepare_reference_for_cycle(cycle_id):
        """
        Generate or select one reference when entering a cycle.
        """
        nonlocal current_ref_E_obs, current_ref_idx_dict, current_ref_name
        if reference_schedule:
            selected_pool = reference_schedule[cycle_id]
            current_ref_idx_dict = selected_pool["idx_dict"]
            current_ref_name = selected_pool["name"]
        else:
            current_ref_idx_dict = idx_dict
            current_ref_name = f"original_cycle_{cycle_id}"
        use_aug_this_cycle = bool(use_ref_cycle_mask[cycle_id])
        if use_aug_this_cycle:
            current_ref_E_obs = _apply_reference_style(
                E.copy(),
                scale_sigma=0.5,
                drop_alpha=1.0,
                drop_beta=0.5,
                marker_prob=0.1,
                marker_factor=0.5,
                rng=refaug_rng,
            )
            current_ref_name = f"{current_ref_name}_aug"
        else:
            current_ref_E_obs = E
        return

    # 3. Simulate samples.
    if enable_twoview:
        print("\nSimulating paired clean/observed bulk samples...")
    else:
        print("\nSimulating single-view clean bulk samples...")
    X_bulk = np.zeros((K, G))
    X_obs_bulk = np.zeros((K, G))
    P = np.zeros((K, n))

    all_indices = np.arange(K)
    np.random.shuffle(all_indices)
    n_missing_samples = int(K * missing_celltype_fraction)
    n_dominant_samples = int(K * dominant_sample_fraction)
    missing_sample_indices = all_indices[:n_missing_samples]
    dominant_sample_indices = all_indices[n_missing_samples:n_missing_samples+n_dominant_samples]

    for k in tqdm(range(K), desc="Simulating samples"):
        # Generate or select one reference at the start of each cycle.
        cycle_id = int(bank_ids[k])
        if cycle_id != current_cycle_id:
            current_cycle_id = cycle_id
            _prepare_reference_for_cycle(cycle_id)

        current_condition = condition_labels[k]
        base_p = condition_base_proportions[current_condition]
        p = np.random.dirichlet(base_p * concentration_factor * n)

        # Apply missing-cell-type or dominant-cell-type composition patterns.
        if enable_missing_celltypes and k in missing_sample_indices:
            max_missing = n - min_celltypes_per_sample
            n_to_missing = np.random.randint(1, max_missing + 1) if max_missing > 0 else 0
            if n_to_missing > 0:
                missing_probs = 1.0 - base_p / base_p.sum()
                missing_probs = missing_probs / missing_probs.sum()
                missing_indices = np.random.choice(
                    np.arange(n),
                    size=n_to_missing,
                    replace=False,
                    p=missing_probs
                )
                p[missing_indices] = 0
                if p.sum() > 0:
                    p = p / p.sum()

        elif enable_dominant_celltypes and k in dominant_sample_indices:
            dominant_candidate_idx = np.argmax(base_p)
            dominant_fraction = np.random.uniform(min_dominant_fraction, max_dominant_fraction)
            original_p = p.copy()
            p[dominant_candidate_idx] = dominant_fraction
            remaining_fraction = 1.0 - dominant_fraction
            other_indices = [i for i in range(n) if i != dominant_candidate_idx]
            if len(other_indices) > 0:
                other_original_sum = original_p[other_indices].sum()
                if other_original_sum > 0:
                    p[other_indices] = original_p[other_indices] / other_original_sum * remaining_fraction
                else:
                    p[other_indices] = remaining_fraction / len(other_indices)

        p[p < 1e-10] = 0
        p = p / p.sum() if p.sum() > 0 else np.ones(n) / n

        Nk = np.round(N * p).astype(int)
        if Nk.sum() != N:
            Nk[np.argmax(p)] += (N - Nk.sum())
        P[k, :] = Nk / Nk.sum()

        bulk_vector = np.zeros(G)
        obs_bulk_vector = np.zeros(G)

        for i, ct in enumerate(celltypes):
            Ni = Nk[i]
            if Ni == 0:
                continue

            idx_pool = current_ref_idx_dict[ct]
            chosen_idx = np.random.choice(idx_pool, size=Ni, replace=True)

            subE_clean = E[chosen_idx, :].copy().astype(np.float64)
            subE_obs = current_ref_E_obs[chosen_idx, :].copy().astype(np.float64)

            bulk_vector += subE_clean.sum(axis=0)
            obs_bulk_vector += subE_obs.sum(axis=0)

        X_bulk[k, :] = bulk_vector
        X_obs_bulk[k, :] = obs_bulk_vector

    # Build output data frames.
    index_names = [f"sample_{i+1}" for i in range(K)]
    P_df = pd.DataFrame(P, columns=celltypes, index=index_names)

    # Assign batches.
    batch_ids = np.random.randint(0, n_batches, size=K)
    B_df = pd.DataFrame({"batch": batch_ids}, index=index_names)

    # === 4. Clean source teacher before technical perturbation ===
    X_clean_raw = np.round(X_bulk).astype(int)
    X_clean = np.maximum(X_clean_raw, 0)
    X_clean_df = pd.DataFrame(X_clean, columns=adata_filtered.var_names, index=index_names)
    use_batch_mask = (
        np.random.rand(K) < prob_batch_inj
        if enable_twoview
        else np.zeros(K, dtype=bool)
    )

    # 5. Apply technical noise and batch effects.
    if enable_twoview:
        print("\nApplying technical noise and batch effects with probabilistic masks...")
    else:
        print("\nSkipping two-view technical perturbation; using the clean view as observed input.")

    X_proc = np.maximum(np.round(X_obs_bulk), 0).astype(float)

    if enable_twoview:
        X_aug = X_proc.copy()
        X_batch = _apply_batch_effects(
            X_aug, batch_ids, n_batch_factors, affected_gene_range,
            log_uk_range, sigma_k_range, individual_gene_variance
        )
        X_aug = np.where(use_batch_mask[:, None], X_batch, X_aug)

        if enable_confidence_gene_dropout:
            X_aug = _apply_confidence_gene_dropout(
                X_aug,
                gene_confidence_vector,
                confidence_gene_dropout_max,
                confidence_gene_dropout_kappa,
                confidence_gene_dropout_retention,
                confidence_dropout_rng,
            )
        else:
            X_dropout = _apply_expression_based_dropout(
                X_aug, batch_ids, dropout_expression_threshold,
                dropout_enhancement_factor, batch_dropout_range,
                sample_variance_dropout
            )
            X_aug = np.where(use_batch_mask[:, None], X_dropout, X_aug)

        if gaussian_noise_level > 0:
            X_noisy = X_aug.copy()
            for i in range(X_aug.shape[0]):
                noise = np.zeros(X_aug.shape[1])
                for j in range(X_aug.shape[1]):
                    if X_aug[i, j] > 0:
                        gene_noise_std = gaussian_noise_level * X_aug[i, j]
                        noise[j] = np.round(np.random.normal(0, gene_noise_std))
                X_noisy[i] += noise
            X_noisy = np.maximum(X_noisy, 0)
            X_aug = np.where(use_batch_mask[:, None], X_noisy, X_aug)

        X_proc = X_aug

    X_proc_df = pd.DataFrame(X_proc, columns=adata_filtered.var_names, index=index_names)

    # === 6. Shared SABER analysis-scale transform ===
    X_clean_scaled = transform_bulk_matrix_to_analysis_scale(X_clean)
    X_clean_scaled_df = pd.DataFrame(
        X_clean_scaled,
        columns=adata_filtered.var_names,
        index=index_names,
    )
    X_scaled = transform_bulk_matrix_to_analysis_scale(X_proc)
    X_final_df = pd.DataFrame(X_scaled, columns=adata_filtered.var_names, index=index_names)

    print("\nSimulation complete!")
    return (
        X_final_df,
        X_clean_scaled_df,
        X_proc_df,
        X_clean_df,
        P_df,
        B_df,
        celltypes,
    )


# =========================================================================
# Simulation helper functions
# =========================================================================

def _apply_batch_effects(X, batch_ids, n_batch_factors, affected_gene_range, log_uk_range, sigma_k_range, individual_gene_variance):
    K, G = X.shape
    X_batch = X.copy().astype(float)

    for b in np.unique(batch_ids):
        batch_idx = np.where(batch_ids == b)[0]

        for factor_idx in range(n_batch_factors):
            n_affected = np.random.randint(*affected_gene_range)
            affected_genes = np.random.choice(G, size=n_affected, replace=False)

            log_uk = np.random.uniform(*log_uk_range)
            uk = np.exp(log_uk)
            sigma_k = np.random.uniform(*sigma_k_range)

            total_variance = sigma_k**2 + individual_gene_variance**2
            scale_factors = np.random.normal(loc=uk, scale=np.sqrt(total_variance), size=n_affected)
            scale_factors = np.clip(scale_factors, 0.1, 10.0)

            for j, gene_idx in enumerate(affected_genes):
                X_batch[batch_idx, gene_idx] *= scale_factors[j]

    X_batch = np.maximum(np.round(X_batch), 0)
    return X_batch


def _apply_expression_based_dropout(X, batch_ids, thresholds, enhancement_factors, batch_dropout_range, sample_variance_dropout):
    X_dropout = X.copy()
    dropout_probs = np.full(X.shape, 0.0)

    unique_batches = np.unique(batch_ids)
    batch_base_rates = {b: np.random.uniform(*batch_dropout_range) for b in unique_batches}

    sample_rates = np.zeros(len(X))
    for i, batch_id in enumerate(batch_ids):
        individual_rate = np.random.normal(batch_base_rates[batch_id], sample_variance_dropout)
        sample_rates[i] = np.clip(individual_rate, 0, 0.05)

    for i in range(len(thresholds) + 1):
        if i == 0:
            mask = (X > 0) & (X < thresholds[0])
            enhancement = enhancement_factors[0]
        elif i == len(thresholds):
            mask = (X > 0) & (X >= thresholds[-1])
            enhancement = 1.0
        else:
            mask = (X > 0) & (X >= thresholds[i-1]) & (X < thresholds[i])
            enhancement = enhancement_factors[i]

        for sample_idx in range(len(X)):
            sample_mask = mask[sample_idx]
            if np.any(sample_mask):
                dropout_probs[sample_idx][sample_mask] = sample_rates[sample_idx] * enhancement

    mask = np.random.random(X.shape) < dropout_probs
    X_dropout[mask] = 0.0

    return X_dropout


def _apply_confidence_gene_dropout(
    X,
    gene_confidence,
    dropout_max,
    dropout_kappa,
    dropout_retention,
    rng,
):
    if dropout_max <= 0:
        return X.copy()

    X_dropout = np.asarray(X, dtype=float).copy()
    gene_means = X_dropout.mean(axis=0, keepdims=True)
    dropout_probs = dropout_max * np.power(1.0 - gene_confidence, dropout_kappa)
    keep_mask = rng.random(X_dropout.shape) >= dropout_probs[None, :]
    shrunk = gene_means + dropout_retention * (X_dropout - gene_means)
    X_dropout[~keep_mask] = shrunk[~keep_mask]
    return X_dropout


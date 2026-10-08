# saber/pipeline/prop_train.py
"""
PROP training / unified training pipeline with local MMD

Workflow:
scRNA-seq & Real Bulk
→ align common genes
→ select marker-balanced/HVG/stable genes
→ simulate source paired clean/observed bulk from scRNA-seq
→ preprocess source data
→ build Dual-Domain DataLoader (Source + Target)
→ Stage 1: Pretrain proportion encoder & head on source proportions
→ Stage 2: conditional local MMD finetuning (Source vs Target)
→ return trained model and aligned data
"""

import numpy as np
import torch

from saber.pipeline.simulate import run_simulation
from saber.pipeline.preprocess import preprocess_simulated_data
from saber.pipeline.train import run_training

from saber.data.dataloader import MultiStageDataLoader
from saber.data.transform import (
    resolve_bulk_transform_info,
    transform_bulk_dataframe_to_analysis_scale,
)
from saber.models.saber import SABER
from saber.models.loss import SABERLoss
from saber.simulate.gene_selection import select_genes_for_saber
from saber.reproducibility import set_global_seed


def _select_target_training_bulk(bulk_common, n_target_sample_to_train=None):
    if n_target_sample_to_train is None:
        return bulk_common

    n_target = int(n_target_sample_to_train)
    if n_target <= 0 or n_target >= len(bulk_common):
        return bulk_common
    return bulk_common.iloc[:n_target]


def _resolve_reproducibility(config, simulation_config):
    reproducibility_config = config.get("reproducibility") or {}
    if "seed" in reproducibility_config:
        seed = reproducibility_config["seed"]
    else:
        seed = simulation_config.get("random_state", 42)

    if seed is not None:
        seed = int(seed)

    deterministic = bool(reproducibility_config.get("deterministic", False))
    return seed, deterministic


def _allocate_proportional_subset_counts(counts, n_cells):
    counts = np.asarray(counts, dtype=int)
    n_celltypes = len(counts)
    if n_cells < n_celltypes:
        raise ValueError(
            "reference_subset.n_cells must be at least the number of cell types "
            f"({n_celltypes}); got {n_cells}."
        )

    allocations = np.ones(n_celltypes, dtype=int)
    remaining = int(n_cells) - n_celltypes
    if remaining == 0:
        return allocations

    capacities = counts - allocations
    raw_extra = remaining * counts.astype(float) / counts.sum()
    extras = np.minimum(np.floor(raw_extra).astype(int), capacities)
    allocations += extras
    leftover = int(n_cells) - int(allocations.sum())

    remainders = raw_extra - np.floor(raw_extra)
    while leftover > 0:
        candidates = np.where(allocations < counts)[0]
        if len(candidates) == 0:
            break
        order = sorted(
            candidates,
            key=lambda idx: (-remainders[idx], -counts[idx], idx),
        )
        for idx in order:
            if leftover == 0:
                break
            if allocations[idx] < counts[idx]:
                allocations[idx] += 1
                leftover -= 1

    return allocations


def _subset_reference_adata(adata, config, seed=None):
    subset_config = config.get("reference_subset") or {}
    n_cells = subset_config.get("n_cells")
    if n_cells is None:
        return adata

    n_cells = int(n_cells)
    total_cells = int(getattr(adata, "n_obs", adata.shape[0]))
    if n_cells >= total_cells:
        return adata

    simulation_config = config.get("simulation") or {}
    celltype_col = simulation_config.get("celltype_col", "celltype")
    if celltype_col not in adata.obs.columns:
        raise KeyError(f"celltype_col '{celltype_col}' not found in adata.obs.")

    labels = adata.obs[celltype_col].astype(str).to_numpy()
    celltypes, counts = np.unique(labels, return_counts=True)
    allocations = _allocate_proportional_subset_counts(counts, n_cells)

    rng = np.random.default_rng(seed)
    selected_indices = []
    for celltype, allocation in zip(celltypes, allocations):
        pool = np.where(labels == celltype)[0]
        selected = rng.choice(pool, size=int(allocation), replace=False)
        selected_indices.extend(selected.tolist())

    selected_indices = np.sort(np.asarray(selected_indices, dtype=int))
    subset = adata[selected_indices, :].copy()
    print(
        "[SABER] Reference subset: using "
        f"{subset.n_obs} of {total_cells} cells across {len(celltypes)} cell types."
    )
    return subset


def run_prop_train_from_scRNAseq(
    adata,
    bulk,
    config,
    device="cuda",
):
    """
    Run SABER PROP training (Dual-Domain training) from scRNA-seq and bulk inputs.

    Parameters
    ----------
    adata : AnnData
        Single-cell RNA-seq data (Source pool)
    bulk : pd.DataFrame
        Real unlabelled bulk RNA-seq data (Target Domain)
    config : dict
        Full SABER config dict
    device : str
        "cuda" or "cpu"
    """

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    simulation_config = (config.get("simulation") or {}).copy()
    seed, deterministic = _resolve_reproducibility(config, simulation_config)
    set_global_seed(seed, deterministic=deterministic)
    simulation_config["random_state"] = seed
    adata = _subset_reference_adata(adata, config, seed=seed)

    # ==================================================
    # Step 0. Find common genes and select informative genes
    # ==================================================
    print("[SABER] Aligning scRNA-seq and bulk RNA-seq data on common genes...")
    common_gene = sorted(list(set(adata.var_names) & set(bulk.columns)))
    
    if len(common_gene) == 0:
        raise ValueError("No common genes found between scRNA-seq and bulk RNA-seq data.")

    print(f"[SABER] Number of common genes: {len(common_gene)}")

    bulk_common = bulk[common_gene]
    adata_common = adata[:, common_gene]
    real_bulk_transform_info = resolve_bulk_transform_info(
        bulk_common.values,
    )
    simulation_config["real_bulk_transform_info"] = real_bulk_transform_info
    print(f"[SABER] Real Bulk transform decision: {real_bulk_transform_info['summary']}")

    bulk_train_common = _select_target_training_bulk(
        bulk_common,
        simulation_config.get("n_target_sample_to_train"),
    )
    if len(bulk_train_common) != len(bulk_common):
        print(
            "[SABER] Target training subset: using first "
            f"{len(bulk_train_common)} of {len(bulk_common)} target samples "
            "for gene selection and MMD adaptation."
        )

    selected_genes, selection_info = select_genes_for_saber(
        adata=adata_common,
        bulk=bulk_train_common,
        sim_config=simulation_config,
        return_info=True,
    )

    adata_align = adata_common[:, selected_genes].copy()
    bulk_align = bulk_common[selected_genes]
    bulk_train_align = bulk_train_common[selected_genes]
    print(f"[SABER] Number of selected genes: {len(selected_genes)}")

    # ==================================================
    # Step 1. Simulate bulk data from scRNA-seq (Source Domain)
    # ==================================================
    enable_gene_selection = bool(simulation_config.get("enable_gene_selection", True))
    enable_twoview = bool(simulation_config.get("enable_twoview", True))
    requested_confidence_dropout = bool(
        simulation_config.get("enable_confidence_gene_dropout", True)
    )
    simulation_config["gene_confidence"] = selection_info.get(
        "gene_confidence",
        [1.0] * len(selected_genes),
    )
    simulation_config["enable_confidence_gene_dropout"] = (
        enable_gene_selection and enable_twoview and requested_confidence_dropout
    )
    if enable_twoview:
        print("[SABER] Simulating paired clean/observed bulk data from scRNA-seq...")
    else:
        print("[SABER] Simulating single-view clean bulk data from scRNA-seq...")
    sim_data = run_simulation(
        adata=adata_align,
        sim_config=simulation_config,
    )

    # ==================================================
    # Step 2. Preprocess simulated data & Align Target Bulk
    # ==================================================
    print("[SABER] Preprocessing simulated data...")
    processed = preprocess_simulated_data(
        sim_data=sim_data
    )
    
    celltypes = processed["P"].columns.tolist()
    num_cell_types = len(celltypes)
    
    genes = processed["gene_names"]
    gene_dim = len(genes)

    if genes != selected_genes:
        raise RuntimeError(
            "Selected gene order changed during simulation preprocessing. "
            "Source and target genes must remain identical."
        )
    
    print("[SABER] Aligning Target Bulk genes to Source genes...")
    # Force target bulk to use the exact selected gene order consumed by source simulation.
    bulk_align = bulk_align[genes] 
    bulk_train_align = bulk_train_align[genes]
    print(f"[SABER] Final Target Bulk dim: {bulk_align.shape}")
    if len(bulk_train_align) != len(bulk_align):
        print(f"[SABER] Target Bulk dim for training: {bulk_train_align.shape}")

    print("[SABER] Transforming Target Bulk to shared SABER analysis scale (auto)...")
    bulk_target_scaled, target_transform_info = transform_bulk_dataframe_to_analysis_scale(
        bulk_train_align,
        transform_info=real_bulk_transform_info,
        return_info=True,
    )
    print(f"[SABER] Target Bulk transform decision reused: {target_transform_info['summary']}")

    # ==================================================
    # Step 3. Build Dual-Domain DataLoader
    # ==================================================
    enable_twoview = bool(processed.get("enable_twoview", True))
    if enable_twoview:
        print("[SABER] Building Source and Target DataLoaders with paired source views...")
    else:
        print("[SABER] Building Source and Target DataLoaders with single source view...")
    multi_loader = MultiStageDataLoader(
        X_source=processed["X_obs"],
        X_source_clean=processed["X_clean"] if enable_twoview else None,
        P_source=processed["P"],
        X_target=bulk_target_scaled,
        batch_size=config["dataloader"].get("batch_size", 128),
        train_ratio=config["dataloader"].get("train_ratio", 0.8),
        device=device,
        seed=seed,
    )

    # ==================================================
    # Step 4. Initialize model and loss
    # ==================================================
    print("[SABER] Initializing proportion-only SABER model and local MMD loss...")
    model_config = config["model"].copy()
    model_config["num_cell_types"] = num_cell_types
    model_config["gene_dim"] = gene_dim
    model_config["prop_dim"] = model_config.get("prop_dim", 64)
    model_config["analysis_transform"] = "auto"
    model_config["target_bulk_transform_info"] = target_transform_info
    model = SABER(model_config).to(device)

    criterion = SABERLoss()

    # ==================================================
    # Step 5. Run Dual-Track Training
    # ==================================================
    print("[SABER] Run proportion-only local MMD Training...")
    
    train_config = config["training"].copy()
    train_config.update(config.get("loss", {}))

    model, history = run_training(
        model=model,
        criterion=criterion,
        multi_loader=multi_loader,
        config=train_config,
        device=device,
    )

    return model, history, bulk_align, genes, celltypes, model_config

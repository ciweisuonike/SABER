# saber/cli.py

import argparse
import os

def main():
    parser = argparse.ArgumentParser(
        prog="saber",
        description="SABER: proportion-only bulk RNA-seq deconvolution with local MMD adaptation"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    prop_parser = subparsers.add_parser(
        "prop",
        help="Train SABER-PROP and predict proportions"
    )
    prop_parser.add_argument(
        "--adata",
        type=str,
        required=True,
        help="Input scRNA-seq AnnData (.h5ad) as Source Domain"
    )

    prop_parser.add_argument(
        "--celltypes",
        nargs="+",
        default=None,
        help="List of cell types to keep from the scRNA-seq reference. If omitted, use all."
    )

    prop_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="YAML config file"
    )

    prop_parser.add_argument(
        "--bulk",
        type=str,
        required=True,
        help="Real unlabelled bulk data (.csv) as Target Domain"
    )

    prop_parser.add_argument(
        "--prop-dir",
        type=str,
        required=True,
        help="Output directory for proportions"
    )

    prop_parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Optional output directory for model.pt and bulk_align.csv"
    )

    prop_parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda or cpu"
    )

    prop_parser.add_argument(
        "--prefix",
        type=str,
        default="proportions",
        help="File prefix for proportions output"
    )

    prop_predict_parser = subparsers.add_parser(
        "prop-predict",
        help="Predict proportions with a trained SABER-PROP model"
    )
    prop_predict_parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model file path (.pt)"
    )

    prop_predict_parser.add_argument(
        "--bulk",
        type=str,
        required=True,
        help="Bulk data file path (.csv)"
    )

    prop_predict_parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda or cpu"
    )

    prop_predict_parser.add_argument(
        "--prop-dir",
        type=str,
        required=True,
        help="Output directory for proportions"
    )

    prop_predict_parser.add_argument(
        "--prefix",
        type=str,
        default="proportions",
        help="File prefix for proportions output"
    )

    args = parser.parse_args()
    if args.command == "prop":
        run_prop_cli(args)
    elif args.command == "prop-predict":
        run_prop_predict_cli(args)


def run_prop_cli(args):
    model, genes, celltypes = run_prop_train_cli(args)

    print("\n[SABER] Running SABER-PROP prediction on input bulk...")
    os.makedirs(args.prop_dir, exist_ok=True)

    from saber.pipeline.predict import predict_bulk_with_model

    proportions = predict_bulk_with_model(
        model=model,
        genes=genes,
        celltypes=celltypes,
        bulk_file=args.bulk,
        device=args.device
    )
    proportions.to_csv(os.path.join(args.prop_dir, f"{args.prefix}.csv"), index=True)

    print(f"[SABER] Proportions saved to {args.prop_dir}")
    print("[SABER] SABER-PROP completed successfully.")


def run_prop_train_cli(args):
    import yaml
    import torch
    import scanpy as sc
    import pandas as pd

    from saber.pipeline.prop_train import run_prop_train_from_scRNAseq

    print("\n[SABER] Loading config...")
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    print("[SABER] Loading scRNA-seq (Source) data...")
    adata = sc.read_h5ad(args.adata)

    if args.celltypes is not None:
        try:
            celltype_col = config["simulation"]["celltype_col"]
        except KeyError as exc:
            raise KeyError(
                "Explicit --celltypes filtering requires simulation.celltype_col in the config."
            ) from exc

        print(f"[SABER] Filtering scRNA-seq using column '{celltype_col}'...")
        if celltype_col not in adata.obs.columns:
            raise KeyError(
                f"Column '{celltype_col}' not found in adata.obs. "
                f"Available columns: {adata.obs.columns.tolist()}"
            )

        initial_count = adata.n_obs
        adata = adata[adata.obs[celltype_col].isin(args.celltypes)].copy()
        print(
            f"[SABER] Kept {len(args.celltypes)} cell types. "
            f"Cells: {initial_count} -> {adata.n_obs}"
        )
        if adata.n_obs == 0:
            raise ValueError(f"No cells left after filtering for {args.celltypes}!")

    print("[SABER] Loading bulk (Target) data...")
    bulk = pd.read_csv(args.bulk, index_col=0)

    print("\n[SABER] Running SABER-PROP training...")
    model, history, bulk_align, genes, celltypes, model_config = run_prop_train_from_scRNAseq(
        adata=adata,
        config=config,
        bulk=bulk,
        device=args.device,
    )

    if args.model_dir is not None:
        os.makedirs(args.model_dir, exist_ok=True)

        model_save_path = os.path.join(args.model_dir, "model.pt")
        print(f"\n[SABER] Saving model to {model_save_path}")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": model_config,
                "history": history,
                "common_gene": genes,
                "celltypes": celltypes
            },
            model_save_path
        )

        bulk_align.to_csv(os.path.join(args.model_dir, "bulk_align.csv"), index=True, header=True)
    else:
        print("\n[SABER] No model directory provided; skipping model and bulk_align.csv saving.")

    print("[SABER] PROP training completed successfully.")
    return model, genes, celltypes


def run_prop_predict_cli(args):
    from saber.pipeline.predict import run_predict

    print("\n[SABER] Running proportion prediction...")
    os.makedirs(args.prop_dir, exist_ok=True)

    proportions = run_predict(
        model_path=args.model,
        bulk_file=args.bulk,
        device=args.device
    )
    
    proportions.to_csv(os.path.join(args.prop_dir, f"{args.prefix}.csv"), index=True)
    
    print(f"[SABER] Proportions saved to {args.prop_dir}")
    print("[SABER] Prediction completed successfully.")


if __name__ == "__main__":
    main()

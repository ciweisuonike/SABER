import torch
import pandas as pd

from saber.models.saber import SABER
from saber.data.transform import (
    resolve_bulk_transform_info,
    transform_bulk_matrix_to_analysis_scale,
)


def load_model(model_path, device="cuda", return_config=False):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)

    model = SABER(config=checkpoint["config"]).to(device)
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError as exc:
        raise RuntimeError(
            "The checkpoint architecture is incompatible with the proportion-only "
            "model. Train a compatible checkpoint before prediction."
        ) from exc
    model.eval()

    genes = checkpoint["common_gene"]
    celltypes = checkpoint["celltypes"]
    model_config = checkpoint["config"]
    if return_config:
        return model, genes, celltypes, model_config
    return model, genes, celltypes


def load_bulk_data(bulk_file, target_genes):
    bulk_data = pd.read_csv(bulk_file, index_col=0)

    missing_genes = set(target_genes) - set(bulk_data.columns)
    if missing_genes:
        print(f"[Warning] Input data is missing {len(missing_genes)} training genes; filling them with 0.")
        for gene in missing_genes:
            bulk_data[gene] = 0.0

    bulk_data = bulk_data[target_genes]

    transform_info = resolve_bulk_transform_info(bulk_data.values)
    x_scaled, transform_info = transform_bulk_matrix_to_analysis_scale(
        bulk_data.values,
        transform_info=transform_info,
        return_info=True,
    )

    print(
        f"[Predict] Loaded, aligned, and scaled bulk data with shape: {x_scaled.shape} "
        "(auto)"
    )
    print(f"[Predict] Bulk transform: {transform_info['summary']}")
    return x_scaled, bulk_data.index


def predict(X_numpy, model, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        x_tensor = torch.FloatTensor(X_numpy).to(device)
        proportions, _, _z_prop = model(x_tensor)

        proportions = proportions.cpu().numpy()

    return proportions


def predict_bulk_with_model(model, genes, celltypes, bulk_file, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()
    x_numpy, sample_indices = load_bulk_data(bulk_file, genes)
    proportions = predict(x_numpy, model, device)

    df_proportions = pd.DataFrame(proportions, columns=celltypes, index=sample_indices)

    return df_proportions


def run_predict(model_path, bulk_file, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    model, genes, celltypes, _model_config = load_model(
        model_path,
        device,
        return_config=True,
    )
    return predict_bulk_with_model(model, genes, celltypes, bulk_file, device)

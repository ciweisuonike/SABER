#!/usr/bin/env python3
from __future__ import annotations
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path("/Path/to/PBMC")
OUTPUT_DIR = Path("/Path/to/output/PBMC_monaco_cross_platform_consistency")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


REF_LABELS = ["3k", "4k", "6k", "8k", "20k", "33k", "Combined"]
PLOT_REF_LABELS = REF_LABELS + ["Mean"]
BENCHMARK_METHODS = ["SABER", "DWLS", "BayesPrism", "Scadenpytorch", "TAPE", "RNASieve", "NNLS"]
ABLATION_METHODS = [
    "SABER",
    "SABER_wo_alignment",
    "SABER_wo_twoview",
    "SABER_wo_gene_selection",
    "SABER_w_gene_selection",
    "SABER_w_twoview",
    "SABER_w_alignment",
    "SABER_wo_all",
]
METHOD_LABELS = {
    "SABER": "SABER",
    "Scadenpytorch": "Scaden",
    "RNASieve": "RNASieve",
    "NNLS": "NNLS",
    "TAPE": "TAPE",
    "BayesPrism": "BayesPrism",
    "DWLS": "DWLS",
    "SABER_wo_alignment": "wo_alignment",
    "SABER_wo_twoview": "wo_twoview",
    "SABER_wo_gene_selection": "wo_gene_selection",
    "SABER_w_gene_selection": "w_gene_selection",
    "SABER_w_twoview": "w_twoview",
    "SABER_w_alignment": "w_alignment",
    "SABER_wo_all": "wo_all",
}
sns.set_theme(style="white", context="paper")
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def read_prediction(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index = [str(x).strip().strip('"').strip("'") for x in df.index]
    df.columns = [str(x).strip() for x in df.columns]
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how="all")
    return df


def prediction_filename(method: str, ref_code: str, mode: str) -> str:
    if mode == "benchmark" or method == "SABER":
        return f"{method}{ref_code}.csv"
    suffix = method.replace("SABER_", "")
    return f"SABER{ref_code}_{suffix}.csv" if ref_code else f"SABER_{suffix}.csv"


def numeric_pair(x: Sequence[float], y: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    return (x_arr[mask], y_arr[mask])


def safe_pearson(x: Sequence[float], y: Sequence[float]) -> float:
    x_arr, y_arr = numeric_pair(x, y)
    if x_arr.size < 2 or np.std(x_arr) == 0 or np.std(y_arr) == 0:
        return np.nan
    return float(pearsonr(x_arr, y_arr)[0])


def compute_ccc(x: Sequence[float], y: Sequence[float]) -> float:
    x_arr, y_arr = numeric_pair(x, y)
    if x_arr.size < 2 or x_arr.size != y_arr.size:
        return np.nan
    mean_x = float(np.mean(x_arr))
    mean_y = float(np.mean(y_arr))
    var_x = float(np.var(x_arr, ddof=0))
    var_y = float(np.var(y_arr, ddof=0))
    covariance = float(np.mean((x_arr - mean_x) * (y_arr - mean_y)))
    denominator = var_x + var_y + (mean_x - mean_y) ** 2
    if np.isclose(denominator, 0.0, atol=1e-15, rtol=0.0):
        return np.nan
    if np.isclose(var_x, 0.0, atol=1e-15, rtol=0.0) or np.isclose(var_y, 0.0, atol=1e-15, rtol=0.0):
        return 0.0
    return float(2.0 * covariance / denominator)


def arithmetic_mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    return float(np.mean(arr))


def renormalize_rows(df: pd.DataFrame) -> pd.DataFrame:
    row_sums = df.sum(axis=1, skipna=True)
    valid = np.isfinite(row_sums) & (row_sums != 0)
    out = df.copy()
    out.loc[valid, :] = out.loc[valid, :].div(row_sums.loc[valid], axis=0)
    return out


def calculate_pair_metrics(
    rna_df: pd.DataFrame, array_df: pd.DataFrame
) -> Tuple[Dict[str, float], List[Dict], List[Dict]]:
    sample_pearsons: List[float] = []
    sample_cccs: List[float] = []
    sample_details: List[Dict] = []
    for sample in rna_df.index:
        x = rna_df.loc[sample].to_numpy(dtype=float)
        y = array_df.loc[sample].to_numpy(dtype=float)
        pearson_r = safe_pearson(x, y)
        ccc = compute_ccc(x, y)
        sample_pearsons.append(pearson_r)
        sample_cccs.append(ccc)
        sample_details.append(
            {"Sample": sample, "PearsonR": pearson_r, "CCC": ccc, "NCellTypes": len(rna_df.columns)}
        )
    celltype_pearsons: List[float] = []
    celltype_cccs: List[float] = []
    celltype_details: List[Dict] = []
    for celltype in rna_df.columns:
        x = rna_df[celltype].to_numpy(dtype=float)
        y = array_df[celltype].to_numpy(dtype=float)
        pearson_r = safe_pearson(x, y)
        ccc = compute_ccc(x, y)
        celltype_pearsons.append(pearson_r)
        celltype_cccs.append(ccc)
        celltype_details.append(
            {"CellType": celltype, "PearsonR": pearson_r, "CCC": ccc, "NSamples": len(rna_df.index)}
        )
    summary = {
        "sample_pearsonr": arithmetic_mean(sample_pearsons),
        "sample_ccc": arithmetic_mean(sample_cccs),
        "celltype_pearsonr": arithmetic_mean(celltype_pearsons),
        "celltype_ccc": arithmetic_mean(celltype_cccs),
    }
    return (summary, sample_details, celltype_details)


def build_summary_row(
    method: str, ref_label: str, raw_records: List[Dict], unit_type: str, n_other: int
) -> Dict:
    pearson_values = [record["PearsonR"] for record in raw_records]
    ccc_values = [record["CCC"] for record in raw_records]
    pearson_arr = np.asarray(pearson_values, dtype=float)
    ccc_arr = np.asarray(ccc_values, dtype=float)
    row = {
        "Method": method,
        "MethodLabel": METHOD_LABELS.get(method, method),
        "Ref": ref_label,
        "Mean_PearsonR": arithmetic_mean(pearson_values),
        "Mean_CCC": arithmetic_mean(ccc_values),
        "Valid_Pearson_Count": int(np.isfinite(pearson_arr).sum()),
        "Valid_CCC_Count": int(np.isfinite(ccc_arr).sum()),
    }
    if unit_type == "celltype":
        row["Total_CellTypes"] = len(raw_records)
        row["NSamples"] = n_other
    elif unit_type == "sample":
        row["Total_Samples"] = len(raw_records)
        row["NCellTypes"] = n_other
    else:
        raise ValueError(f"Unknown unit_type: {unit_type}")
    return row


METRIC_TITLES = {
    "sample_pearsonr": "Per-sample Pearson r",
    "sample_ccc": "Per-sample CCC",
    "celltype_pearsonr": "Per-cell-type Pearson r",
    "celltype_ccc": "Per-cell-type CCC",
}
def plot_heatmap(
    matrix: pd.DataFrame,
    metric_name: str,
    methods: Sequence[str],
    output_prefix: str,
    title_prefix: str,
) -> None:
    plot_matrix = matrix.reindex(index=methods, columns=PLOT_REF_LABELS)
    values = plot_matrix.to_numpy(dtype=float)
    cmap = sns.blend_palette(
        ["#5BA6C9", "#DCEAF1", "#F7F7F2", "#F6D8AE", "#E89A6A", "#C65353"],
        as_cmap=True
    )
    cmap.set_bad("#E5E5E5")
    if metric_name.endswith("pearsonr"):
        norm = mpl.colors.Normalize(vmin=0.5, vmax=1.0, clip=True)
    else:
        norm = mpl.colors.Normalize(vmin=0.0, vmax=0.75, clip=True)
    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    image = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(len(PLOT_REF_LABELS)))
    ax.set_xticklabels(PLOT_REF_LABELS, rotation=0)
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels([METHOD_LABELS.get(method, method) for method in methods])
    ax.set_xlabel("Reference")
    ax.set_ylabel("Method")
    ax.axvline(len(REF_LABELS) - 0.5, color="black", linewidth=1.1, linestyle="-", zorder=5)
    ax.set_xticks(np.arange(-0.5, values.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, values.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)
    for tick in ax.get_xticklabels():
        if tick.get_text() == "Mean":
            tick.set_fontweight("bold")
    best_mask = np.zeros(values.shape, dtype=bool)
    for col_idx in range(values.shape[1]):
        col = values[:, col_idx]
        finite = np.isfinite(col)
        best = np.nanmax(col)
        best_mask[:, col_idx] = finite & np.isclose(col, best, rtol=1e-09, atol=1e-12)
    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            if not np.isfinite(value):
                label = "NA"
                color = "black"
                weight = "normal"
            else:
                label = f"{value:.2f}"
                if best_mask[row_idx, col_idx]:
                    color = "white"
                    weight = "bold"
                else:
                    color = "black"
                    weight = "medium"
            ax.text(
                col_idx,
                row_idx,
                label,
                ha="center",
                va="center",
                fontsize=8.0,
                color=color,
                fontweight=weight,
                zorder=6,
            )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
        spine.set_color("black")
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    cbar.set_label(METRIC_TITLES[metric_name], rotation=90, labelpad=10)
    cbar.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_title(
        f"{title_prefix}: {METRIC_TITLES[metric_name]}\nGSE107011 RNA-seq vs GSE106898 Array",
        pad=10,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / f"{output_prefix}_per_{metric_name}.pdf",
        format="pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def run_analysis(
    mode: str,
    methods: Sequence[str],
    output_prefix: str,
    title_prefix: str,
) -> None:
    paired_predictions = {}
    required_celltypes = None
    for method in methods:
        for ref_code, ref_label in zip(["3k", "4k", "6k", "8k", "20k", "33k", ""], REF_LABELS):
            filename = prediction_filename(method, ref_code, mode)
            rna_df = read_prediction(PROJECT_ROOT / "results" / "monaco" / filename)
            array_df = read_prediction(PROJECT_ROOT / "results" / "monacoarray" / filename)
            if required_celltypes is None:
                required_celltypes = list(rna_df.columns)
            paired_predictions[method, ref_label] = (
                rna_df.loc[:, required_celltypes].copy(),
                array_df.loc[:, required_celltypes].copy(),
            )

    matrices = {
        metric: pd.DataFrame(np.nan, index=methods, columns=REF_LABELS, dtype=float)
        for metric in ("sample_pearsonr", "sample_ccc", "celltype_pearsonr", "celltype_ccc")
    }
    per_sample_raw_records: List[Dict] = []
    per_celltype_raw_records: List[Dict] = []
    per_sample_summary_records: List[Dict] = []
    per_celltype_summary_records: List[Dict] = []
    for method in methods:
        for ref_label in REF_LABELS:
            rna_df, array_df = paired_predictions[method, ref_label]
            common_samples = sorted(set(rna_df.index).intersection(array_df.index))
            rna_sub = renormalize_rows(rna_df.loc[common_samples, required_celltypes].copy())
            array_sub = renormalize_rows(array_df.loc[common_samples, required_celltypes].copy())
            metrics, sample_details, celltype_details = calculate_pair_metrics(rna_sub, array_sub)
            for record in sample_details:
                per_sample_raw_records.append(
                    {
                        "Method": method,
                        "MethodLabel": METHOD_LABELS.get(method, method),
                        "Ref": ref_label,
                        **record,
                    }
                )
            for record in celltype_details:
                per_celltype_raw_records.append(
                    {
                        "Method": method,
                        "MethodLabel": METHOD_LABELS.get(method, method),
                        "Ref": ref_label,
                        **record,
                    }
                )
            per_sample_summary_records.append(
                build_summary_row(
                    method=method,
                    ref_label=ref_label,
                    raw_records=sample_details,
                    unit_type="sample",
                    n_other=len(required_celltypes),
                )
            )
            per_celltype_summary_records.append(
                build_summary_row(
                    method=method,
                    ref_label=ref_label,
                    raw_records=celltype_details,
                    unit_type="celltype",
                    n_other=len(common_samples),
                )
            )
            for metric, value in metrics.items():
                matrices[metric].loc[method, ref_label] = value

    pd.DataFrame(per_celltype_raw_records).to_csv(
        OUTPUT_DIR / f"{output_prefix}_per_celltype_raw.csv", index=False
    )
    pd.DataFrame(per_celltype_summary_records).to_csv(
        OUTPUT_DIR / f"{output_prefix}_per_celltype_summary.csv", index=False
    )
    pd.DataFrame(per_sample_raw_records).to_csv(
        OUTPUT_DIR / f"{output_prefix}_per_sample_raw.csv", index=False
    )
    pd.DataFrame(per_sample_summary_records).to_csv(
        OUTPUT_DIR / f"{output_prefix}_per_sample_summary.csv", index=False
    )
    for metric_name in ("sample_pearsonr", "sample_ccc", "celltype_pearsonr", "celltype_ccc"):
        matrix = matrices[metric_name].copy()
        matrix["Mean"] = [
            arithmetic_mean(matrix.loc[method, REF_LABELS].to_numpy(dtype=float))
            for method in matrix.index
        ]
        plot_heatmap(
            matrix[PLOT_REF_LABELS],
            metric_name,
            methods,
            output_prefix,
            title_prefix,
        )


run_analysis(
    mode="benchmark",
    methods=BENCHMARK_METHODS,
    output_prefix="monaco_cross_platform",
    title_prefix="Monaco cross-platform consistency",
)
run_analysis(
    mode="ablation",
    methods=ABLATION_METHODS,
    output_prefix="monaco_cross_platform_ablation",
    title_prefix="Monaco cross-platform ablation consistency",
)

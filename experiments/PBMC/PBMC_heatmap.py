#!/usr/bin/env python3
from __future__ import annotations
import warnings
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.ticker import MaxNLocator
from scipy.stats import pearsonr
from matplotlib.colors import Normalize, TwoSlopeNorm, PowerNorm
warnings.filterwarnings("ignore")
PROJECT_ROOT = Path("/Path/to/PBMC")
OUTPUT_DIR = Path("/Path/to/output/PBMC_heatmaps")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


REF_LABELS = ["3k", "4k", "6k", "8k", "20k", "33k", "Combined"]
PLOT_REF_LABELS = REF_LABELS + ["Mean"]
SCENARIOS = [
    {
        "dataset": "monaco",
        "label": "RNA-seq: GSE107011",
        "true_path": PROJECT_ROOT / "data" / "monaco" / "P_merged.csv",
        "results_dir": PROJECT_ROOT / "results" / "monaco",
        "preprocess": "none",
    },
    {
        "dataset": "monacoarray",
        "label": "Array: GSE106898",
        "true_path": PROJECT_ROOT / "data" / "monacoarray" / "P_merged.csv",
        "results_dir": PROJECT_ROOT / "results" / "monacoarray",
        "preprocess": "none",
    },
    {
        "dataset": "quantiseq",
        "label": "RNA-seq: GSE107572",
        "true_path": PROJECT_ROOT / "data" / "quantiseq" / "proportions.csv",
        "results_dir": PROJECT_ROOT / "results" / "quantiseq",
        "preprocess": "none",
    },
    {
        "dataset": "sdy67",
        "label": "RNA-seq: SDY67",
        "true_path": PROJECT_ROOT / "data" / "sdy67" / "proportions.csv",
        "results_dir": PROJECT_ROOT / "results" / "sdy67",
        "preprocess": "none",
    },
    {
        "dataset": "scper",
        "label": "Array: GSE107990",
        "true_path": PROJECT_ROOT / "data" / "scper" / "proportions.csv",
        "results_dir": PROJECT_ROOT / "results" / "scper",
        "preprocess": "none",
    },
    {
        "dataset": "shaw",
        "label": "Array: GSE59654",
        "true_path": PROJECT_ROOT / "data" / "shaw" / "proportions.csv",
        "results_dir": PROJECT_ROOT / "results" / "shaw",
        "preprocess": "keep_shaw_4_renorm",
    },
    {
        "dataset": "newman",
        "label": "Array: GSE65133",
        "true_path": PROJECT_ROOT / "data" / "newman" / "proportions.csv",
        "results_dir": PROJECT_ROOT / "results" / "newman",
        "preprocess": "drop_dendritics_renorm",
    },
    {
        "dataset": "morandini",
        "label": "RNA-seq: GSE193141",
        "true_path": PROJECT_ROOT / "data" / "morandini" / "proportions.csv",
        "results_dir": PROJECT_ROOT / "results" / "morandini",
        "preprocess": "drop_dendritics_renorm",
    },
    {
        "dataset": "harrison",
        "label": "RNA-seq: GSE120502",
        "true_path": PROJECT_ROOT / "data" / "harrison" / "proportions.csv",
        "results_dir": PROJECT_ROOT / "results" / "harrison",
        "preprocess": "drop_dendritics_renorm",
    },
]
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
    "SABER_wo_alignment": "w/o alignment",
    "SABER_wo_twoview": "w/o two-view",
    "SABER_wo_gene_selection": "w/o gene selection",
    "SABER_w_gene_selection": "w/ gene selection",
    "SABER_w_twoview": "w/ two-view",
    "SABER_w_alignment": "w/ alignment",
    "SABER_wo_all": "w/o all",
}
sns.set_theme(style="white", context="paper")
plt.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 10,
        "axes.labelsize": 8,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 5.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def safe_pearson(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    if y_true_arr.size < 2:
        return np.nan
    if np.isclose(np.std(y_true_arr), 0.0, atol=1e-15, rtol=0.0):
        return np.nan
    if np.isclose(np.std(y_pred_arr), 0.0, atol=1e-15, rtol=0.0):
        return np.nan
    return float(pearsonr(y_true_arr, y_pred_arr)[0])


def compute_ccc(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    if y_true_arr.size < 2:
        return np.nan
    mean_true = float(np.mean(y_true_arr))
    mean_pred = float(np.mean(y_pred_arr))
    var_true = float(np.var(y_true_arr))
    var_pred = float(np.var(y_pred_arr))
    covariance = float(np.mean((y_true_arr - mean_true) * (y_pred_arr - mean_pred)))
    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
    if np.isclose(denominator, 0.0, atol=1e-15, rtol=0.0):
        return np.nan
    if np.isclose(var_true, 0.0, atol=1e-15, rtol=0.0) or np.isclose(var_pred, 0.0, atol=1e-15, rtol=0.0):
        return 0.0
    return float(2.0 * covariance / denominator)


def summarize_metric_values(values: Sequence[float]) -> Tuple[float, int, int, int, float]:
    arr = np.asarray(values, dtype=float)
    total_count = int(arr.size)
    valid_count = int(np.isfinite(arr).sum())
    undefined_count = total_count - valid_count
    mean_value = float(np.nanmean(arr))
    valid_rate = float(valid_count / total_count)
    return (mean_value, valid_count, total_count, undefined_count, valid_rate)


def common_true_pred(
    true_raw: pd.DataFrame, pred_raw: pd.DataFrame, scenario: Dict
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    true_df = true_raw.copy()
    pred_df = pred_raw.copy()
    true_df.index = true_df.index.astype(str)
    pred_df.index = pred_df.index.astype(str)
    if scenario.get("preprocess", "none") == "drop_dendritics_renorm":
        pred_df = pred_df.drop(columns=["Dendritics"])
        row_sums = pred_df.sum(axis=1)
        row_sums[row_sums == 0] = 1
        pred_df = pred_df.div(row_sums, axis=0)
    elif scenario.get("preprocess", "none") == "keep_shaw_4_renorm":
        required_celltypes = list(true_df.columns)
        true_df = true_df[required_celltypes].apply(pd.to_numeric, errors="coerce")
        pred_df = pred_df[required_celltypes].apply(pd.to_numeric, errors="coerce")
        true_row_sums = true_df.sum(axis=1)
        true_row_sums[true_row_sums == 0] = 1
        true_df = true_df.div(true_row_sums, axis=0)
        pred_row_sums = pred_df.sum(axis=1)
        pred_row_sums[pred_row_sums == 0] = 1
        pred_df = pred_df.div(pred_row_sums, axis=0)
    required_samples = list(true_df.index)
    required_celltypes = list(true_df.columns)
    true_sub = true_df.loc[required_samples, required_celltypes].apply(pd.to_numeric, errors="coerce")
    pred_sub = pred_df.loc[required_samples, required_celltypes].apply(pd.to_numeric, errors="coerce")
    return (true_sub, pred_sub, required_samples, required_celltypes)


def build_matrix(data: pd.DataFrame, metric_slug: str, mode: str, methods: Sequence[str]) -> pd.DataFrame:
    plot_data = data[data["MetricSlug"] == metric_slug]
    plot_data = plot_data[plot_data["Mode"] == mode]
    base_matrix = plot_data.pivot_table(
        index=["Dataset", "Ref"], columns="BaseMethod", values="Value", aggfunc="mean"
    ).reindex(
        index=pd.MultiIndex.from_tuples(
            [(scenario["dataset"], ref) for scenario in SCENARIOS for ref in REF_LABELS],
            names=["Dataset", "Ref"],
        ),
        columns=list(methods),
    )
    output_parts: List[pd.DataFrame] = []
    for scenario in SCENARIOS:
        dataset = scenario["dataset"]
        output_parts.append(
            base_matrix.reindex(
                pd.MultiIndex.from_product([[dataset], REF_LABELS], names=["Dataset", "Ref"])
            )
        )
        mean_values = base_matrix.reindex(
            pd.MultiIndex.from_product([[dataset], REF_LABELS], names=["Dataset", "Ref"])
        ).mean(axis=0, skipna=True)
        mean_row = pd.DataFrame(
            [mean_values.to_numpy(dtype=float)],
            index=pd.MultiIndex.from_tuples([(dataset, "Mean")], names=["Dataset", "Ref"]),
            columns=list(methods),
        )
        output_parts.append(mean_row)
    matrix = pd.concat(output_parts, axis=0)
    return matrix.reindex(
        index=pd.MultiIndex.from_tuples(
            [(scenario["dataset"], ref) for scenario in SCENARIOS for ref in PLOT_REF_LABELS],
            names=["Dataset", "Ref"],
        ),
        columns=list(methods),
    )


def annotate_heatmap_cells(ax: plt.Axes, matrix: pd.DataFrame, higher_is_better: bool) -> None:
    values = matrix.to_numpy(dtype=float)
    for row_idx in range(values.shape[0]):
        row = values[row_idx]
        finite_mask = np.isfinite(row)
        best_value = np.nanmax(row) if higher_is_better else np.nanmin(row)
        best_mask = finite_mask & np.isclose(row, best_value, rtol=1e-09, atol=1e-12)
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            if not np.isfinite(value):
                label = "NA"
                text_color = "black"
                fontweight = "normal"
            else:
                label = f"{value:.2f}"
                if best_mask[col_idx]:
                    text_color = "white"
                    fontweight = "bold"
                else:
                    text_color = "black"
                    fontweight = "medium"
            ax.text(
                col_idx,
                row_idx,
                label,
                ha="center",
                va="center",
                fontsize=4.8,
                color=text_color,
                fontweight=fontweight,
                zorder=5,
                clip_on=True,
            )


def style_heatmap_axis(ax: plt.Axes, matrix: pd.DataFrame, methods: Sequence[str]) -> None:
    n_rows, n_cols = matrix.shape
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(
        [METHOD_LABELS.get(method, method) for method in methods], rotation=35, ha="right", rotation_mode="anchor"
    )
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(PLOT_REF_LABELS * len(SCENARIOS))
    ax.tick_params(axis="y", length=0, pad=3)
    ax.set_ylabel("Reference")
    for tick_label in ax.get_yticklabels():
        tick_label.set_clip_on(False)
        if tick_label.get_text() == "Mean":
            tick_label.set_fontweight("bold")
    rows_per_dataset = len(PLOT_REF_LABELS)
    for dataset_idx in range(len(SCENARIOS)):
        mean_boundary = dataset_idx * rows_per_dataset + len(REF_LABELS) - 0.5
        ax.axhline(mean_boundary, color="#555555", linewidth=0.55, linestyle="--", zorder=4)
    for dataset_idx in range(1, len(SCENARIOS)):
        dataset_boundary = dataset_idx * rows_per_dataset - 0.5
        ax.axhline(dataset_boundary, color="black", linewidth=0.95, zorder=4)
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.3)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.65)
        spine.set_color("black")


def add_dataset_group_labels(ax: plt.Axes) -> None:
    rows_per_dataset = len(PLOT_REF_LABELS)
    for dataset_idx, scenario in enumerate(SCENARIOS):
        center_y = dataset_idx * rows_per_dataset + (rows_per_dataset - 1) / 2
        n_sample = scenario.get("n_sample", "NA")
        ax.text(
            -0.13,
            center_y - 0.42,
            scenario["label"],
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=6.8,
            fontweight="bold",
            clip_on=False,
        )
        ax.text(
            -0.13,
            center_y + 0.52,
            f"n={n_sample}",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=6.2,
            fontweight="normal",
            clip_on=False,
        )


def summarize_best_scenarios(
    matrix: pd.DataFrame, higher_is_better: bool
) -> Tuple[pd.Series, pd.Series, int]:
    counts = pd.Series(0, index=matrix.columns, dtype=int)
    evaluable_scenarios = 0
    for (_, ref_label), row in matrix.iterrows():
        if ref_label == "Mean":
            continue
        values = row.to_numpy(dtype=float)
        finite_mask = np.isfinite(values)
        evaluable_scenarios += 1
        best_value = np.nanmax(values) if higher_is_better else np.nanmin(values)
        best_mask = finite_mask & np.isclose(values, best_value, rtol=1e-09, atol=1e-12)
        for method, is_best in zip(matrix.columns, best_mask):
            if is_best:
                counts.loc[method] += 1
    percentages = counts.astype(float) / evaluable_scenarios * 100.0
    return (counts, percentages, evaluable_scenarios)


def plot_best_scenario_barplot(
    ax: plt.Axes, matrix: pd.DataFrame, methods: Sequence[str], higher_is_better: bool
) -> None:
    counts, percentages, n_scenarios = summarize_best_scenarios(
        matrix=matrix, higher_is_better=higher_is_better
    )
    y = np.arange(len(methods))
    widths = counts.reindex(methods).to_numpy(dtype=float)
    pct_values = percentages.reindex(methods).to_numpy(dtype=float)
    bars = ax.barh(y, widths, height=0.5, color="#8A8A8A", edgecolor="black", linewidth=0.55, zorder=3)
    max_width = float(np.max(widths))
    label_offset = max(0.18, max_width * 0.035)
    for bar, count, percentage in zip(bars, widths, pct_values):
        ax.text(
            count + label_offset,
            bar.get_y() + bar.get_height() / 2,
            f"{int(count)} ({percentage:.1f}%)",
            ha="left",
            va="center",
            fontsize=5.5,
            color="black",
            clip_on=False,
        )
    ax.set_yticks(y)
    ax.set_yticklabels([METHOD_LABELS.get(method, method) for method in methods], ha="right")
    ax.tick_params(axis="y", length=0, pad=2, labelsize=6.0)
    ax.invert_yaxis()
    ax.set_title(f"Best scenarios (n={n_scenarios})", fontsize=7.6, fontweight="bold", pad=4)
    ax.set_xlabel("Count", labelpad=2)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
    ax.set_xlim(0, max(1.0, max_width + label_offset * 6.2))
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.45, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.65)
    ax.spines["bottom"].set_linewidth(0.65)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")


def plot_metric_heatmap(
    data: pd.DataFrame,
    metric_slug: str,
    metric_label: str,
    colorbar_label: str,
    higher_is_better: bool,
    mode: str,
    mode_label: str,
    filename_prefix: str,
    methods: Sequence[str],
) -> None:
    matrix = build_matrix(data=data, metric_slug=metric_slug, mode=mode, methods=methods)
    table_prefix = f"{filename_prefix}_all_datasets_all_references_{metric_slug}"
    matrix.reset_index().to_csv(OUTPUT_DIR / f"{table_prefix}_matrix.csv", index=False)
    counts, percentages, n_scenarios = summarize_best_scenarios(
        matrix=matrix, higher_is_better=higher_is_better
    )
    pd.DataFrame(
        {
            "Method": list(methods),
            "MethodLabel": [METHOD_LABELS.get(method, method) for method in methods],
            "BestScenarioCount": counts.reindex(methods).to_numpy(dtype=int),
            "BestScenarioPercent": percentages.reindex(methods).to_numpy(dtype=float),
            "NEvaluableScenarios": n_scenarios,
            "Mode": mode,
            "MetricSlug": metric_slug,
        }
    ).to_csv(OUTPUT_DIR / f"{table_prefix}_best_scenarios.csv", index=False)
    if metric_slug.endswith("pearsonr"):
        cmap = sns.blend_palette(
            ["#5BA6C9", "#DCEAF1", "#F7F7F2", "#F6D8AE", "#E89A6A", "#C65353"],
            as_cmap=True
        )
        norm = Normalize(vmin=-0.1, vmax=0.75, clip=True)

    elif metric_slug.endswith("ccc"):
        cmap = sns.blend_palette(
            ["#5BA6C9", "#DCEAF1", "#F7F7F2", "#F6D8AE", "#E89A6A", "#C65353"],
            as_cmap=True
        )
        norm = Normalize(vmin=-0.1, vmax=0.5, clip=True)
    else:
        valid_values = matrix.to_numpy(dtype=float).ravel()
        valid_values = valid_values[np.isfinite(valid_values)]
        cmap = sns.blend_palette(["#3A8D6D", "#F6F4EF", "#C95F5F"], as_cmap=True)
        norm = Normalize(vmin=0.0, vmax=max(float(np.max(valid_values)), 1e-06))
    cmap.set_bad("#E5E5E5")
    fig = plt.figure(figsize=(9.4, 7.4))
    grid = fig.add_gridspec(nrows=1, ncols=3, width_ratios=[5.25, 0.075, 1.55], wspace=0.22)
    ax = fig.add_subplot(grid[0, 0])
    cax = fig.add_subplot(grid[0, 1])
    bar_ax = fig.add_subplot(grid[0, 2])
    image = ax.imshow(
        matrix.to_numpy(dtype=float), cmap=cmap, norm=norm, aspect="auto", interpolation="nearest"
    )
    style_heatmap_axis(ax, matrix, methods)
    add_dataset_group_labels(ax)
    annotate_heatmap_cells(ax=ax, matrix=matrix, higher_is_better=higher_is_better)
    colorbar = fig.colorbar(image, cax=cax, extend="neither")
    colorbar.ax.yaxis.set_ticks_position("left")
    colorbar.ax.yaxis.set_label_position("left")
    colorbar.set_label(colorbar_label, rotation=90, labelpad=5)
    colorbar.ax.tick_params(labelsize=6.5, pad=1.5)
    plot_best_scenario_barplot(
        ax=bar_ax, matrix=matrix, methods=methods, higher_is_better=higher_is_better
    )
    fig.suptitle(
        f"{mode_label}: {metric_label} across all datasets and references",
        y=0.976,
        fontsize=11.2,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.185, right=0.975, top=0.915, bottom=0.105)
    fig.savefig(OUTPUT_DIR / f"{filename_prefix}_all_datasets_all_references_{metric_slug}.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)


celltype_rows: List[Dict] = []
sample_rows: List[Dict] = []
celltype_detail_rows: List[Dict] = []
sample_detail_rows: List[Dict] = []
prediction_cache: Dict[Path, pd.DataFrame] = {}
for scenario in SCENARIOS:
    dataset = scenario["dataset"]
    true_path: Path = scenario["true_path"]
    results_dir: Path = scenario["results_dir"]
    true_raw = pd.read_csv(true_path, index_col=0)
    scenario["n_sample"] = int(true_raw.shape[0])
    for mode, method_order in (("benchmark", BENCHMARK_METHODS), ("ablation", ABLATION_METHODS)):
        for ref_code, ref_label in zip(["3k", "4k", "6k", "8k", "20k", "33k", ""], REF_LABELS):
            for base_method in method_order:
                if mode == "benchmark" or base_method == "SABER":
                    filename = f"{base_method}{ref_code}.csv"
                else:
                    suffix = base_method.replace("SABER_", "")
                    filename = f"SABER{ref_code}_{suffix}.csv" if ref_code else f"SABER_{suffix}.csv"
                pred_path = results_dir / filename
                if pred_path not in prediction_cache:
                    prediction_cache[pred_path] = pd.read_csv(pred_path, index_col=0)
                pred_raw = prediction_cache[pred_path]
                true_sub, pred_sub, common_samples, common_celltypes = common_true_pred(true_raw, pred_raw, scenario)
                metric_context = {
                    "Dataset": dataset,
                    "Scenario": scenario["label"],
                    "Mode": mode,
                    "Ref": ref_label,
                    "BaseMethod": base_method,
                }
                celltype_mae_values: List[float] = []
                celltype_pearson_values: List[float] = []
                celltype_ccc_values: List[float] = []
                for celltype in common_celltypes:
                    y_true = true_sub[celltype].to_numpy(dtype=float)
                    y_pred = pred_sub[celltype].to_numpy(dtype=float)
                    celltype_mae = float(np.mean(np.abs(y_true - y_pred)))
                    celltype_pearson = safe_pearson(y_true, y_pred)
                    celltype_ccc = compute_ccc(y_true, y_pred)
                    celltype_mae_values.append(celltype_mae)
                    celltype_pearson_values.append(celltype_pearson)
                    celltype_ccc_values.append(celltype_ccc)
                    celltype_detail_rows.append(
                        {
                            **metric_context,
                            "CellType": celltype,
                            "MAE": celltype_mae,
                            "PearsonR": celltype_pearson,
                            "CCC": celltype_ccc,
                            "NSamples": len(common_samples),
                        }
                    )
                for metric_slug, metric_values in [
                    ("celltype_mae", celltype_mae_values),
                    ("celltype_pearsonr", celltype_pearson_values),
                    ("celltype_ccc", celltype_ccc_values),
                ]:
                    (
                        metric_value,
                        valid_count,
                        total_count,
                        undefined_count,
                        valid_rate,
                    ) = summarize_metric_values(metric_values)
                    celltype_rows.append(
                        {
                            **metric_context,
                            "MetricSlug": metric_slug,
                            "Value": metric_value,
                            "ValidCount": valid_count,
                            "TotalCount": total_count,
                            "UndefinedCount": undefined_count,
                            "ValidRate": valid_rate,
                            "NCellTypes": len(common_celltypes),
                            "NSamples": len(common_samples),
                        }
                    )
                sample_mae_values: List[float] = []
                sample_pearson_values: List[float] = []
                sample_ccc_values: List[float] = []
                for sample in common_samples:
                    y_true = true_sub.loc[sample].to_numpy(dtype=float)
                    y_pred = pred_sub.loc[sample].to_numpy(dtype=float)
                    sample_mae = float(np.mean(np.abs(y_true - y_pred)))
                    sample_pearson = safe_pearson(y_true, y_pred)
                    sample_ccc = compute_ccc(y_true, y_pred)
                    sample_mae_values.append(sample_mae)
                    sample_pearson_values.append(sample_pearson)
                    sample_ccc_values.append(sample_ccc)
                    sample_detail_rows.append(
                        {
                            "Dataset": dataset,
                            "Scenario": scenario["label"],
                            "Mode": mode,
                            "Ref": ref_label,
                            "BaseMethod": base_method,
                            "Sample": sample,
                            "MAE": sample_mae,
                            "PearsonR": sample_pearson,
                            "CCC": sample_ccc,
                            "NCellTypes": len(common_celltypes),
                        }
                    )
                for metric_slug, metric_values in [
                    ("sample_mae", sample_mae_values),
                    ("sample_pearsonr", sample_pearson_values),
                    ("sample_ccc", sample_ccc_values),
                ]:
                    (
                        metric_value,
                        valid_count,
                        total_count,
                        undefined_count,
                        valid_rate,
                    ) = summarize_metric_values(metric_values)
                    sample_rows.append(
                        {
                            "Dataset": dataset,
                            "Scenario": scenario["label"],
                            "Mode": mode,
                            "Ref": ref_label,
                            "BaseMethod": base_method,
                            "MetricSlug": metric_slug,
                            "Value": metric_value,
                            "ValidCount": valid_count,
                            "TotalCount": total_count,
                            "UndefinedCount": undefined_count,
                            "ValidRate": valid_rate,
                            "NCellTypes": len(common_celltypes),
                            "NSamples": len(common_samples),
                        }
                    )
celltype_summary = pd.DataFrame(celltype_rows)
sample_summary = pd.DataFrame(sample_rows)
celltype_summary.to_csv(OUTPUT_DIR / "celltype_summary_long.csv", index=False)
sample_summary.to_csv(OUTPUT_DIR / "sample_summary_long.csv", index=False)
pd.DataFrame(celltype_detail_rows).to_csv(OUTPUT_DIR / "celltype_metrics_long.csv", index=False)
pd.DataFrame(sample_detail_rows).to_csv(OUTPUT_DIR / "sample_metrics_long.csv", index=False)
for mode, mode_label, filename_prefix, methods in [
    ("benchmark", "Benchmark methods", "benchmark", BENCHMARK_METHODS),
    ("ablation", "SABER ablation", "ablation", ABLATION_METHODS),
]:
    for data, metric_slug, metric_label, colorbar_label, higher_is_better in [
        (celltype_summary, "celltype_mae", "Per-cell-type MAE", "Mean per-cell-type MAE", False),
        (celltype_summary, "celltype_pearsonr", "Per-cell-type Pearson r", "Mean per-cell-type Pearson r", True),
        (celltype_summary, "celltype_ccc", "Per-cell-type CCC", "Mean per-cell-type CCC", True),
        (sample_summary, "sample_mae", "Per-sample MAE", "Mean per-sample MAE", False),
        (sample_summary, "sample_pearsonr", "Per-sample Pearson r", "Mean per-sample Pearson r", True),
        (sample_summary, "sample_ccc", "Per-sample CCC", "Mean per-sample CCC", True),
    ]:
        plot_metric_heatmap(
            data, metric_slug, metric_label, colorbar_label, higher_is_better,
            mode, mode_label, filename_prefix, methods,
        )

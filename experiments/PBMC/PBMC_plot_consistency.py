#!/usr/bin/env python3
from __future__ import annotations
import itertools
import warnings
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 6.5,
        "axes.linewidth": 0.55,
        "axes.labelsize": 6.5,
        "axes.titlesize": 7.5,
        "axes.titleweight": "bold",
        "xtick.labelsize": 5.2,
        "ytick.labelsize": 5.8,
        "legend.fontsize": 6,
        "figure.dpi": 300,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 2.2,
        "ytick.major.size": 2.2,
        "xtick.major.width": 0.55,
        "ytick.major.width": 0.55,
        "savefig.bbox": "tight",
    }
)
PROJECT_ROOT = Path("/Path/to/PBMC")
OUTPUT_DIR = Path("/Path/to/output/PBMC_consistency")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REF_LABELS = ["3k", "4k", "6k", "8k", "20k", "33k", "Combined"]
SCENARIOS: List[Dict] = [
    {
        "dataset": "monaco",
        "label": "RNA-seq: GSE107011",
        "platform": "RNAseq",
        "true_path": PROJECT_ROOT / "data" / "monaco" / "P_merged.csv",
        "results_dir": PROJECT_ROOT / "results" / "monaco",
        "preprocess": "none",
    },
    {
        "dataset": "monacoarray",
        "label": "Array: GSE106898",
        "platform": "Array",
        "true_path": PROJECT_ROOT / "data" / "monacoarray" / "P_merged.csv",
        "results_dir": PROJECT_ROOT / "results" / "monacoarray",
        "preprocess": "none",
    },
    {
        "dataset": "quantiseq",
        "label": "RNA-seq: GSE107572",
        "platform": "RNAseq",
        "true_path": PROJECT_ROOT / "data" / "quantiseq" / "proportions.csv",
        "results_dir": PROJECT_ROOT / "results" / "quantiseq",
        "preprocess": "none",
    },
    {
        "dataset": "sdy67",
        "label": "RNA-seq: SDY67",
        "platform": "RNAseq",
        "true_path": PROJECT_ROOT / "data" / "sdy67" / "proportions.csv",
        "results_dir": PROJECT_ROOT / "results" / "sdy67",
        "preprocess": "none",
    },
    {
        "dataset": "scper",
        "label": "Array: GSE107990",
        "platform": "Array",
        "true_path": PROJECT_ROOT / "data" / "scper" / "proportions.csv",
        "results_dir": PROJECT_ROOT / "results" / "scper",
        "preprocess": "none",
    },
    {
        "dataset": "shaw",
        "label": "Array: GSE59654",
        "platform": "Array",
        "true_path": PROJECT_ROOT / "data" / "shaw" / "proportions.csv",
        "results_dir": PROJECT_ROOT / "results" / "shaw",
        "preprocess": "keep_shaw_4_renorm",
    },
    {
        "dataset": "newman",
        "label": "Array: GSE65133",
        "platform": "Array",
        "true_path": PROJECT_ROOT / "data" / "newman" / "proportions.csv",
        "results_dir": PROJECT_ROOT / "results" / "newman",
        "preprocess": "drop_dendritics_renorm",
    },
    {
        "dataset": "morandini",
        "label": "RNA-seq: GSE193141",
        "platform": "RNAseq",
        "true_path": PROJECT_ROOT / "data" / "morandini" / "proportions.csv",
        "results_dir": PROJECT_ROOT / "results" / "morandini",
        "preprocess": "drop_dendritics_renorm",
    },
    {
        "dataset": "harrison",
        "label": "RNA-seq: GSE120502",
        "platform": "RNAseq",
        "true_path": PROJECT_ROOT / "data" / "harrison" / "proportions.csv",
        "results_dir": PROJECT_ROOT / "results" / "harrison",
        "preprocess": "drop_dendritics_renorm",
    },
]
MODE_CONFIG = {
    "benchmark": {
        "methods": ["SABER", "DWLS", "BayesPrism", "Scadenpytorch", "TAPE", "RNASieve", "NNLS"],
        "labels": {
            "SABER": "SABER", "DWLS": "DWLS", "BayesPrism": "BayesPrism",
            "Scadenpytorch": "Scaden", "TAPE": "TAPE", "RNASieve": "RNASieve", "NNLS": "NNLS",
        },
        "colors": {
            "SABER": "#C95F5F", "DWLS": "#A1C181", "BayesPrism": "#D98C80",
            "Scadenpytorch": "#6B9AC4", "TAPE": "#E0A458", "RNASieve": "#B48EAD", "NNLS": "#9A9A9A",
        },
        "title": "Benchmark methods",
        "reference_name": "SABER",
    },
    "ablation": {
        "methods": [
            "SABER", "SABER_wo_alignment", "SABER_wo_twoview", "SABER_wo_gene_selection",
            "SABER_w_gene_selection", "SABER_w_twoview", "SABER_w_alignment", "SABER_wo_all",
        ],
        "labels": {
            "SABER": "SABER", "SABER_w_gene_selection": "w_gene_selection",
            "SABER_wo_gene_selection": "wo_gene_selection", "SABER_w_twoview": "w_twoview",
            "SABER_wo_twoview": "wo_twoview", "SABER_w_alignment": "w_alignment",
            "SABER_wo_alignment": "wo_alignment", "SABER_wo_all": "wo_all",
        },
        "colors": {
            "SABER": "#C95F5F", "wo_alignment": "#A6DBA0", "wo_twoview": "#67A9CF",
            "wo_gene_selection": "#EF8A62", "w_gene_selection": "#B2182B", "w_twoview": "#2166AC",
            "w_alignment": "#4D9221", "wo_all": "#636363",
        },
        "title": "SABER ablation",
        "reference_name": "SABER",
    },
}


def renormalize_rows(df: pd.DataFrame) -> pd.DataFrame:
    row_sums = df.sum(axis=1)
    row_sums[row_sums == 0] = 1
    return df.div(row_sums, axis=0)


def safe_pearsonr_with_status(y_true: Sequence[float], y_pred: Sequence[float]) -> Tuple[float, str]:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    if y_true_arr.size < 2:
        return (np.nan, "Insufficient_n")
    std_true = float(np.std(y_true_arr, ddof=1))
    std_pred = float(np.std(y_pred_arr, ddof=1))
    constant_true = np.isclose(std_true, 0.0, atol=1e-15, rtol=0.0)
    constant_pred = np.isclose(std_pred, 0.0, atol=1e-15, rtol=0.0)
    if constant_true and constant_pred:
        return (np.nan, "Constant_both")
    if constant_true:
        return (np.nan, "Constant_refA")
    if constant_pred:
        return (np.nan, "Constant_refB")
    r_val, _ = pearsonr(y_true_arr, y_pred_arr)
    return (float(r_val), "Valid")


def lin_ccc_with_status(y_true: Sequence[float], y_pred: Sequence[float]) -> Tuple[float, str]:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    if y_true_arr.size < 2:
        return (np.nan, "Insufficient_n")
    mean_true = float(np.mean(y_true_arr))
    mean_pred = float(np.mean(y_pred_arr))
    var_true = float(np.var(y_true_arr, ddof=0))
    var_pred = float(np.var(y_pred_arr, ddof=0))
    covariance = float(np.mean((y_true_arr - mean_true) * (y_pred_arr - mean_pred)))
    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
    if np.isclose(denominator, 0.0, atol=1e-15, rtol=0.0):
        return (np.nan, "Zero_denominator")
    constant_true = np.isclose(var_true, 0.0, atol=1e-15, rtol=0.0)
    constant_pred = np.isclose(var_pred, 0.0, atol=1e-15, rtol=0.0)
    if constant_true or constant_pred:
        return (0.0, "Valid")
    return (float(2.0 * covariance / denominator), "Valid")


def collect_consistency(mode: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = MODE_CONFIG[mode]
    methods = cfg["methods"]
    labels = cfg["labels"]
    celltype_records: List[Dict] = []
    sample_records: List[Dict] = []
    qc_records: List[Dict] = []
    for scenario in SCENARIOS:
        dataset = scenario["dataset"]
        platform = scenario["platform"]
        true_df = pd.read_csv(scenario["true_path"], index_col=0).copy()
        true_df.index = true_df.index.astype(str)
        required_samples = true_df.index.tolist()
        required_celltypes = list(true_df.columns)
        for base_method in methods:
            method_preds: Dict[str, pd.DataFrame] = {}
            for ref_code, ref_label in zip(["3k", "4k", "6k", "8k", "20k", "33k", ""], REF_LABELS):
                if mode == "benchmark" or base_method == "SABER":
                    filename = f"{base_method}{ref_code}.csv"
                else:
                    suffix = base_method.replace("SABER_", "")
                    filename = f"SABER{ref_code}_{suffix}.csv" if ref_code else f"SABER_{suffix}.csv"
                pred_df = pd.read_csv(scenario["results_dir"] / filename, index_col=0).copy()
                pred_df.index = pred_df.index.astype(str)
                if scenario.get("preprocess", "none") == "drop_dendritics_renorm":
                    pred_df = renormalize_rows(
                        pred_df.drop(columns=["Dendritics"]).apply(pd.to_numeric, errors="coerce")
                    )
                elif scenario.get("preprocess", "none") == "keep_shaw_4_renorm":
                    pred_df = renormalize_rows(
                        pred_df[required_celltypes].apply(pd.to_numeric, errors="coerce")
                    )
                pred_df = pred_df.loc[required_samples, required_celltypes].copy()
                pred_df = pred_df.apply(pd.to_numeric, errors="coerce")
                method_preds[ref_label] = pred_df
                qc_records.append(
                    {
                        "Dataset": dataset,
                        "Mode": mode,
                        "Method": base_method,
                        "Ref": ref_label,
                        "Status": "Passed",
                        "Reason": "",
                    }
                )
            for ref_a, ref_b in itertools.combinations(REF_LABELS, 2):
                df_a = method_preds[ref_a]
                df_b = method_preds[ref_b]
                for celltype in required_celltypes:
                    values_a = df_a[celltype].to_numpy(dtype=float)
                    values_b = df_b[celltype].to_numpy(dtype=float)
                    pearson_r, pearson_status = safe_pearsonr_with_status(values_a, values_b)
                    ccc, ccc_status = lin_ccc_with_status(values_a, values_b)
                    celltype_records.append(
                        {
                            "Evaluation": "PerCellType",
                            "Mode": mode,
                            "Method": base_method,
                            "Method_Label": labels[base_method],
                            "Platform": platform,
                            "Dataset": dataset,
                            "Scenario": scenario["label"],
                            "Ref_A": ref_a,
                            "Ref_B": ref_b,
                            "CellType": celltype,
                            "N_Samples": len(required_samples),
                            "PearsonR": pearson_r,
                            "Pearson_Status": pearson_status,
                            "CCC": ccc,
                            "CCC_Status": ccc_status,
                        }
                    )
                for sample in required_samples:
                    values_a = df_a.loc[sample, required_celltypes].to_numpy(dtype=float)
                    values_b = df_b.loc[sample, required_celltypes].to_numpy(dtype=float)
                    pearson_r, pearson_status = safe_pearsonr_with_status(values_a, values_b)
                    ccc, ccc_status = lin_ccc_with_status(values_a, values_b)
                    sample_records.append(
                        {
                            "Evaluation": "PerSample",
                            "Mode": mode,
                            "Method": base_method,
                            "Method_Label": labels[base_method],
                            "Platform": platform,
                            "Dataset": dataset,
                            "Scenario": scenario["label"],
                            "Ref_A": ref_a,
                            "Ref_B": ref_b,
                            "Sample": sample,
                            "N_CellTypes": len(required_celltypes),
                            "PearsonR": pearson_r,
                            "Pearson_Status": pearson_status,
                            "CCC": ccc,
                            "CCC_Status": ccc_status,
                        }
                    )
    return (pd.DataFrame(celltype_records), pd.DataFrame(sample_records), pd.DataFrame(qc_records))


def summarize_by_method(df: pd.DataFrame, mode: str, evaluation_name: str) -> pd.DataFrame:
    methods = MODE_CONFIG[mode]["methods"]
    labels = MODE_CONFIG[mode]["labels"]
    records: List[Dict] = []
    for method in methods:
        method_df = df[df["Method"] == method].copy()
        for metric, status_col in [("PearsonR", "Pearson_Status"), ("CCC", "CCC_Status")]:
            total_count = len(method_df)
            valid_mask = (method_df[status_col] == "Valid") & np.isfinite(
                pd.to_numeric(method_df[metric], errors="coerce")
            )
            values = pd.to_numeric(method_df.loc[valid_mask, metric], errors="coerce").dropna()
            valid_count = len(values)
            undefined_count = total_count - valid_count
            if valid_count == 0:
                mean_val = np.nan
                sd_val = np.nan
                sem_val = np.nan
            elif valid_count == 1:
                mean_val = float(values.mean())
                sd_val = 0.0
                sem_val = 0.0
            else:
                mean_val = float(values.mean())
                sd_val = float(values.std(ddof=1))
                sem_val = float(sd_val / np.sqrt(valid_count))
            status_counts = method_df[status_col].value_counts()
            records.append(
                {
                    "Evaluation": evaluation_name,
                    "Mode": mode,
                    "Method": method,
                    "Method_Label": labels[method],
                    "Metric": metric,
                    "Mean": mean_val,
                    "SD": sd_val,
                    "SEM": sem_val,
                    "Valid_Count": valid_count,
                    "Total_Count": total_count,
                    "Undefined_Count": undefined_count,
                    "Valid_Rate": valid_count / total_count,
                    "Undefined_Rate": undefined_count / total_count,
                    "Constant_refA_Count": int(status_counts.get("Constant_refA", 0)),
                    "Constant_refB_Count": int(status_counts.get("Constant_refB", 0)),
                    "Constant_both_Count": int(status_counts.get("Constant_both", 0)),
                    "Insufficient_n_Count": int(status_counts.get("Insufficient_n", 0)),
                    "Zero_denominator_Count": int(status_counts.get("Zero_denominator", 0)),
                }
            )
    return pd.DataFrame(records)


def plot_consistency_summary(summary_df: pd.DataFrame, mode: str, title: str, output_prefix: str) -> None:
    cfg = MODE_CONFIG[mode]
    labels = cfg["labels"]
    colors = cfg["colors"]
    available_methods = [method for method in cfg["methods"] if method in summary_df["Method"].unique()]
    x = np.arange(len(available_methods)) * 0.58
    if mode == "ablation":
        bar_colors = [colors[labels[method]] for method in available_methods]
        fig_width = max(4.13, 0.2 * len(available_methods) * 2)
        fig_height = 2.15
    else:
        bar_colors = [colors[method] for method in available_methods]
        fig_width = max(3.87, 0.18 * len(available_methods) * 2)
        fig_height = 2.05
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), gridspec_kw={"wspace": 0.32})
    for ax, (metric, panel_title, y_label) in zip(
        axes, [("CCC", "CCC", "Mean CCC"), ("PearsonR", "Pearson r", "Mean Pearson r")]
    ):
        panel_df = summary_df[summary_df["Metric"] == metric].set_index("Method").reindex(available_methods)
        means = panel_df["Mean"].astype(float).values
        sems = panel_df["SEM"].astype(float).fillna(0.0).values
        ax.bar(
            x,
            means,
            yerr=sems,
            color=bar_colors,
            edgecolor="black",
            linewidth=0.35,
            width=0.3,
            error_kw={"elinewidth": 0.55, "ecolor": "black", "capsize": 1.5, "capthick": 0.55},
        )
        ax.set_title(panel_title, pad=3)
        ax.set_ylabel(y_label, labelpad=1.5)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [labels[method] for method in available_methods],
            rotation=55,
            ha="right",
            rotation_mode="anchor",
        )
        ax.set_xlim(x[0] - 0.32, x[-1] + 0.32)
        observed_max = float(np.max((means + sems)[np.isfinite(means + sems)]))
        y_max = max(observed_max * 1.1, observed_max + 0.02, 0.1)
        ax.set_ylim(0.0, y_max)
    fig.suptitle(title, fontsize=8.2, fontweight="bold", y=1.03)
    for ax in fig.axes:
        ax.tick_params(axis="both", which="both", direction="out", length=2.2, width=0.55, pad=1.0)
        ax.spines["left"].set_linewidth(0.55)
        ax.spines["bottom"].set_linewidth(0.55)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    plt.tight_layout(pad=0.35, w_pad=0.3)
    fig.savefig(OUTPUT_DIR / f"{output_prefix}.pdf", bbox_inches="tight")
    plt.close(fig)


def process_mode_platform(
    mode: str, platform_key: str, platform_label: str, df_celltype: pd.DataFrame, df_sample: pd.DataFrame
) -> None:
    platform_celltype = df_celltype[df_celltype["Platform"] == platform_key].copy()
    platform_sample = df_sample[df_sample["Platform"] == platform_key].copy()
    summary_celltype = summarize_by_method(platform_celltype, mode, f"{platform_label}_PerCellType")
    summary_sample = summarize_by_method(platform_sample, mode, f"{platform_label}_PerSample")
    prefix_base = f"PBMC_{mode}_{platform_key.lower()}"
    platform_celltype.to_csv(OUTPUT_DIR / f"{prefix_base}_per_celltype_consistency_raw.csv", index=False)
    summary_celltype.to_csv(OUTPUT_DIR / f"{prefix_base}_per_celltype_consistency_summary.csv", index=False)
    plot_consistency_summary(
        summary_df=summary_celltype,
        mode=mode,
        title=f"{platform_label}: cross-reference consistency per cell type",
        output_prefix=f"{prefix_base}_per_celltype_consistency_barplot",
    )
    platform_sample.to_csv(OUTPUT_DIR / f"{prefix_base}_per_sample_consistency_raw.csv", index=False)
    summary_sample.to_csv(OUTPUT_DIR / f"{prefix_base}_per_sample_consistency_summary.csv", index=False)
    plot_consistency_summary(
        summary_df=summary_sample,
        mode=mode,
        title=f"{platform_label}: cross-reference consistency per sample",
        output_prefix=f"{prefix_base}_per_sample_consistency_barplot",
    )


for mode in ["benchmark", "ablation"]:
    df_celltype, df_sample, qc_df = collect_consistency(mode)
    qc_df.to_csv(OUTPUT_DIR / f"PBMC_{mode}_cross_reference_consistency_QC.csv", index=False)
    for platform_key, platform_label in [("RNAseq", "RNA-seq"), ("Array", "Array")]:
        process_mode_platform(
            mode=mode,
            platform_key=platform_key,
            platform_label=platform_label,
            df_celltype=df_celltype,
            df_sample=df_sample,
        )

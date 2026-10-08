# Reproducing the analyses

The files in this directory contain the analysis code used to summarize the SABER benchmark results.

| Directory | File | Analysis |
| --- | --- | --- |
| `PBMC/` | `PBMC_heatmap.py` | Generates performance heatmaps across PBMC benchmark datasets and single-cell references. |
| `PBMC/` | `PBMC_plot_consistency.py` | Evaluates cross-dataset and cross-reference consistency of PBMC predictions. |
| `PBMC/` | `PBMC_monaco_cross_platform_consistency.py` | Evaluates matched Monaco RNA-seq–microarray prediction consistency. |
| `pancreas/` | `Pancreas_analysis.ipynb` | Analyzes HbA1c–beta-cell associations, ND–T2D differences and cross-reference consistency. |
| `IPF/` | `IPF_analysis.ipynb` | Analyzes control–IPF differences and cross-reference consistency. |
| `cancer/` | `BRCA_analysis.ipynb` | Analyzes BRCA predictions across PAM50 and immune subtypes. |
| `cancer/` | `CRC_analysis.ipynb` | Analyzes CRC predictions across consensus molecular subtypes. |
| `cancer/` | `LUAD_analysis.ipynb` | Analyzes LUAD predictions across molecular subtypes. |
| `cancer/` | `OV_analysis.ipynb` | Analyzes OV predictions across gene-expression subtypes. |
| `simulation_IPF/` | `IPF_reference_robustness.ipynb` | Evaluates cross-reference robustness in the directed IPF pseudo-bulk simulations and SABER ablations. |

# General method execution

methods/ provides general entry points for SABER, BayesPrism, BisqueRNA, DWLS, MuSiC, NNLS, RNA-Sieve, SCDC, Scaden and TAPE, which can be applied across different tissues and datasets.
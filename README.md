# SABER: stability-aware bulk RNA-seq deconvolution

![SABER](https://img.shields.io/badge/SABER-v0.1.0-blue) ![License](https://img.shields.io/badge/license-MIT-green)

SABER is a single-cell-reference-guided framework for estimating cell-type proportions from bulk RNA-seq data under source–target distribution shifts.

The associated manuscript is currently in preparation.

## Tutorial

### Installation

Clone this repository and create the provided base Conda environment:

```bash
git clone https://github.com/ciweisuonike/SABER.git
cd SABER
conda env create -f environment.yml
conda activate saber
```

PyTorch is installed separately so that you can select the build appropriate for your hardware. SABER was developed with PyTorch 2.6.0. Install one of the following options.

For CPU execution on Linux or Windows:

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
```

For CPU execution on macOS:

```bash
pip install torch==2.6.0
```

For NVIDIA GPU execution, install the wheel matching your CUDA environment. PyTorch 2.6.0 provides CUDA 11.8, 12.4 and 12.6 builds; for example, for CUDA 12.4:

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

Replace `cu124` with `cu118` or `cu126` when appropriate. Then install SABER:

```bash
pip install .
```

### Running

SABER requires a single-cell reference, a target bulk expression matrix and a YAML configuration file.

| Input | Format | Required structure |
| --- | --- | --- |
| Single-cell reference | AnnData `.h5ad` | Cells × genes in `adata.X`; cell annotations in `adata.obs` |
| Target bulk expression | `.csv` | Samples × genes; the first column contains sample identifiers |
| Configuration | `.yaml` | SABER simulation, model and training settings |

**`adata.X` must contain raw counts. Do not provide log1p-transformed, normalized or otherwise scaled expression values.**

For example, `adata.X` represents a raw-count matrix such as:

| Cell | GeneA | GeneB | GeneC |
| --- | ---: | ---: | ---: |
| Cell1 | 3 | 0 | 7 |
| Cell2 | 0 | 4 | 2 |
| Cell3 | 6 | 1 | 0 |

`adata.obs` must contain a `celltype` column. A `reference` column is optional:

| Cell | celltype | reference (optional) |
| --- | --- | --- |
| Cell1 | T cell | dataset1 |
| Cell2 | B cell | dataset1 |
| Cell3 | T cell | dataset2 |

The target bulk CSV follows this layout:

| Sample | GeneA | GeneB | GeneC |
| --- | ---: | ---: | ---: |
| Sample1 | 125.0 | 48.0 | 230.0 |
| Sample2 | 97.0 | 65.0 | 184.0 |

The `celltype` column is required. An optional `reference` column can be provided to indicate the source dataset or reference cohort from which each cell was derived. It is recommended that this column represent dataset-level reference identity rather than individual samples or donors. When the single-cell reference consists of a single dataset, the `reference` column can be omitted. Gene identifiers must be consistent between the single-cell reference and the target bulk expression matrix.

#### Train and predict directly

`saber prop` trains SABER using the single-cell reference and target bulk data, then predicts cell-type proportions for the input bulk samples:

```bash
saber prop \
  --adata /path/to/reference.h5ad \
  --bulk /path/to/bulk.csv \
  --config configs/config.yaml \
  --prop-dir output/proportions \
  --model-dir output/model \
  --device cpu
```

The predicted proportions are written to `output/proportions/proportions.csv`. When `--model-dir` is provided, the trained model is saved as `output/model/model.pt`.

#### Predict with a saved model

`saber prop-predict` loads a previously trained model and predicts proportions for another bulk expression matrix:

```bash
saber prop-predict \
  --model output/model/model.pt \
  --bulk /path/to/new_bulk.csv \
  --prop-dir output/new_predictions \
  --device cpu
```

The new bulk matrix must use gene identifiers compatible with those stored in the trained model.

## Example

The `examples/` directory provides a compact, directly runnable pancreas example:

| File | Provenance and preparation |
| --- | --- |
| `GSE84133_subset.h5ad` | A 500-cell subset of the human pancreas single-cell reference from [GSE84133](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE84133).`adata.X` contains raw counts. |
| `GSE50244_expression.csv` | The sample-by-gene raw-count matrix from [GSE50244](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE50244). |
| `GSE50244_metadata.csv` | GEO metadata for GSE50244.|

Open `examples/analysis.ipynb` to run SABER and plot the association between the predicted beta-cell proportion and HbA1c.

## Experiments

The analysis scripts used for the PBMC, pancreatic-islet, IPF, cancer and cross-reference simulation experiments are provided in `experiments/`. See `experiments/README.md` for their required inputs, path configuration and execution order.

## Issues

If you encounter a problem when using SABER, please open an issue in this repository.

## Citation

Software citation metadata is provided in `CITATION.cff`. The manuscript citation will be added when the associated article becomes available.

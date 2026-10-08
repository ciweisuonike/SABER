#!/usr/bin/env bash
conda activate saber
mkdir -p results

saber prop \
  --adata GSE84133_subset.h5ad \
  --bulk GSE50244_expression.csv \
  --config config.yaml \
  --prop-dir results \
  --prefix GSE50244_SABER \
  --device "cpu" # or cuda
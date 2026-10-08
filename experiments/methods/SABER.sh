#!/bin/bash
saber prop \
  --adata "path to single-cell reference.h5ad" \
  --config "path to configuration.yaml" \
  --bulk "path to bulk expression.csv" \
  --prop-dir "path to proportions output directory" \
  --device "cuda" \
  --prefix "SABER"

echo "Done."

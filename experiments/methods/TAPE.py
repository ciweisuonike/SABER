from TAPE import Deconvolution
import numpy as np
import scanpy as sc
import pandas as pd
from pathlib import Path

BULK_FILE = "path to bulk expression.tsv"
OUTPUT_FILE = "path to TAPE proportions.csv"

adata = sc.read_h5ad("path to single-cell reference.h5ad")
X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
sc_ref = pd.DataFrame(X, index=adata.obs["celltype"].astype(str).values, columns=adata.var.index.astype(str))
bulkdata = pd.read_csv(BULK_FILE, sep="	", index_col=0)
SignatureMatrix, CellFractionPrediction = Deconvolution(
    sc_ref,
    bulkdata,
    sep="	",
    scaler="mms",
    datatype="counts",
    genelenfile="path to gene lengths.txt",
    mode="overall",
    adaptive=True,
    variance_threshold=0.98,
    save_model_name="path to TAPE model prefix",
    batch_size=128,
    epochs=128,
    seed=1,
)
CellFractionPrediction.to_csv(OUTPUT_FILE, sep=",", index=True)

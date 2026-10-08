from TAPE.deconvolution import ScadenDeconvolution
import numpy as np
import scanpy as sc
import pandas as pd

adata = sc.read_h5ad("path to single-cell reference.h5ad")
X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
sc_ref = pd.DataFrame(X, index=adata.obs["celltype"].astype(str).values, columns=adata.var.index.astype(str))
bulkdata = pd.read_csv("path to bulk expression.tsv", sep="	", index_col=0)
Pred = ScadenDeconvolution(sc_ref, bulkdata, sep="	", batch_size=128, epochs=128)
Pred.to_csv("path to Scaden proportions.csv", index=True)

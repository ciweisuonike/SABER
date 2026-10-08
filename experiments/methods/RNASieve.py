from collections import defaultdict
import numpy as np
import pandas as pd
from rnasieve.preprocessing import model_from_raw_counts
import scanpy as sc
import scipy.sparse as sparse

print("Loading data...")
bulk = pd.read_csv("path to bulk expression.csv", index_col=0)
sc_adata = sc.read_h5ad("path to single-cell reference.h5ad")

genes = sc_adata.var_names.intersection(bulk.index)
sc2 = sc_adata[:, genes].copy()
bulk2 = bulk.loc[genes, :]
scX = sc2.X

print("Preparing data...")
counts_by_celltype = {}
for ct in sc2.obs["celltype"].unique():
    idx = np.where(sc2.obs["celltype"].values == ct)[0]
    Xct = scX[idx, :]
    if sparse.issparse(Xct):
        Xct = Xct.toarray()
    counts_by_celltype[ct] = Xct.T.astype(np.float32)

psis = bulk2.to_numpy(dtype=np.float32)
G = len(genes)
assert psis.shape[0] == G, "Gene count mismatch"
for ct, Xct in counts_by_celltype.items():
    assert Xct.shape[0] == G, f"Gene count mismatch for cell type {ct}"
    print(f"{ct}: {Xct.shape}")

print("Training model...")
model, cleaned_psis = model_from_raw_counts(counts_by_celltype, psis)
print("Predicting proportions...")
proportions = model.predict(cleaned_psis)
proportions.index = bulk2.columns
proportions.to_csv("path to RNASieve proportions.csv")
print("Done.")

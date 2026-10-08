library(Matrix)
library(nnls)
library(SingleCellExperiment)

cat("Reading single-cell data...
")
counts <- readMM("path to single-cell counts.mtx")
cell_metadata <- read.csv("path to single-cell metadata.csv", row.names = 1)
gene_metadata <- read.csv("path to single-cell genes.csv", row.names = 1)
rownames(counts) <- rownames(gene_metadata)
colnames(counts) <- rownames(cell_metadata)

if (!"celltype" %in% colnames(cell_metadata)) stop("metadata must contain celltype")
celltype <- cell_metadata$celltype

cat("Normalizing scRNA-seq counts to CPM...
")
lib_sc <- Matrix::colSums(counts)
if (any(lib_sc == 0)) stop("Some single cells have zero library size")
counts_cpm <- counts %*% Diagonal(x = 1 / lib_sc) * 1e6

cat("Building cell-type reference matrix...
")
celltypes <- unique(celltype)
ref_mat <- sapply(celltypes, function(ct) {
  idx <- which(celltype == ct)
  if (length(idx) == 1) counts_cpm[, idx] else Matrix::rowMeans(counts_cpm[, idx, drop = FALSE])
})
ref_mat <- as.matrix(ref_mat)
colnames(ref_mat) <- celltypes

cat("Reading bulk RNA-seq data...
")
bulk.mtx <- as.matrix(read.csv("path to bulk expression.csv", row.names = 1))
lib_bulk <- colSums(bulk.mtx)
if (any(lib_bulk == 0)) stop("Some bulk samples have zero library size")
bulk_cpm <- sweep(bulk.mtx, 2, lib_bulk, "/") * 1e6

common_genes <- intersect(rownames(ref_mat), rownames(bulk_cpm))
if (length(common_genes) < 100) stop("Too few overlapping genes between scRNA and bulk")
ref_use <- ref_mat[common_genes, , drop = FALSE]
bulk_use <- bulk_cpm[common_genes, , drop = FALSE]

cat("Running NNLS deconvolution...
")
nnls_res <- apply(bulk_use, 2, function(y) {
  fit <- nnls(ref_use, y)
  p <- fit$x
  p[p < 0] <- 0
  if (sum(p) == 0) rep(0, length(p)) else p / sum(p)
})
nnls_res <- t(nnls_res)
colnames(nnls_res) <- colnames(ref_use)
rownames(nnls_res) <- colnames(bulk_use)
write.csv(nnls_res, file="path to NNLS proportions.csv")
cat("Done.
")

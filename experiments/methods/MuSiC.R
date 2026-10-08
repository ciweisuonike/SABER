BULK_FILE <- "path to bulk expression.csv"
OUTPUT_FILE <- "path to MuSiC proportions.csv"

library(SingleCellExperiment)
library(Matrix)
library(MuSiC)

print("Reading data...")
counts <- readMM("path to single-cell counts.mtx")
cell_metadata <- read.csv("path to single-cell metadata.csv", row.names=1)
gene_metadata <- read.csv("path to single-cell genes.csv", row.names=1)
rownames(counts) <- rownames(gene_metadata)
colnames(counts) <- rownames(cell_metadata)

sce <- SingleCellExperiment(assays = list(counts = counts), colData = cell_metadata, rowData = gene_metadata)

print("Reading bulk data...")
bulk.mtx <- as.matrix(read.csv(BULK_FILE, row.names=1))
print(dim(bulk.mtx))

print("Running MuSiC...")
res <- music_prop(bulk.mtx, sce, clusters="celltype", samples="sample", verbose=TRUE)
cell_fractions <- as.data.frame(res$Est.prop.weighted)
write.csv(cell_fractions, file=OUTPUT_FILE)

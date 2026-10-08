BULK_FILE <- "path to bulk expression.csv"
OUTPUT_FILE <- "path to SCDC proportions.csv"

library(SCDC)
library(Matrix)

print("prepare scrna")
fdata <- read.csv("path to single-cell genes.csv", row.names=1)
pdata <- read.csv("path to single-cell metadata.csv", row.names=1)
sc_counts <- readMM("path to single-cell counts.mtx")
colnames(sc_counts) <- rownames(pdata)
rownames(sc_counts) <- rownames(fdata)

sc_eset <- getESET(sc_counts, pdata = pdata, fdata = fdata)

print("prepare bulk")
bulk_counts <- as.matrix(read.csv(BULK_FILE, row.names=1))
common_genes <- intersect(rownames(sc_counts), rownames(bulk_counts))
sc_eset <- sc_eset[common_genes, ]
bulk_counts <- bulk_counts[common_genes, ]

bulk_pdata <- data.frame(sample = colnames(bulk_counts), row.names = colnames(bulk_counts))
bulk_fdata <- data.frame(gene = rownames(bulk_counts), row.names = rownames(bulk_counts))
bulk_eset <- getESET(bulk_counts, pdata = bulk_pdata, fdata = bulk_fdata)

print("Run SCDC")
actual_ct <- unique(pdata$celltype)
res <- SCDC_prop(bulk.eset = bulk_eset, sc.eset = sc_eset,
                 ct.varname = "celltype", sample = "sample", ct.sub = actual_ct)
write.csv(res$prop.est.mvw, file=OUTPUT_FILE)

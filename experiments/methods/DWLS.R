BULK_FILE <- "path to bulk expression.csv"
OUTPUT_FILE <- "path to DWLS proportions.csv"
suppressPackageStartupMessages({
  library(Matrix)
})

source("path to DWLS helper functions.R")

COUNTS_FILE <- "path to single-cell counts.mtx"
METADATA_FILE <- "path to single-cell metadata.csv"
GENES_FILE <- "path to single-cell genes.csv"

cat("Reading single-cell data...
")
dataSC <- readMM(COUNTS_FILE)
metadata <- read.csv(METADATA_FILE, row.names = 1, check.names = FALSE)
genes <- read.csv(GENES_FILE, row.names = 1, check.names = FALSE)
rownames(dataSC) <- rownames(genes)
colnames(dataSC) <- rownames(metadata)
if (!"celltype" %in% colnames(metadata)) stop("metadata must contain celltype")

labels <- as.character(metadata$celltype)
if (length(labels) != ncol(dataSC)) stop("labels length != ncol(dataSC)")

downsampled <- stratifiedDownsampleCells(dataSC = dataSC, labels = labels,
                                         metadata = metadata, max_cells = 5000, seed = 1)
dataSC <- as.matrix(downsampled$data)
labels <- downsampled$labels
metadata <- downsampled$metadata
colnames(dataSC) <- rownames(metadata)

cat("Building signature matrix using MAST...
")
dwls_tmp <- tempfile("DWLS_")
dir.create(dwls_tmp, recursive = TRUE, showWarnings = FALSE)
on.exit(unlink(dwls_tmp, recursive = TRUE, force = TRUE), add = TRUE)
Signature <- buildSignatureMatrixMAST(scdata = dataSC, id = labels, path = dwls_tmp,
                                      diff.cutoff = 0.5, pval.cutoff = 0.01)

run_dwls <- function(bulk_file, out_file) {
  dataBulk <- as.matrix(read.csv(bulk_file, row.names = 1, check.names = FALSE))
  common_genes <- intersect(rownames(Signature), rownames(dataBulk))
  if (length(common_genes) == 0) stop("No overlapping genes between Signature and bulk")
  samples <- colnames(dataBulk)
  res_list <- lapply(samples, function(sample_name) {
    bulk_vec <- dataBulk[, sample_name]
    names(bulk_vec) <- rownames(dataBulk)
    tr <- trimData(Signature, bulk_vec)
    solveDampenedWLSWithOSQPFallback(tr$sig, tr$bulk, sample_name)
  })
  solDWLS <- as.data.frame(do.call(rbind, res_list))
  rownames(solDWLS) <- samples
  dir.create(dirname(out_file), recursive = TRUE, showWarnings = FALSE)
  writeOSQPFallbackReport(res_list, samples, out_file)
  write.csv(solDWLS, file = out_file)
}

run_dwls(BULK_FILE, OUTPUT_FILE)

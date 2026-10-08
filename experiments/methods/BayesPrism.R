BULK_FILE <- "path to bulk expression.csv"
OUTPUT_FILE <- "path to BayesPrism proportions.csv"

suppressPackageStartupMessages({
  library(BayesPrism)
  library(Matrix)
})

COUNTS_FILE <- "path to single-cell counts.mtx"
METADATA_FILE <- "path to single-cell metadata.csv"
GENES_FILE <- "path to single-cell genes.csv"
N_CORES <- 50

cat("Reading single-cell data...
")
counts <- readMM(COUNTS_FILE)
cell_metadata <- read.csv(METADATA_FILE, row.names = 1, check.names = FALSE)
gene_metadata <- read.csv(GENES_FILE, row.names = 1, check.names = FALSE)
rownames(counts) <- rownames(gene_metadata)
colnames(counts) <- rownames(cell_metadata)
if (!"celltype" %in% colnames(cell_metadata)) stop("metadata must contain celltype")

cell.type.labels <- as.character(cell_metadata$celltype)
cell.state.labels <- cell.type.labels
sc.dat <- as.matrix(t(counts))
rownames(sc.dat) <- colnames(counts)
colnames(sc.dat) <- rownames(counts)

run_bayesprism <- function(bulk_file, out_file) {
  bulk.mtx <- as.matrix(read.csv(bulk_file, row.names = 1, check.names = FALSE))
  bk.dat <- t(bulk.mtx)
  rownames(bk.dat) <- colnames(bulk.mtx)
  colnames(bk.dat) <- rownames(bulk.mtx)
  common_genes <- intersect(colnames(sc.dat), colnames(bk.dat))
  if (length(common_genes) < 100) stop("Too few overlapping genes between scRNA and bulk")
  sc.use <- sc.dat[, common_genes, drop = FALSE]
  bk.use <- bk.dat[, common_genes, drop = FALSE]
  if (any(sc.use < 0) || any(bk.use < 0)) stop("BayesPrism input matrices must be non-negative")
  keep_cells <- rowSums(sc.use) > 0
  sc.use <- sc.use[keep_cells, , drop = FALSE]
  cell.type.use <- cell.type.labels[keep_cells]
  cell.state.use <- cell.state.labels[keep_cells]
  keep_samples <- rowSums(bk.use) > 0
  bk.use <- bk.use[keep_samples, , drop = FALSE]
  sc.dat.filtered <- cleanup.genes(input = sc.use, input.type = "count.matrix", species = "hs",
                                   gene.group = c("Rb", "Mrp", "other_Rb", "chrM", "MALAT1", "chrX", "chrY"),
                                   exp.cells = 5)
  sc.dat.filtered.pc <- select.gene.type(sc.dat.filtered, gene.type = "protein_coding")
  genes.use <- intersect(colnames(sc.dat.filtered.pc), colnames(bk.use))
  if (length(genes.use) < 100) stop("Too few overlapping genes after BayesPrism gene filtering")
  sc.dat.filtered.pc <- sc.dat.filtered.pc[, genes.use, drop = FALSE]
  bk.use <- bk.use[, genes.use, drop = FALSE]
  myPrism <- new.prism(reference = sc.dat.filtered.pc, mixture = bk.use, input.type = "count.matrix",
                       cell.type.labels = cell.type.use, cell.state.labels = cell.state.use,
                       key = NULL, outlier.cut = 0.01, outlier.fraction = 0.1)
  bp.res <- run.prism(prism = myPrism, n.cores = N_CORES)
  theta <- get.fraction(bp = bp.res, which.theta = "final", state.or.type = "type")
  dir.create(dirname(out_file), recursive = TRUE, showWarnings = FALSE)
  write.csv(theta, file = out_file)
}

run_bayesprism(BULK_FILE, OUTPUT_FILE)

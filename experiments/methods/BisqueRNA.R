library(Biobase)
library(Matrix)

print("prepare bulk")
bulk.matrix <- as.matrix(read.csv("path to bulk expression.csv", row.names=1))
bulk.eset <- Biobase::ExpressionSet(assayData = bulk.matrix)

print("prepare scrna")
sc.counts.matrix <- Matrix::readMM("path to single-cell counts.mtx")
sc.counts.matrix <- as.matrix(sc.counts.matrix)
gene.metadata <- read.csv("path to single-cell genes.csv", row.names=1)
cell.metadata <- read.csv("path to single-cell metadata.csv", row.names=1)
rownames(sc.counts.matrix) <- rownames(gene.metadata)
colnames(sc.counts.matrix) <- rownames(cell.metadata)
sample.ids <- colnames(sc.counts.matrix)
individual.labels <- cell.metadata$sample
cell.type.labels <- cell.metadata$celltype

sc.pheno <- data.frame(check.names=FALSE, check.rows=FALSE, stringsAsFactors=FALSE,
                       row.names=sample.ids, SubjectName=individual.labels, cellType=cell.type.labels)
sc.meta <- data.frame(labelDescription=c("SubjectName", "cellType"),
                      row.names=c("SubjectName", "cellType"))
sc.pdata <- new("AnnotatedDataFrame", data=sc.pheno, varMetadata=sc.meta)
sc.eset <- Biobase::ExpressionSet(assayData = sc.counts.matrix, phenoData = sc.pdata)

print("Run BisqueRNA")
res <- BisqueRNA::ReferenceBasedDecomposition(bulk.eset, sc.eset, markers=NULL, use.overlap=FALSE)
ref.based.estimates <- res$bulk.props
write.csv(t(ref.based.estimates), file="path to BisqueRNA proportions.csv")

library(Seurat)
library(Signac)
library(Matrix)
library(EnsDb.Hsapiens.v86)
library(BSgenome.Hsapiens.UCSC.hg38)
library(future)
library(pheatmap)
library(dplyr)

metadata <- read.csv("/root/PRAGA/NewSDI/NewSDI-0/Figures/Figure_result_4/metadata.csv", row.names = 1)
rna_genes <- read.csv("/root/PRAGA/NewSDI/NewSDI-0/Figures/Figure_result_4/rna_genes.csv", row.names = 1, check.names = FALSE)
atac_peaks <- read.csv("/root/PRAGA/NewSDI/NewSDI-0/Figures/Figure_result_4/atac_peaks.csv", row.names = 1, check.names = FALSE)
rna_mtx <- readMM("/root/PRAGA/NewSDI/NewSDI-0/Figures/Figure_result_4/rna_counts.mtx")
atac_mtx <- readMM("/root/PRAGA/NewSDI/NewSDI-0/Figures/Figure_result_4/atac_counts.mtx")

rownames(rna_mtx) <- rownames(rna_genes)
colnames(rna_mtx) <- rownames(metadata)

rownames(atac_mtx) <- rownames(atac_peaks)
colnames(atac_mtx) <- rownames(metadata)

adata <- CreateSeuratObject(counts = rna_mtx, meta.data = metadata, assay = "RNA")
atac_assay <- CreateChromatinAssay(counts = atac_mtx, sep = c(":", "-"))
adata[["ATAC"]] <- atac_assay
Idents(adata) <- "SpaMEDM_2025"

DefaultAssay(adata) <- "RNA"
adata <- NormalizeData(object = adata, verbose = TRUE)
adata <- FindVariableFeatures(object = adata, nfeatures = 3000, verbose = TRUE)
adata <- ScaleData(object = adata, verbose = TRUE)

DefaultAssay(adata) <- "ATAC"
adata <- RunTFIDF(object = adata)
adata <- FindTopFeatures(object = adata)
adata <- RunSVD(object = adata)
annotations <- GetGRangesFromEnsDb(ensdb = EnsDb.Hsapiens.v86)

current_levels <- seqlevels(annotations)
new_levels <- paste0("chr", current_levels)
new_levels[new_levels == "chrMT"] <- "chrM"

seqlevels(annotations) <- new_levels
Annotation(adata) <- annotations

adata <- RegionStats(object = adata, assay = "ATAC", genome = BSgenome.Hsapiens.UCSC.hg38)
head(adata[["ATAC"]]@meta.features)
plan("multicore", workers = 4)

options(future.globals.maxSize = 32000 * 1024^2)
set.seed(2024)

get_count_layer <- function(object, assay_name) {
  tryCatch(
    GetAssayData(
      object = object,
      assay = assay_name,
      layer = "counts"
    ),
    error = function(e) {
      GetAssayData(
        object = object,
        assay = assay_name,
        slot = "counts"
      )
    }
  )
}

rna_counts <- get_count_layer(object = adata, assay_name = "RNA")
min_gene_cells <- max(10, ceiling(0.01 * ncol(rna_counts)))
gene_detection_counts <- Matrix::rowSums(rna_counts > 0)

genes_to_test <- names(
  gene_detection_counts[
    gene_detection_counts >= min_gene_cells
  ]
)

adata <- LinkPeaks(
  object = adata,
  peak.assay = "ATAC",
  expression.assay = "RNA",
  genes.use = genes_to_test,
  pvalue_cutoff = 1.01,
  score_cutoff = -1,
  min.cells = 10,
  verbose = TRUE
)

links_granges <- Links(adata)
links <- as.data.frame(links_granges)
required_columns <- c("gene", "peak", "score", "zscore", "pvalue")

missing_columns <- setdiff(required_columns, colnames(links))

links <- links %>%
  filter(
    is.finite(pvalue),
    is.finite(score),
    !is.na(gene),
    !is.na(peak)
  )

links$p_adjusted <- p.adjust(
  p = links$pvalue,
  method = "BH"
)

all_sig_links <- links %>%
  filter(
    p_adjusted < 0.05,
    abs(score) >= 0.05
  ) %>%
  arrange(p_adjusted)

write.csv(
  x = all_sig_links,
  file = "fig4_F.csv",
  row.names = FALSE,
  quote = FALSE
)

plot_genes <- as.character(all_sig_links$gene)
plot_peaks <- as.character(all_sig_links$peak)

linked_genes <- unique(plot_genes)
linked_peaks <- unique(plot_peaks)

missing_genes <- setdiff(linked_genes, rownames(adata[["RNA"]]))
missing_peaks <- setdiff(linked_peaks, rownames(adata[["ATAC"]]))
domain_values <- as.character(adata@meta.data[colnames(adata), "SpaMEDM_2025"])
names(domain_values) <- colnames(adata)
identity_levels <- levels(Idents(adata))
observed_domains <- unique(domain_values)
domain_levels <- identity_levels[identity_levels %in% observed_domains]

if (length(domain_levels) == 0) {domain_levels <- observed_domains}
additional_domains <- setdiff(observed_domains, domain_levels)
domain_levels <- c(domain_levels, additional_domains)
domain_index <- match(domain_values, domain_levels)
domain_design <- Matrix::sparseMatrix(
  i = seq_along(domain_index),
  j = domain_index,
  x = rep(1, length(domain_index)),
  dims = c(
    length(domain_index),
    length(domain_levels)
  ),
  dimnames = list(
    colnames(adata),
    domain_levels
  )
)

metadata_domain_counts <- table(
  factor(
    domain_values,
    levels = domain_levels
  )
)

get_count_layer <- function(object, assay_name) {
  tryCatch(
    GetAssayData(
      object = object,
      assay = assay_name,
      layer = "counts"
    ),
    error = function(e) {
      GetAssayData(
        object = object,
        assay = assay_name,
        slot = "counts"
      )
    }
  )
}

rna_counts <- get_count_layer(object = adata, assay_name = "RNA")
atac_counts <- get_count_layer(object = adata, assay_name = "ATAC")

rna_pseudobulk_counts <- rna_counts %*% domain_design
atac_pseudobulk_counts <- atac_counts %*% domain_design

dimnames(rna_pseudobulk_counts) <- list(
  rownames(rna_counts),
  domain_levels
)

dimnames(atac_pseudobulk_counts) <- list(
  rownames(atac_counts),
  domain_levels
)

normalize_pseudobulk <- function(
  count_matrix,
  scale_factor = 10000
) {
  original_dimnames <- dimnames(count_matrix)
  library_sizes <- Matrix::colSums(count_matrix)
  if (any(library_sizes <= 0)) {
    bad_domains <- colnames(count_matrix)[library_sizes <= 0]
    stop(
      "Zero-count pseudobulk domains detected: ",
      paste(bad_domains, collapse = ", ")
    )
  }

  scaling_factors <- scale_factor / library_sizes
  scaling_matrix <- Matrix::Diagonal(
    n = ncol(count_matrix),
    x = scaling_factors
  )

  dimnames(scaling_matrix) <- list(
    colnames(count_matrix),
    colnames(count_matrix)
  )

  normalized_matrix <- count_matrix %*% scaling_matrix
  dimnames(normalized_matrix) <- original_dimnames

  if (inherits(normalized_matrix, "sparseMatrix")) {
    normalized_matrix@x <- log1p(normalized_matrix@x)
  } else {
    normalized_matrix <- log1p(normalized_matrix)
  }

  dimnames(normalized_matrix) <- original_dimnames

  return(normalized_matrix)
}

rna_pseudobulk_normalized <- normalize_pseudobulk(
  count_matrix = rna_pseudobulk_counts,
  scale_factor = 10000
)

atac_pseudobulk_normalized <- normalize_pseudobulk(
  count_matrix = atac_pseudobulk_counts,
  scale_factor = 10000
)

rna_domain_matrix <- rna_pseudobulk_normalized[
  linked_genes,
  ,
  drop = FALSE
]

atac_domain_matrix <- atac_pseudobulk_normalized[
  linked_peaks,
  ,
  drop = FALSE
]

colnames(rna_domain_matrix) <- domain_levels
colnames(atac_domain_matrix) <- domain_levels

scale_rows_safe <- function(x) {
  feature_names <- rownames(x)
  domain_names <- colnames(x)
  x <- as.matrix(x)
  row_means <- rowMeans(x)
  row_sds <- apply(x, 1, sd)
  row_sds[
    !is.finite(row_sds) |
      row_sds == 0
  ] <- 1
  result <- sweep(
    x,
    MARGIN = 1,
    STATS = row_means,
    FUN = "-"
  )
  result <- sweep(
    result,
    MARGIN = 1,
    STATS = row_sds,
    FUN = "/"
  )
  result[!is.finite(result)] <- 0
  dimnames(result) <- list(
    feature_names,
    domain_names
  )
  return(result)
}

rna_mat_scaled_unique <- scale_rows_safe(rna_domain_matrix)
atac_mat_scaled_unique <- scale_rows_safe(atac_domain_matrix)
mat_rna_side <- rna_mat_scaled_unique[
  plot_genes,
  ,
  drop = FALSE
]

mat_atac_side <- atac_mat_scaled_unique[
  plot_peaks,
  ,
  drop = FALSE
]

colnames(mat_rna_side) <- domain_levels
colnames(mat_atac_side) <- domain_levels

combined_mat_side <- cbind(
  mat_rna_side,
  mat_atac_side
)

domain_labels <- domain_levels
number_of_domains <- length(domain_labels)

internal_colnames <- c(
  paste0("RNA_", domain_labels),
  paste0("ATAC_", domain_labels)
)

colnames(combined_mat_side) <- internal_colnames

col_annotation <- data.frame(
  Modality = rep(
    c("RNA", "ATAC"),
    each = number_of_domains
  )
)

rownames(col_annotation) <- internal_colnames

display_labels <- rep(
  domain_labels,
  times = 2
)

my_colors <- colorRampPalette(
  c("navy", "white", "firebrick3")
)(100)

my_breaks <- seq(
  from = -3,
  to = 3,
  length.out = 101
)

output_height <- ifelse(
  nrow(combined_mat_side) > 1000,
  20,
  12
)

pdf(file = "fig4_F.pdf", width = 8, height = 10)

pheatmap(
  combined_mat_side,
  cluster_rows = TRUE,
  cluster_cols = FALSE,
  annotation_col = col_annotation,
  annotation_names_col = FALSE,
  labels_col = display_labels,
  show_rownames = FALSE,
  border_color = NA,
  gaps_col = number_of_domains,
  color = my_colors,
  breaks = my_breaks,

  main = paste0(
    "Peak2Gene Links (n = ",
    nrow(all_sig_links),
    ")"
  ),
  angle_col = "0",
  fontsize_col = 16,
  fontsize = 14,
  treeheight_row = 0
)

dev.off()


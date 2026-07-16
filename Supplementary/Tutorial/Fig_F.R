library(Seurat)
library(Signac)
library(Matrix)

metadata <- read.csv("/root/PRAGA/NewSDI/NewSDI-0/Figures/Figure_result_4/metadata.csv", row.names = 1)
rna_genes <- read.csv("/root/PRAGA/NewSDI/NewSDI-0/Figures/Figure_result_4/rna_genes.csv", row.names = 1, check.names = FALSE)
atac_peaks <- read.csv("/root/PRAGA/NewSDI/NewSDI-0/Figures/Figure_result_4/atac_peaks.csv", row.names = 1, check.names = FALSE)

rna_mtx <- readMM("/root/PRAGA/NewSDI/NewSDI-0/Figures/Figure_result_4/rna_counts.mtx")
rownames(rna_mtx) <- rownames(rna_genes)
colnames(rna_mtx) <- rownames(metadata)

adata <- CreateSeuratObject(counts = rna_mtx, meta.data = metadata, assay = "RNA")

atac_mtx <- readMM("/root/PRAGA/NewSDI/NewSDI-0/Figures/Figure_result_4/atac_counts.mtx")
rownames(atac_mtx) <- rownames(atac_peaks)
colnames(atac_mtx) <- rownames(metadata)

atac_assay <- CreateChromatinAssay(counts = atac_mtx, sep = c(":", "-"))
adata[["ATAC"]] <- atac_assay

Idents(adata) <- "SpaMEDM_2025"

print(adata)

DefaultAssay(adata) <- "RNA"
adata <- NormalizeData(adata)
adata <- FindVariableFeatures(adata, nfeatures=3000)
adata <- ScaleData(adata)

DefaultAssay(adata) <- "ATAC"
adata <- RunTFIDF(adata)
adata <- FindTopFeatures(adata)
adata <- RunSVD(adata)

library(EnsDb.Hsapiens.v86)
library(BSgenome.Hsapiens.UCSC.hg38)

annotations <- GetGRangesFromEnsDb(ensdb = EnsDb.Hsapiens.v86)
current_levels <- seqlevels(annotations)

new_levels <- paste0("chr", current_levels)

new_levels[new_levels == "chrMT"] <- "chrM"

seqlevels(annotations) <- new_levels
Annotation(adata) <- annotations

adata <- RegionStats(adata, genome = BSgenome.Hsapiens.UCSC.hg38)
head(adata[["ATAC"]]@meta.features)

library(future)
plan("multicore", workers = 30)
options(future.globals.maxSize = 32000 * 1024^2)
set.seed(2024)

adata <- LinkPeaks(
  object = adata,
  peak.assay = "ATAC",
  expression.assay = "RNA",
  pvalue_cutoff = 0.05,
  score_cutoff = 0.05,
  genes.use = VariableFeatures(adata, assay = "RNA")
)

library(pheatmap)
library(dplyr)

links <- as.data.frame(Links(adata))
links$p_adjusted <- p.adjust(links$pvalue, method = "BH")
all_sig_links <- links %>% filter(p_adjusted < 0.05)

plot_genes <- all_sig_links$gene
plot_peaks <- all_sig_links$peak
linked_genes <- unique(plot_genes)
linked_peaks <- unique(plot_peaks)

Idents(adata) <- "SpaMEDM_2025"
avg_data <- AverageExpression(adata, features = c(linked_genes, linked_peaks), return.seurat = FALSE)

scale_rows_safe <- function(x) {
  res <- t(scale(t(x)))
  res[is.na(res)] <- 0
  return(res)
}

rna_mat_scaled <- scale_rows_safe(avg_data$RNA[linked_genes, ])
atac_mat_scaled <- scale_rows_safe(avg_data$ATAC[linked_peaks, ])

clean_colnames <- gsub("^g", "", colnames(rna_mat_scaled))

mat_rna_side <- rna_mat_scaled[as.character(plot_genes), ]
mat_atac_side <- atac_mat_scaled[as.character(plot_peaks), ]

combined_mat_side <- cbind(mat_rna_side, mat_atac_side)

internal_colnames <- c(paste0("RNA_", clean_colnames), paste0("ATAC_", clean_colnames))
colnames(combined_mat_side) <- internal_colnames

col_annotation <- data.frame(
  Modality = rep(c("RNA", "ATAC"), each = length(clean_colnames))
)
rownames(col_annotation) <- internal_colnames

display_labels <- rep(clean_colnames, 2)
my_colors <- colorRampPalette(c("navy", "white", "firebrick3"))(100)
my_breaks <- seq(-3, 3, length.out = 101)

output_height <- ifelse(nrow(combined_mat_side) > 1000, 20, 12)
pdf("fig4_F.pdf", width = 8, height = 10)

pheatmap(
  combined_mat_side,
  cluster_rows = TRUE,
  cluster_cols = FALSE,
  annotation_col = col_annotation,
  labels_col = display_labels,
  show_rownames = FALSE,
  border_color = NA,
  gaps_col = 10,
  color = my_colors,
  breaks = my_breaks,
  annotation_names_col = FALSE,
  main = paste0("Peak2Gene Links (n = ", nrow(all_sig_links), ")"),
  angle_col = "0",
  fontsize_col = 16,
  fontsize = 14,
  treeheight_row = 0,
)


write.csv(
  x = all_sig_links,
  file = "fig4_F.csv",
  row.names = FALSE,
  quote = FALSE
)

dev.off()



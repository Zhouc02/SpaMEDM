library(dplyr)
library(clusterProfiler)
library(org.Mm.eg.db)
library(ggplot2)

markers.all <- read.csv('C:/Users/Admin/Documents/IllustratorProjects/SpaMEDM/Supple1/sup1_C_D.csv')

colnames(markers.all)[colnames(markers.all) == "group"] <- "cluster"
colnames(markers.all)[colnames(markers.all) == "names"] <- "gene"
colnames(markers.all)[colnames(markers.all) == "logfoldchanges"] <- "avg_log2FC"

markers <- markers.all %>%
  dplyr::group_by(cluster) %>%
  dplyr::top_n(n = 100, wt = avg_log2FC)

gene_list <- markers$gene
gene_entrez <- bitr(gene_list, 
                    fromType = "SYMBOL", 
                    toType = "ENTREZID", 
                    OrgDb = org.Mm.eg.db)

markers_with_entrez <- merge(markers, gene_entrez, by.x = "gene", by.y = "SYMBOL")

cluster7_markers_entrez <- subset(markers_with_entrez, as.character(cluster) == "10")

cluster7_gene_ids <- cluster7_markers_entrez$ENTREZID

go_result_c7_all <- enrichGO(gene          = cluster7_gene_ids,
                             OrgDb         = org.Mm.eg.db,
                             keyType       = 'ENTREZID',
                             ont           = "BP", 
                             pAdjustMethod = "BH",
                             pvalueCutoff  = 0.05,
                             qvalueCutoff  = 0.05)

pdf('./figS1_D.pdf', onefile = FALSE, width = 10, height = 8)

p <- dotplot(go_result_c7_all, showCategory = 15, font.size = 17)

p_final <- p + theme(
  legend.title = element_text(size = 17), 
  legend.text  = element_text(size = 17)    
)

print(p_final)
dev.off()

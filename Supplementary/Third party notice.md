# Third-Party Code Attribution

This project reuses or adapts code from the following open-source projects. We gratefully acknowledge their contributions.

| Source Project | License | Files/Functions | Role in SpaMEDM | Modifications |
|:---|:---|:---|:---|:---|
| [SpatialGlue](https://github.com/JinmiaoChenLab/SpatialGlue) | AGPL-3.0 | `preprocess.py` (pca) | Dimension reduction with PCA algorithm | No modification |
| [SpatialGlue](https://github.com/JinmiaoChenLab/SpatialGlue) | AGPL-3.0 | `preprocess.py` (clr_normalize_each_cell) | Normalize count vector for each cell | No modification |
| [SpatialGlue](https://github.com/JinmiaoChenLab/SpatialGlue) | AGPL-3.0 | `preprocess.py` (lsi) | ATAC/Epigenome LSI analysis | No modification |
| [SpatialGlue](https://github.com/JinmiaoChenLab/SpatialGlue) | AGPL-3.0 | `preprocess.py` (tfidf) | Protein TF-IDF normalization | No modification |
| [SpatialGlue](https://github.com/JinmiaoChenLab/SpatialGlue) | AGPL-3.0 | `utils.py` (mclust_R) | Clustering using the mclust algorithm | No modification |
| [SpatialGlue](https://github.com/JinmiaoChenLab/SpatialGlue) | AGPL-3.0 | `utils.py` (clustering) | Spatial clustering based the latent representation | Add K-means |
| [SpatialGlue](https://github.com/JinmiaoChenLab/SpatialGlue) | AGPL-3.0 | `utils.py` (search_res) | Searching corresponding resolution according to given cluster number for leiden and louvain | No modification |
| [SpatialGlue](https://github.com/JinmiaoChenLab/SpatialGlue) | AGPL-3.0 | `model.py` (Attention) | Inter- and Intra-modality fusion | Remove additional dimension parameter. Introduce train/inference mode distinction, returning only fused features during training and additionally returning attention weights during inference. Add explicit weighted summation |
| [smows](https://github.com/Kyochilian/smows) | MIT | `preprocess.py` (post_proC) | Spectral clustering | No modification |

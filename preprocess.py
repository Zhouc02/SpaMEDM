import os

import h5py
import scipy
import anndata
import sklearn
import torch
import random
import numpy as np
import scanpy as sc
import pandas as pd
from typing import Optional
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from sklearn import cluster
from sklearn.preprocessing import normalize
from torch.backends import cudnn
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph
from sklearn.decomposition import PCA
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix


def pca(adata, use_reps=None, n_comps=10):

    """Dimension reduction with PCA algorithm"""

    pca = PCA(n_components=n_comps)

    if use_reps is not None:
        feat_pca = pca.fit_transform(adata.obsm[use_reps])
    else:
        if isinstance(adata.X, csc_matrix) or isinstance(adata.X, csr_matrix):
            feat_pca = pca.fit_transform(adata.X.toarray())
        else:
            feat_pca = pca.fit_transform(adata.X)

    return feat_pca


def clr_normalize_each_cell(adata, inplace=True):

    """Normalize count vector for each cell, i.e. for each row of .X"""

    import numpy as np
    import scipy

    def seurat_clr(x):
        # TODO: support sparseness
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x))
        return np.log1p(x / exp)

    if not inplace:
        adata = adata.copy()

    # apply to dense or sparse matrix, along axis. returns dense matrix
    adata.X = np.apply_along_axis(
        seurat_clr, 1, (adata.X.A if scipy.sparse.issparse(adata.X) else np.array(adata.X))
    )
    return adata


def lsi(
        adata: anndata.AnnData, n_components: int = 20,
        use_highly_variable: Optional[bool] = None, **kwargs) -> None:
    r"""
    LSI analysis (following the Seurat v3 approach)
    """
    if use_highly_variable is None:
        use_highly_variable = "highly_variable" in adata.var
    adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
    X = tfidf(adata_use.X)
    X_norm = sklearn.preprocessing.Normalizer(norm="l1").fit_transform(X)
    X_norm = np.log1p(X_norm * 1e4)
    X_lsi = sklearn.utils.extmath.randomized_svd(X_norm, n_components, **kwargs)[0]
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    adata.obsm["X_lsi"] = X_lsi[:, 1:]


def tfidf(X):
    r"""
    TF-IDF normalization (following the Seurat v3 approach)
    """
    idf = X.shape[0] / X.sum(axis=0)
    if scipy.sparse.issparse(X):
        tf = X.multiply(1 / X.sum(axis=1))
        return tf.multiply(idf)
    else:
        tf = X / X.sum(axis=1, keepdims=True)
        return tf * idf


def fix_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def construct_neighbor_graph(adata_omics1, adata_omics2, mode):
    omics1_feat, omics2_feat = adata_omics1.obsm['feat'], adata_omics2.obsm['feat']

    adj_omics1, adj_norm_omics1 = construct_graph(omics1_feat, k=10)
    adj_omics2, adj_norm_omics2 = construct_graph(omics2_feat, k=10)

    adj_spatial, adj_spatial_norm = construct_graph(adata_omics1.obsm['spatial'], k=10)

    if mode:
        adj_spatial_refine_omics1 = refine_adj_spatial(adj_omics1, adj_spatial)
        adj_spatial_refine_omics2 = refine_adj_spatial(adj_omics2, adj_spatial)
        return (omics1_feat, omics2_feat, adj_norm_omics1, adj_norm_omics2,
                adj_spatial_refine_omics1, adj_spatial_refine_omics2)
    else:
        return (omics1_feat, omics2_feat, adj_norm_omics1, adj_norm_omics2,
                adj_spatial_norm, adj_spatial_norm)


def degree_power(A, k):
    degrees = np.power(np.array(A.sum(1)), k).flatten()
    degrees[np.isinf(degrees)] = 0.
    if sp.issparse(A):
        D = sp.diags(degrees)
    else:
        D = np.diag(degrees)
    return D


def norm_adj(A):
    normalized_D = degree_power(A, -0.5)
    output = normalized_D.dot(A).dot(normalized_D)
    return output


def construct_graph(count, k=10, mode="connectivity"):
    countp = count
    A = kneighbors_graph(countp, k, mode=mode, metric="euclidean", include_self=True)
    adj = A.toarray()

    adj = (adj.T + adj) / 2
    adj_n = norm_adj(adj)

    return adj, adj_n


def refine_adj_spatial(feature_graph, spatial_graph):
    mask = np.logical_and(feature_graph > 0, spatial_graph > 0)
    spatial_graph_refine = np.where(mask, spatial_graph, 0)

    return norm_adj(spatial_graph_refine)


def post_proC(C, K, d=11, alpha=4):
    C = 0.5 * (C + C.T)
    r = d * K + 1
    U, S, _ = svds(C, r, v0=np.ones(C.shape[0]))
    U = U[:, ::-1]
    S = np.sqrt(S[::-1])
    S = np.diag(S)
    U = U.dot(S)
    U = normalize(U, norm='l2', axis=1)
    Z = U.dot(U.T)
    Z = Z * (Z > 0)
    L = np.abs(Z ** alpha)
    L = L / L.max()
    L = 0.5 * (L + L.T)
    spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',
                                          assign_labels='discretize')
    spectral.fit(L)
    grp = spectral.fit_predict(L) + 1
    return grp, U


def read_and_preprocess_data(args):
    adata_omics1, adata_omics2 = None, None
    if 'HLN' in args.data_type or 'SPOTS' in args.data_type:
        adata_omics1 = sc.read_h5ad(args.file_fold + 'adata_RNA.h5ad')
        adata_omics2 = sc.read_h5ad(args.file_fold + 'adata_ADT.h5ad')
    elif 'H3K' in args.data_type or 'Mouse_RNA_ATAC' in args.data_type:
        adata_omics1 = sc.read_h5ad(args.file_fold + 'adata_RNA.h5ad')
        adata_omics2 = sc.read_h5ad(args.file_fold + 'adata_peaks_normalized.h5ad')
    elif 'E1' in args.data_type:
        adata_omics1 = sc.read_h5ad(args.file_fold + 'adata_RNA.h5ad')
        adata_omics2 = sc.read_h5ad(args.file_fold + 'adata_Peak.h5ad')
    elif 'Slide' in args.data_type:
        adata_omics1 = sc.read_h5ad(args.file_fold + 'adata_RNA.h5ad')
        adata_omics2 = sc.read_h5ad(args.file_fold + 'adata_ATAC.h5ad')
    elif args.data_type == 'meta':
        adata_omics1 = sc.read_h5ad(args.file_fold + 'adata_RNA.h5ad')
        adata_omics2 = sc.read_h5ad(args.file_fold + 'adata_metabolic.h5ad')
    elif args.data_type == 'CITE':
        data_mat = h5py.File(args.file_fold + 'Spatial_CITE_seq_HumanTonsil_RNA_Protein.h5')
        df_data_RNA = np.array(data_mat['X_gene']).astype('float64')
        df_data_protein = np.array(data_mat['X_protein']).astype('float64')
        loc = np.array(data_mat['pos']).astype('float64')
        gene_names = list(data_mat['gene'])
        gene_names = [gene.decode("utf-8") for gene in gene_names]
        protein_names = list(data_mat['protein'])
        protein_names = [protein.decode("utf-8") for protein in protein_names]
        protein_names = [protein.split(".")[0] for protein in protein_names]
        adata_omics1 = sc.AnnData(df_data_RNA, dtype="float64")
        adata_omics1.index = gene_names
        adata_omics2 = sc.AnnData(df_data_protein, dtype="float64")
        adata_omics2.index = protein_names
        adata_omics1.obsm['spatial'] = np.array(loc)
        adata_omics2.obsm['spatial'] = np.array(loc)

    adata_omics1.var_names_make_unique()
    adata_omics2.var_names_make_unique()

    fix_seed(args.random_seed)

    if 'HLN' in args.data_type:
        sc.pp.filter_genes(adata_omics1, min_cells=10)
        sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
        sc.pp.normalize_total(adata_omics1, target_sum=1e4)
        sc.pp.log1p(adata_omics1)
        sc.pp.scale(adata_omics1)
        adata_omics1_high = adata_omics1[:, adata_omics1.var['highly_variable']]
        adata_omics1.obsm['feat'] = pca(adata_omics1_high, n_comps=100)

        adata_omics2 = clr_normalize_each_cell(adata_omics2)
        sc.pp.scale(adata_omics2)
        adata_omics2.obsm['feat'] = pca(adata_omics2, n_comps=adata_omics2.n_vars - 1)
        data = construct_neighbor_graph(adata_omics1, adata_omics2, mode=args.pruning)

    elif 'H3K' in args.data_type or 'Mouse_RNA_ATAC' in args.data_type:
        sc.pp.filter_genes(adata_omics1, min_cells=10)
        sc.pp.filter_cells(adata_omics1, min_genes=200)
        sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
        sc.pp.normalize_total(adata_omics1, target_sum=1e4)
        sc.pp.log1p(adata_omics1)
        sc.pp.scale(adata_omics1)
        adata_omics1_high = adata_omics1[:, adata_omics1.var['highly_variable']]
        adata_omics1.obsm['feat'] = pca(adata_omics1_high, n_comps=50)

        adata_omics2 = adata_omics2[
            adata_omics1.obs_names].copy()  # .obsm['X_lsi'] represents the dimension reduced feature
        if 'X_lsi' not in adata_omics2.obsm.keys():
            sc.pp.highly_variable_genes(adata_omics2, flavor="seurat_v3", n_top_genes=3000)
            lsi(adata_omics2, use_highly_variable=False, n_components=51)
        adata_omics2.obsm['feat'] = adata_omics2.obsm['X_lsi'].copy()
        data = construct_neighbor_graph(adata_omics1, adata_omics2, mode=args.pruning)

    elif 'SPOTS' in args.data_type:
        sc.pp.filter_genes(adata_omics1, min_cells=10)
        sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
        sc.pp.normalize_total(adata_omics1, target_sum=1e4)
        sc.pp.log1p(adata_omics1)
        sc.pp.scale(adata_omics1)
        adata_omics1_high = adata_omics1[:, adata_omics1.var['highly_variable']]
        adata_omics1.obsm['feat'] = pca(adata_omics1_high, n_comps=100)

        adata_omics2 = clr_normalize_each_cell(adata_omics2)
        sc.pp.scale(adata_omics2)
        adata_omics2.obsm['feat'] = pca(adata_omics2, n_comps=adata_omics2.n_vars - 1)
        data = construct_neighbor_graph(adata_omics1, adata_omics2, mode=args.pruning)

    elif args.data_type == 'meta':
        sc.pp.filter_genes(adata_omics1, min_cells=50)
        sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
        adata_omics1_high = adata_omics1[:, adata_omics1.var['highly_variable']]
        adata_omics1.obsm['feat'] = pca(adata_omics1_high, n_comps=50)

        adata_omics2.obsm['feat'] = pca(adata_omics2, n_comps=50)
        data = construct_neighbor_graph(adata_omics1, adata_omics2, mode=args.pruning)

    elif 'Slide' in args.data_type or 'E1' in args.data_type:
        sc.pp.filter_genes(adata_omics1, min_cells=50)
        sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
        sc.pp.normalize_total(adata_omics1, target_sum=1e4)
        sc.pp.log1p(adata_omics1)
        sc.pp.scale(adata_omics1)
        adata_omics1_high = adata_omics1[:, adata_omics1.var['highly_variable']]
        adata_omics1.obsm['feat'] = pca(adata_omics1_high, n_comps=50)

        adata_omics2 = adata_omics2[adata_omics1.obs_names].copy()
        if 'X_lsi' not in adata_omics2.obsm.keys():
            sc.pp.highly_variable_genes(adata_omics2, flavor="seurat_v3", n_top_genes=3000)
            lsi(adata_omics2, use_highly_variable=False, n_components=21)
        adata_omics2.obsm['feat'] = adata_omics2.obsm['X_lsi'].copy()
        data = construct_neighbor_graph(adata_omics1, adata_omics2, mode=args.pruning)

    elif args.data_type == 'CITE':
        sc.pp.normalize_total(adata_omics1, target_sum=1e4)
        sc.pp.log1p(adata_omics1)
        sc.pp.scale(adata_omics1)
        adata_omics1.obsm['feat'] = pca(adata_omics1, n_comps=100)

        adata_omics2 = clr_normalize_each_cell(adata_omics2)
        sc.pp.scale(adata_omics2)
        adata_omics2.obsm['feat'] = pca(adata_omics2, n_comps=50)
        data = construct_neighbor_graph(adata_omics1, adata_omics2, mode=args.pruning)
    else:
        assert 0

    return data, adata_omics1

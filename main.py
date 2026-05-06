import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import kneighbors_graph

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["R_HOME"] = "/root/anaconda3/envs/PRAGA/lib/R"

import argparse
from preprocess import fix_seed, read_and_preprocess_data
from metrics import metric
from train import Train
from utils import clustering
from params import params

def main(args):
    fix_seed(args.random_seed)
    data, adata_omics1 = read_and_preprocess_data(args)
    model = Train(data, args)
    label, SVD_U, emb, z1, z2, z = model.train()
    adata = adata_omics1.copy()
    adata.obsm['SpaMEDM'] = emb.copy()
    adata.obs['SpaMEDM'] = label
    adata.obs['SpaMEDM'] = adata.obs.SpaMEDM.astype('category')
    if args.tool != 'svd':
        clustering(adata, key='SpaMEDM', add_key='SpaMEDM', n_clusters=args.n_clusters, method=args.tool, use_pca=True)
    metric(args, adata)

    if args.data_type == 'CITE':
        y_max = np.max(adata.obsm['spatial'][:, 1])
        adata.obsm['spatial'][:, 1] = y_max - adata.obsm['spatial'][:, 1]
    if args.data_type == 'Mouse_RNA_H3K27ac' or args.data_type == 'Mouse_RNA_H3K4me3':
        spatial_coords = adata.obsm['spatial'].copy()
        rotated_coords = np.column_stack([spatial_coords[:, 1], -spatial_coords[:, 0]])
        adata.obsm['spatial'] = rotated_coords
    if args.data_type == 'Mouse_RNA_H3K27me3':
        spatial_coords = adata.obsm['spatial'].copy()
        rotated_coords = np.column_stack([spatial_coords[:, 1], -spatial_coords[:, 0]])
        rotated_coords[:, 1] = -rotated_coords[:, 1]
        adata.obsm['spatial'] = rotated_coords
    if 'E1' in args.data_type:
        adata.obsm['spatial'][:, 1] *= -1
    if 'SPOTS' in args.data_type:
        adata.obsm['spatial'] = np.rot90(np.rot90(np.rot90(np.array(adata.obsm['spatial'])).T).T).T
        adata.obsm['spatial'][:, 1] = -1 * adata.obsm['spatial'][:, 1]
    import scanpy as sc
    sc.pl.embedding(adata, basis='spatial', color='SpaMEDM', s=100, show=False)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to modify global variable')
    parser.add_argument('--data_type', type=str, default='Mouse_RNA_ATAC', help='data_type')
    parser.add_argument('--random_seed', type=int, default=2024, help='random_seed')
    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    parser.add_argument('--tool', type=str, default='svd', choices=['mclust', 'leiden', 'louvain'], help='tool for clustering')
    parser.add_argument('--dim_output', type=int, default=64, help='dimension of output data')
    parser.add_argument('--weight1', type=float, default=15, help='weight1')
    parser.add_argument('--weight2', type=float, default=0.01, help='weight2')
    parser.add_argument('--weight3', type=float, default=0.5, help='weight3')
    parser.add_argument('--weight4', type=float, default=1, help='weight4')
    parser.add_argument('--epochs', type=int, default=400, help='epochs')
    parser.add_argument('--mask', type=float, default=0.25, help='mask')
    parser.add_argument('--pruning', action='store_true', help='if utilize pruned spatial graph')
    parser.add_argument('--single', action='store_true', help='if utilize single spatial graph')
    args = parser.parse_args()
    args = params(args)
    main(args)

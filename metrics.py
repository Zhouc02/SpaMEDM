import pandas as pd
from sklearn.metrics import normalized_mutual_info_score, mutual_info_score, adjusted_mutual_info_score
from sklearn.metrics import v_measure_score, homogeneity_score, completeness_score
from sklearn.metrics import adjusted_rand_score, fowlkes_mallows_score
import numpy as np
from sklearn import metrics
from s_dbw import S_Dbw
import scanpy as sc
import squidpy as sq
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.manifold import trustworthiness


def metric(args, adata):
    if args.data_type in ['E11_0-S1', 'Slide', 'E15_5-S1', 'E13_5-S1', 'HLN_A1', 'E18_5-S1', 'HLN_D1', 'meta']:
        GT = []
        with open(f'{args.file_fold}GT_labels.txt', 'r') as f:
            for line in f:
                num = int(line.strip())
                GT.append(num)
        GT_list = GT
        Our_list = adata.obs.SpaMEDM.tolist()
        print(min(GT_list), max(GT_list))
        print(min(Our_list), max(Our_list))
        print(set(GT_list))
        print(set(Our_list))
        print(len(GT_list))
        print(len(Our_list))
        print(f"MI: {mutual_info_score(GT_list, Our_list):.6f}")
        print(f"NMI: {normalized_mutual_info_score(GT_list, Our_list):.6f}")
        print(f"AMI: {adjusted_mutual_info_score(GT_list, Our_list):.6f}")
        print(f"V-measure: {v_measure_score(GT_list, Our_list):.6f}")
        print(f"Homogeneity: {homogeneity_score(GT_list, Our_list):.6f}")
        print(f"Completeness: {completeness_score(GT_list, Our_list):.6f}")
        print(f"ARI: {adjusted_rand_score(GT_list, Our_list):.6f}")
        print(f"FMI: {fowlkes_mallows_score(GT_list, Our_list):.6f}")
import scanpy as sc
import pandas as pd
import numpy as np
# from matplotlib import pyplot as plt
# from matplotlib.lines import Line2D
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score, \
    v_measure_score, fowlkes_mallows_score

data_type = 'Slide'
adata = sc.read_h5ad("/root/PRAGA/Data/Slide_tag/adata_RNA.h5ad")
adata.obs['Combined_Clusters_annotation'] = adata.obs['cluster']

sc.pl.embedding(adata, basis='spatial', color='Combined_Clusters_annotation')
colors = adata.uns['Combined_Clusters_annotation_colors']
ground_truth = adata.obs['Combined_Clusters_annotation']

methods = ['SpaMEDM']


paths = [f'/remote-home/zhouchang/PRAGA/SpaNodeDGI/new_results/adata_{data_type}.h5ad']

best_ARI = []
best_NMI = []
best_AMI = []
best_FMI = []
best_V_Measure = []
best_obs = []

for i, path in enumerate(paths):
    method = methods[i]
    temp_adata = sc.read_h5ad(path)

    current_best_ari = -1.0
    current_best_metrics = {
        'NMI': np.nan,
        'AMI': np.nan,
        'FMI': np.nan,
        'V_Measure': np.nan,
        'obs_name': None
    }

    for col in temp_adata.obs.columns:
        pred_labels = temp_adata.obs[col]

        y_true = ground_truth
        y_pred = pred_labels

        assert len(np.unique(y_pred)) == len(np.unique(ground_truth))

        ari = adjusted_rand_score(y_true, y_pred)
        if ari > current_best_ari:
            current_best_ari = ari
            current_best_metrics['NMI'] = normalized_mutual_info_score(y_true, y_pred)
            current_best_metrics['AMI'] = adjusted_mutual_info_score(y_true, y_pred)
            current_best_metrics['FMI'] = fowlkes_mallows_score(y_true, y_pred)
            current_best_metrics['V_Measure'] = v_measure_score(y_true, y_pred)
            current_best_metrics['obs_name'] = col

    best_ARI.append(current_best_ari)
    best_NMI.append(current_best_metrics['NMI'])
    best_AMI.append(current_best_metrics['AMI'])
    best_FMI.append(current_best_metrics['FMI'])
    best_V_Measure.append(current_best_metrics['V_Measure'])
    best_obs.append(current_best_metrics['obs_name'])
    print(f"{method}: Found best seed -> {current_best_metrics['obs_name']} (ARI={current_best_ari:.4f})")


gt_categories = ground_truth.cat.categories
gt_color_dict = {str(cat): colors[i] for i, cat in enumerate(gt_categories)}
pred_colors = []

for i, path in enumerate(paths):
    method = methods[i]
    best_seed = best_obs[i]
    temp_adata = sc.read_h5ad(path)
    pred_labels = temp_adata.obs[best_seed].cat.codes.astype('int64')
    contingency_matrix = pd.crosstab(ground_truth.tolist(), pred_labels.tolist())
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
    mapping_dict = {contingency_matrix.columns[j]: contingency_matrix.index[i] for i, j in zip(row_ind, col_ind)}
    unique_true_labels = np.unique(ground_truth)
    color_map = {label: colors[i % len(colors)] for i, label in enumerate(unique_true_labels)}
    unique_pred_labels = sorted(contingency_matrix.columns)
    real_colors = []
    for pred_label in unique_pred_labels:
        mapped_true_label = mapping_dict.get(pred_label)
        if mapped_true_label is not None:
            real_colors.append(color_map.get(mapped_true_label, '#808080'))
        else:
            print("Other color!")
            real_colors.append('#808080')
    pred_colors.append(real_colors)

colors = pred_colors[-1]

import plotly.graph_objects as go
import matplotlib.colors as mcolors

gt_labels = contingency_matrix.index.tolist()
pred_labels_list = contingency_matrix.columns.tolist()

node_labels = [f"{label}" for label in gt_labels] + [f"{label+1}" for label in pred_labels_list]

gt_node_colors = [gt_color_dict.get(str(label), '#808080') for label in gt_labels]
pred_node_colors = colors
all_node_colors = gt_node_colors + pred_node_colors

source = []
target = []
value = []
link_colors = []

def hex_to_rgba(hex_color, alpha=0.4):
    try:
        rgb = mcolors.hex2color(hex_color)
        return f"rgba({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)}, {alpha})"
    except:
        return f"rgba(128, 128, 128, {alpha})"

for i, gt in enumerate(gt_labels):
    for j, pred in enumerate(pred_labels_list):
        count = contingency_matrix.loc[gt, pred]
        if count > 0:
            source.append(i)
            target.append(len(gt_labels) + j)
            value.append(count)
            link_colors.append(hex_to_rgba(gt_node_colors[i], alpha=0.35))


fig = go.Figure(data=[go.Sankey(
    node = dict(
      pad = 15,
      thickness = 20,
      line = dict(color = "black", width = 0.5),
      label = node_labels,
      color = all_node_colors
    ),
    link = dict(
      source = source,
      target = target,
      value = value,
      color = link_colors
    ))])

fig.update_layout(
    title_text=f"",
    font_size=16,
    width=700,
    height=600,
    font=dict(
        family="Arial",
        size=18,
        color="black"
    ),
)

fig.write_html("./fig4_D_sankey.html")

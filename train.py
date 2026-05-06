import torch
import torch.nn.functional as F
from tqdm import tqdm
from model import SpaMEDM
import numpy as np
from preprocess import post_proC


class Train:
    def __init__(self, data, args=None):
        self.data = data
        self.datatype = args.data_type
        self.device = args.device
        self.arg = args

        self.X_omics1 = torch.FloatTensor(self.data[0]).to(self.device)
        self.X_omics2 = torch.FloatTensor(self.data[1]).to(self.device)

        self.adj_feature_omics1 = torch.FloatTensor(self.data[2]).to(self.device)
        self.adj_feature_omics2 = torch.FloatTensor(self.data[3]).to(self.device)

        self.adj_spatial_omics1 = torch.FloatTensor(self.data[4]).to(self.device)
        self.adj_spatial_omics2 = torch.FloatTensor(self.data[5]).to(self.device)

        self.edge_adj_index1 = self.adj_spatial_omics1.to_sparse()._indices().to(self.device)
        self.edge_adj_index2 = self.adj_spatial_omics2.to_sparse()._indices().to(self.device)

        self.dim_input1 = self.X_omics1.shape[1]
        self.dim_input2 = self.X_omics2.shape[1]

    def train(self):
        self.model = SpaMEDM(
            self.dim_input1, self.dim_input2,
            hidden_dim=128,
            out_dim=self.arg.dim_output,
            mask_rate=self.arg.mask,
            single=self.arg.single,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=5e-3)

        self.model.train()
        for epoch in tqdm(range(self.arg.epochs)):
            loss1, loss2, loss3, loss4 = self.model(self.X_omics1, self.X_omics2,
                              self.adj_spatial_omics1, self.adj_spatial_omics2, self.adj_feature_omics1, self.adj_feature_omics2,
                              self.edge_adj_index1, self.edge_adj_index2)
            loss = loss1 * self.arg.weight1 + loss2 * self.arg.weight2 + loss3 * self.arg.weight3 + loss4 * self.arg.weight4

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.model.eval()
        with torch.no_grad():
            emb, a, b, c = self.model(self.X_omics1, self.X_omics2,
                              self.adj_spatial_omics1, self.adj_spatial_omics2, self.adj_feature_omics1, self.adj_feature_omics2,
                              self.edge_adj_index1, self.edge_adj_index2)

        emb_norm = F.normalize(emb, p=2, dim=1)
        C_matrix = torch.mm(emb_norm, emb_norm.t())
        C_matrix = torch.relu(C_matrix).cpu().numpy()
        label, SVD_U = post_proC(C_matrix, K=self.arg.n_clusters)

        return label, SVD_U, emb_norm.detach().cpu().numpy(), a.detach().cpu().numpy(), b.detach().cpu().numpy(), c.detach().cpu().numpy()

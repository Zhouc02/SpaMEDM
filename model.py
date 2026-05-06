import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


def sce_loss(x, y, alpha=2):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1.0 - (x * y).sum(dim=-1)).pow_(alpha)
    return loss.mean()


def semantic_alignment(z1, z2):
    return 1.0 - F.cosine_similarity(z1, z2, dim=1).mean()


class GCNLayer(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(GCNLayer, self).__init__()
        self.gc1 = nn.Linear(in_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.gc2 = nn.Linear(hidden_dim, out_dim)
        self.ln2 = nn.LayerNorm(out_dim)
        self.act = nn.PReLU()

    def forward(self, x, adj):
        h = self.gc1(torch.matmul(adj, x))
        h = self.act(self.ln1(h))
        h = self.gc2(torch.matmul(adj, h))
        h = self.ln2(h)
        return h


class Attention(nn.Module):
    def __init__(self, hidden):
        super(Attention, self).__init__()
        self.w_omega = nn.Parameter(torch.Tensor(hidden, hidden))
        self.u_omega = nn.Parameter(torch.Tensor(hidden, 1))
        nn.init.xavier_uniform_(self.w_omega)
        nn.init.xavier_uniform_(self.u_omega)

    def forward(self, z1, z2):
        emb_stack = torch.stack([z1, z2], dim=1)
        v = torch.tanh(torch.matmul(emb_stack, self.w_omega))
        vu = torch.matmul(v, self.u_omega).squeeze(-1)
        alpha = F.softmax(vu, dim=1)
        z_fused = z1 * alpha[:, 0:1] + z2 * alpha[:, 1:2]
        if self.training:
            return z_fused
        else:
            return z_fused, alpha


class Denoise_Net(nn.Module):
    def __init__(self, latent_dim):
        super(Denoise_Net, self).__init__()
        self.time_emb = nn.Sequential(
            nn.Linear(1, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim)
        )
        self.enc1 = GCNConv(latent_dim, latent_dim, cached=False)
        self.dec1 = GCNConv(latent_dim, latent_dim, cached=False)

    def forward(self, z, edge_index, t):
        t_emb = self.time_emb(t.float().view(-1, 1))
        h1 = F.elu(self.enc1(z, edge_index) + t_emb)
        noise_pred = self.dec1(h1, edge_index)
        return noise_pred


class Diffusion_Process(nn.Module):
    def __init__(self, latent_dim, T=1000):
        super(Diffusion_Process, self).__init__()
        self.T = T
        self.net = Denoise_Net(latent_dim)
        self.register_buffer('beta', torch.linspace(1e-4, 0.02, T))
        self.register_buffer('alpha', 1. - self.beta)
        self.register_buffer('alpha_bar', torch.cumprod(self.alpha, dim=0))

    def forward(self, z, edge_index):
        t = torch.randint(0, self.T, (1,), device=z.device)
        a_bar_t = self.alpha_bar[t]
        noise = torch.randn_like(z)
        z_t = torch.sqrt(a_bar_t) * z + torch.sqrt(1 - a_bar_t) * noise
        noise_pred = self.net(z_t, edge_index, t)
        loss_diff = F.mse_loss(noise_pred, noise)
        return loss_diff


class SpaMEDM(nn.Module):
    def __init__(self, in_dim1, in_dim2, hidden_dim=128, out_dim=64, mask_rate=0.25, single=False):
        super(SpaMEDM, self).__init__()
        self.mask_rate = mask_rate
        self.single = single

        self.mask_token1 = nn.Parameter(torch.randn(1, in_dim1) * 0.1)
        self.mask_token2 = nn.Parameter(torch.randn(1, in_dim2) * 0.1)

        self.enc1_spa = GCNLayer(in_dim1, hidden_dim, out_dim)
        if not single:
            self.enc1_fea = GCNLayer(in_dim1, hidden_dim, out_dim)

        self.enc2_spa = GCNLayer(in_dim2, hidden_dim, out_dim)
        if not single:
            self.enc2_fea = GCNLayer(in_dim2, hidden_dim, out_dim)

        if not single:
            self.intra_attn1 = Attention(out_dim)
            self.intra_attn2 = Attention(out_dim)

        self.inter_attn = Attention(out_dim)

        self.diffusion1 = Diffusion_Process(out_dim)
        self.diffusion2 = Diffusion_Process(out_dim)


        self.dec1 = GCNLayer(out_dim, hidden_dim, in_dim1)
        self.dec2 = GCNLayer(out_dim, hidden_dim, in_dim2)

    def generate_mask(self, x):
        num_nodes = x.shape[0]
        perm = torch.randperm(num_nodes, device=x.device)
        return perm[:int(self.mask_rate * num_nodes)]

    def forward(self, x1, x2, adj_spa1, adj_spa2, adj_fea1, adj_fea2, edge_adj_index1, edge_adj_index2):
        if not self.training:
            z1_spa, z1_fea = self.enc1_spa(x1, adj_spa1), self.enc1_fea(x1, adj_fea1) if not self.single else 0
            z2_spa, z2_fea = self.enc2_spa(x2, adj_spa2), self.enc2_fea(x2, adj_fea2) if not self.single else 0

            if not self.single:
                z1_intra, z1_a = self.intra_attn1(z1_spa, z1_fea)
                z2_intra, z2_a = self.intra_attn2(z2_spa, z2_fea)
                z_fused_clean, z_a = self.inter_attn(z1_intra, z2_intra)
                return z_fused_clean, z1_a, z2_a, z_a
            else:
                z_fused_clean, z_a = self.inter_attn(z1_spa, z2_spa)
                return z_fused_clean, z_a, z_a, z_a

        z1_spa_c, z1_fea_c = self.enc1_spa(x1, adj_spa1), self.enc1_fea(x1, adj_fea1) if not self.single else 0
        z2_spa_c, z2_fea_c = self.enc2_spa(x2, adj_spa2), self.enc2_fea(x2, adj_fea2) if not self.single else 0

        if not self.single:
            z1_intra_c = self.intra_attn1(z1_spa_c, z1_fea_c)
            z2_intra_c = self.intra_attn2(z2_spa_c, z2_fea_c)

        loss_diff1 = self.diffusion1(z1_intra_c, edge_adj_index1) if not self.single else self.diffusion1(z1_spa_c, edge_adj_index1)
        loss_diff2 = self.diffusion2(z2_intra_c, edge_adj_index2) if not self.single else self.diffusion2(z2_spa_c, edge_adj_index2)
        loss_diff = loss_diff1 + loss_diff2

        mask_nodes1 = self.generate_mask(x1)
        mask_nodes2 = self.generate_mask(x2)
        x1_masked, x2_masked = x1.clone(), x2.clone()
        x1_masked[mask_nodes1] = self.mask_token1.repeat(mask_nodes1.shape[0], 1)
        x2_masked[mask_nodes2] = self.mask_token2.repeat(mask_nodes2.shape[0], 1)

        z1_spa_m, z1_fea_m = self.enc1_spa(x1_masked, adj_spa1), self.enc1_fea(x1_masked, adj_fea1) if not self.single else 0
        z2_spa_m, z2_fea_m = self.enc2_spa(x2_masked, adj_spa2), self.enc2_fea(x2_masked, adj_fea2) if not self.single else 0

        if not self.single:
            z1_intra_m = self.intra_attn1(z1_spa_m, z1_fea_m)
            z2_intra_m = self.intra_attn2(z2_spa_m, z2_fea_m)

        z_fused_m = self.inter_attn(z1_intra_m, z2_intra_m) if not self.single else self.inter_attn(z1_spa_m, z2_spa_m)

        rec1 = self.dec1(z_fused_m, adj_spa1)
        rec2 = self.dec2(z_fused_m, adj_spa2)

        loss_mae = sce_loss(rec1[mask_nodes1], x1[mask_nodes1]) + \
                   sce_loss(rec2[mask_nodes2], x2[mask_nodes2])

        loss_align = semantic_alignment(z1_intra_c, z2_intra_c) if not self.single else semantic_alignment(z1_spa_c, z2_spa_c)

        C_pred = torch.sigmoid(torch.mm(z_fused_m, z_fused_m.t()))
        loss_graph = F.mse_loss(C_pred, adj_spa1) + F.mse_loss(C_pred, adj_spa2)

        return loss_mae, loss_diff, loss_align, loss_graph

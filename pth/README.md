This folder stores the model weights

## Usage
You can use these weights files, for example:
```Python
self.model = SpaMEDM(
    self.dim_input1, self.dim_input2,
    hidden_dim=128,
    out_dim=self.arg.dim_output,
    mask_rate=self.arg.mask,
    single=self.arg.single,
).to(self.device)

# self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=5e-3)

# self.model.train()
# for epoch in tqdm(range(self.arg.epochs)):
#     loss1, loss2, loss3, loss4 = self.model(self.X_omics1, self.X_omics2,
#                       self.adj_spatial_omics1, self.adj_spatial_omics2, self.adj_feature_omics1, self.adj_feature_omics2,
#                       self.edge_adj_index1, self.edge_adj_index2)
#     loss = loss1 * self.arg.weight1 + loss2 * self.arg.weight2 + loss3 * self.arg.weight3 + loss4 * self.arg.weight4

#     self.optimizer.zero_grad()
#     loss.backward()
#     self.optimizer.step()

self.model.load_state_dict(torch.load('E15_5-S1.pth'))
self.model.eval()
```

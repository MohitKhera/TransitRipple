import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class TransitGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.gru = nn.GRU(hidden_channels, hidden_channels, batch_first=True)
        self.linear = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        time_steps = x.shape[0]
        spatial_outputs = []
        
        for t in range(time_steps):
            x_t = x[t]  # shape: [num_nodes, in_channels]
            # run through GCN layers
            x_t = self.conv1(x_t, edge_index)
            x_t = F.relu(x_t)
            x_t = self.conv2(x_t, edge_index)
            spatial_outputs.append(x_t)

        out = torch.stack(spatial_outputs, dim=1)
        out, _ = self.gru(out)
        out = out[:, -1, :]
        out = self.linear(out)
        return out


import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

with open('/content/drive/MyDrive/datset/before.txt', 'r') as file:
    lines = file.readlines()
csv_data = [line.strip().split('\t') for line in lines]
data = torch.tensor([[float(value) if value != "NaN" else float('nan') for value in row] for row in csv_data], dtype=torch.float32)

# Extract the x and y coordinates
x_o = data[:, 0]
y_o = data[:, 1]


x = data[:, 0]
y = data[:, 1]

missing_indices = torch.isnan(x)
x = x[~missing_indices]
y = y[~missing_indices]


edges = []
for i in range(1, len(x)):
    edges.append((i - 1, i))
edges = torch.tensor(edges, dtype=torch.long).t()

class TrajectoryGNN(nn.Module):
    def __init__(self):
        super(TrajectoryGNN, self).__init__()
        self.conv1 = GCNConv(2, 64)
        self.conv2 = GCNConv(64, 2)
        self.conv2 = GCNConv(64, 2)
        self.conv2 = GCNConv(64, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x


data = Data(x=torch.stack([x, y], dim=1), edge_index=edges)
model = TrajectoryGNN()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(1000):
    model.train()
    optimizer.zero_grad()
    output = model(data)
    # loss = torch.nn.functional.mse_loss(output, data.x)
    loss = torch.nn.functional.huber_loss(output, data.x)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")
completed_trajectory = output.detach().numpy()

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(x_o, y_o, marker='x', linestyle='-', color='red', markersize=6)
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Original Incomplete Trajectory')
plt.grid(True)

for i in range(1, len(completed_trajectory)):
    prev_point = np.array([completed_trajectory[i - 1], completed_trajectory[i - 1]])
    curr_point = np.array([completed_trajectory[i], completed_trajectory[i]])
    dist_prev = np.linalg.norm(prev_point)
    dist_curr = np.linalg.norm(curr_point)
    if dist_prev < dist_curr:
          plt.plot([x[i - 1], x[i]], [y[i - 1], y[i]], marker='x', color='red', linestyle='-')


plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Trajectory Segments')
plt.grid(True)
plt.tight_layout()
plt.show()




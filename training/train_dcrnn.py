import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Load synthetic dataset
data = np.load("data/synthetic_pems_bay.npz", allow_pickle=True)
X, y = data["x"], data["y"]

num_samples, timesteps, num_nodes, features = X.shape
input_size = num_nodes * features

# Reshape to 3D for GRU
X = X.reshape(num_samples, timesteps, input_size)
y = y.reshape(y.shape[0], y.shape[1], num_nodes * features)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

class DCRNN(nn.Module):
    def __init__(self, input_size, hidden_size=64, output_size=None):
        super(DCRNN, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size or input_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out)
        return out

model = DCRNN(input_size=input_size, hidden_size=64, output_size=input_size)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

for epoch in range(5):
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        preds = model(batch_x)
        preds = preds[:, -batch_y.shape[1]:, :]
        loss = loss_fn(preds, batch_y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

torch.save(model.state_dict(), "models/dcrnn_weights.pth")
print("✅ DCRNN training complete.")
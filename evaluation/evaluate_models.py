#import os
import numpy as np
import torch
import torch.nn as nn

from models.dcrnn_model import DCRNN
from models.stgcn_model import STGCN
from models.graph_wavenet_model import GraphWaveNet



# -----------------------------
# Load synthetic dataset
# -----------------------------
data = np.load("data/synthetic_pems_bay.npz", allow_pickle=True)
X, y = data["x"], data["y"]

num_samples, timesteps, num_nodes, features = X.shape
input_size = num_nodes * features

# Reshape to match training scripts
X = X.reshape(num_samples, timesteps, input_size)
y = y.reshape(y.shape[0], y.shape[1], num_nodes * features)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# -----------------------------
# Evaluation helper
# -----------------------------
def evaluate_model(model, weights_path):
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    with torch.no_grad():
        preds = model(X)
        preds = preds[:, -y.shape[1]:, :]  # match target sequence length
        mae = torch.mean(torch.abs(preds - y)).item()
        rmse = torch.sqrt(torch.mean((preds - y) ** 2)).item()
    return mae, rmse

# -----------------------------
# Instantiate models
# -----------------------------
dcrnn = DCRNN(input_size=input_size, hidden_size=64, output_size=input_size)
stgcn = STGCN(input_size=input_size, hidden_size=64, output_size=input_size)
graph_wavenet = GraphWaveNet(input_size=input_size, hidden_size=64, output_size=input_size)

# -----------------------------
# Evaluate all models
# -----------------------------
results = {
    "DCRNN": evaluate_model(dcrnn, "models/dcrnn_weights.pth"),
    "ST-GCN": evaluate_model(stgcn, "models/stgcn_weights.pth"),
    "Graph WaveNet": evaluate_model(graph_wavenet, "models/graph_wavenet_weights.pth"),
}

# -----------------------------
# Print results in table format
# -----------------------------
print("\n✅ Evaluation Results (Synthetic Dataset)\n")
print("{:<15} {:<10} {:<10}".format("Model", "MAE", "RMSE"))
print("-" * 35)
for model_name, (mae, rmse) in results.items():
    print("{:<15} {:<10.4f} {:<10.4f}".format(model_name, mae, rmse))
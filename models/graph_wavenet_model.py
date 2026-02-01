import torch
import torch.nn as nn

class GraphWaveNet(nn.Module):
    def __init__(self, input_size, hidden_size=64, output_size=None):
        super(GraphWaveNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=2)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size, output_size or input_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x = x.transpose(1, 2)  # (batch, input_size, seq_len)
        out = self.conv1(x)    # (batch, hidden_size, seq_len-1)
        out = self.relu(out)
        out = out.transpose(1, 2)  # (batch, seq_len-1, hidden_size)
        out = self.fc(out)         # (batch, seq_len-1, output_size)
        return out
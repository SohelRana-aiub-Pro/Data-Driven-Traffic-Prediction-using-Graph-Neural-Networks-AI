import torch
import torch.nn as nn

class DCRNN(nn.Module):
    def __init__(self, input_size, hidden_size=64, output_size=None):
        super(DCRNN, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size or input_size)

    def forward(self, x):
        out, _ = self.gru(x)  # (batch, seq_len, hidden_size)
        out = self.fc(out)    # (batch, seq_len, output_size)
        return out

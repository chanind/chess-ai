import torch.nn as nn
import torch.nn.functional as F

class ChessValueModel(nn.Module):
  def __init__(self, hidden_size = 128):
    super().__init__()
    self.layer1 = nn.Linear(901, hidden_size)
    self.bn1 = nn.BatchNorm1d(hidden_size)
    self.dropout1 = nn.Dropout(0.8)

    self.layer2 = nn.Linear(hidden_size, hidden_size)
    self.bn2 = nn.BatchNorm1d(hidden_size)
    self.dropout2 = nn.Dropout(0.8)

    self.layer3 = nn.Linear(hidden_size, hidden_size)
    self.bn3 = nn.BatchNorm1d(hidden_size)
    self.dropout3 = nn.Dropout(0.8)

    self.out_layer = nn.Linear(hidden_size, 1)

  def forward(self, x):
    x = F.relu(self.dropout1(self.bn1(self.layer1(x))))
    x = F.relu(self.dropout2(self.bn2(self.layer2(x))))
    x = F.relu(self.dropout3(self.bn3(self.layer3(x))))
    x = self.out_layer(x)

    # value output
    return F.tanh(x)
import torch.nn as nn
import torch

class Gru_model(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Gru_model, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
    
    def forward(self, x):
        output, hidden = self.gru(x)
        return hidden
        


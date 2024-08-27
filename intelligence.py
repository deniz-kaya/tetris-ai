import torch
import torch.nn as nn
import numpy as np

class DeepQNetwork(nn.Module):
    def __init__(self):
        super().__init__()

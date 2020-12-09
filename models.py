# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 08:43:01 2020

@author: Nathan Miguens & Hugues Raguenes
"""

import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, nb_layers, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim) 
        self.relu = nn.ReLU()
        self.layers = nn.ModuleList([ 
            nn.Linear(hidden_dim, hidden_dim) for _ in range(nb_layers)
            ])
        self.fc2 = nn.Linear(hidden_dim, output_dim)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        for layer in self.layers:
            out = layer(out)
            out = self.relu(out)
        out = self.fc2(out)
        return out
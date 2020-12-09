# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 00:29:09 2020

@author: miguens1u
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 17:32:39 2020

@author: Nathan Miguens & Hugues Raguenes

Contexte : Une entreprise essaie d'estimer le trafic d'un carrefour. Elle 
        enregistre pendant 2 ans ce trafic à l'aide d'un capteur. Par la suite,
        ce capteur est nécessaire pour une autre utilisation. Il doit donc être
        retirer du carrefour mais l'on souhaite toujours connaitre son trafic

Objectif : Estimer le trafic de ce carrefour avec les données qu'on récolte 
        ailleurs dans la ville.
        
        Précision : Trouver le trafic du lieu à un instant donné connaissant le trafic des
        autres lieux à ce même moment.
        
Choix du lieu :  BURNET RD / PALM WAY (IBM DRIVEWAY)

Inspired by : 
    - https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/01-basics/feedforward_neural_network/main.py#L37-L49

# 1er model : Multi Layer Perceptron
# 2nd model : CNNs
# 3rd model : LSTM
"""

import pickle
import torch
import torch.nn as nn
import pandas as pd

from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from tqdm import tqdm

from models import MLP

def importMeanStd(path):
    """Import mean std of volume"""
    ms = pd.DataFrame(pickle.load(open( path, "rb" )))
    mean = torch.tensor(ms["Volume"][0])
    std = torch.tensor(ms["Volume"][1])
    return mean, std

def getVectorX(train_indiv):
    """At first we considere nnly Volumes"""
    train = train_indiv[:,-1, 2:]
    labels = train_indiv[:,-1,:2] * std + mean
    return train, labels

def train(model, train_loader, lossF, losses, optimizer):
    """Train processing"""
    model.train()
    for iter, X in enumerate(train_loader):
        train, labels = getVectorX(X)
        # Forward pass 
        outputs = model(train)
        loss = lossF(outputs, labels)
        # Initializing a gradient as 0 so there is no mixing of gradient among the batches
        optimizer.zero_grad() 
        # Propagating the error backward
        loss.backward()
        # Optimizing the parameters
        optimizer.step()
        if iter%100 == 0:
            losses.append(loss)
            test(model, test_loader, accuracies)
            
def test(model, test_loader, accuracies):
    """Test processing"""
    model.eval()
    sumLoss = 0
    N = len(train_loader)
    with torch.no_grad():
        for X in train_loader:
            train, labels = getVectorX(X)
            outputs = model(train)
            sumLoss += 1 - (lossF(outputs, labels))/labels.abs().sum()
    accuracies.append(sumLoss/N)
    
# --------------------------------------------------------------------------- #
# Import data
# --------------------------------------------------------------------------- #

train_set = list(pickle.load(open("data/train_1.p", "rb" )))
test_set = list(pickle.load(open("data/test_1.p", "rb" )))

mean, std = importMeanStd("data/preprocessedData_meanStd.p")

input_dim = list(train_set[0].size())[1] - 2
output_dim = 2

# --------------------------------------------------------------------------- #
# Model and (hyper)parameters
# --------------------------------------------------------------------------- #

hidden_dim = 128
nb_layers = 5

learning_rate = 0.001
nb_epoch = 50
batch_size = 256

model = MLP(input_dim, hidden_dim, nb_layers, output_dim)
lossF = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle = True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle = True)
#val_loader = DataLoader(val_set, batch_size=100)

# --------------------------------------------------------------------------- #
# Train model
# --------------------------------------------------------------------------- #

losses = []
accuracies = []

for epoch in tqdm(range(nb_epoch)):
    train(model, train_loader, lossF, losses, optimizer)
    
plt.plot(losses)
plt.yscale('log')
plt.show()

plt.plot(accuracies)
plt.show()


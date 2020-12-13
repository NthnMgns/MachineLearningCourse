# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 00:29:09 2020

@author: miguens1u
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 17:32:39 2020

@author: Nathan Miguens & Hugues Raguenes

Contexte : 
        Une entreprise (ex: proposant des services de GPS et d'itinéraires) 
        veut estimer le trafic à différents carrefours de la ville, pour ainsi
        pouvoir proposer le trajet avec le moins de circulation.

Objectif : 
        Estimer le trafic à tous les carrefours à une date donnée, 
        connaissant les volumes de trafics antérieurs.

Choix des lieux :  ALL

# 1st model : Multi Layer Perceptron
# 2nd model : CNN
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


def getSequenceMLP(train_indiv):
    """Split in inputs and labels for MLP (input is a vector not a matrice)"""
    train = train_indiv[:,:-1, :].view(train_indiv.shape[0], -1)
    labels = train_indiv[:,-1, :] * std + mean
    return train, labels


def train(model, train_loader, lossF, losses, optimizer):
    """Train processing"""
    model.train()
    for iter, X in enumerate(train_loader):
        train, labels = getSequenceMLP(X)
        # Forward pass 
        outputs = model(train)
        loss = lossF(outputs, labels)
        # Initializing a gradient as 0 so there is no mixing of gradient among the batches
        optimizer.zero_grad() 
        # Propagating the error backward
        loss.backward()
        # Optimizing the parameters
        optimizer.step()
        if iter%10 == 0:
            losses.append(loss)
            test(model, test_loader, accuracies)
            
def test(model, test_loader, accuracies):
    """Test processing"""
    model.eval()
    sumLoss = 0
    N = len(train_loader)
    with torch.no_grad():
        for X in train_loader:
            train, labels = getSequenceMLP(X)
            outputs = model(train)
            sumLoss += 1 - torch.sqrt(lossF(outputs, labels))/labels.mean()
    accuracies.append(sumLoss/N)
    
# --------------------------------------------------------------------------- #
# Import data
# --------------------------------------------------------------------------- #

train_set = list(pickle.load(open("data/train_obj2.p", "rb" )))
test_set = list(pickle.load(open("data/test_obj2.p", "rb" )))

mean, std = importMeanStd("data/preprocessedData_meanStd.p")

size = list(train_set[0].size())
input_dim = (size[0] -1) * size[1] #all locations
output_dim = size[1] #all locations

# --------------------------------------------------------------------------- #
# Model and (hyper)parameters
# --------------------------------------------------------------------------- #

hidden_dim = 256
nb_layers = 5

learning_rate = 0.001
nb_epoch = 100
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

#x_train = train(model, train_loader, lossF, losses, optimizer)

for epoch in tqdm(range(nb_epoch)):
    train(model, train_loader, lossF, losses, optimizer)
    
plt.plot(losses)
#plt.yscale('log')
plt.show()

plt.plot(accuracies)
plt.show()


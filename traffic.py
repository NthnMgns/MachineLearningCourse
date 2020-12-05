# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 17:32:39 2020

@author: Nathan Miguens & Hugues Raguenes

Objectif : A définir
"""

import torch
import pandas as pd
import numpy as np
import pickle

from torch.utils.data import DataLoader

def loadData(path):
    """Load data from pickle serialized file"""
    return pd.DataFrame(pickle.load(open( path, "rb" )))

def splitData(data, train_ratio = [0.6, 0.5], random_state = 0):
    """Split data in train, test and valid sets"""
    train_set = data.sample(frac= train_ratio[0], random_state = random_state)
    test_set = data.drop(train_set.index)
    
    valid_set = test_set.sample(frac= train_ratio[1], random_state = random_state)
    test_set = test_set.drop(valid_set.index)
    
    return pandasToTorch([train_set, test_set, valid_set])

def pandasToTorch(dataFrames):
    """Transform pandas data to torch data"""
    tData = []
    for dataFrame in dataFrames :
        tData.append(DataLoader(dataFrame.values, batch_size=512))
    return tData


    
if __name__ == '__main__':
    data = loadData("data/preprocessedData.p")
    train_set, test_set, valid_set = splitData(data)
    

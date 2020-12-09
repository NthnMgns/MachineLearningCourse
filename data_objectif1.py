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


# 1er essai : 
"""
#-------------------------------# Variables #---------------------------------#

station_name1 = "700 BLK E CESAR CHAVEZ ST EB"
station_name2 = "700 BLK E CESAR CHAVEZ ST WB"

#-----------------------------------------------------------------------------#

import torch
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

def loadData(path):
    """Load data from pickle serialized file"""
    return pd.DataFrame(pickle.load(open( path, "rb" )))

def names_dict(data):
    """Create a dictionnaire of location names"""
    names = data["location_name"].unique()
    name_dict = {name : iter for iter, name in enumerate(names)}
    for iter, name in enumerate(name_dict) :
        if iter == 0 :
            name_dict.pop(name)
            name_dict[name] = name_dict[station_name1]
        elif iter == 1 :
            name_dict.pop(name)
            name_dict[name] = name_dict[station_name2]
        else : 
            break
    name_dict[station_name1] = 0
    name_dict[station_name2] = 1
    return name_dict

def sampleTime(data):
    """Return the time where data exist for our station"""
    timeSample = data.where((data["location_name"] == station_name1) | (data["location_name"] == station_name2))["Full Time"]
    timeSample.drop_duplicates(inplace=True)
    timeSample = timeSample.sort_values()
    return timeSample.reset_index(drop=True)

def dropUseless(data):
    """Delete useless features"""
    uselessFeatures = ['Month','Day','Day of Week', 'Year', 'Minute', 'Hour', 'Time']
    for feat in uselessFeatures :
        data.drop(feat, axis=1, inplace=True)
    return data

def featuresEncoding(data, timeSample, name_dict):
    """Encode informations from data in a correct format :
        - All futur vectors with same dimensions
        - Location_name become index of vectors 
        - Columns 0 and 1 corresponding to labels
        - Other columns are the observations """
    X = []
    for time in tqdm(timeSample) :
        if str(time) != "nan" :
            temp = data.loc[data["Full Time"] == time]
            temp = temp.sort_values(by = ["location_name"])
            temp = temp.set_index(["location_name"])
            temp = temp.reindex(pd.RangeIndex(len(name_dict))).fillna(0.)
            temp = dropUseless(temp)
            X.append(torch.tensor(temp.T.values,dtype=torch.float32))
    return X

def splitData(data):
    """Split data in train, test and valid sets"""
    timeSample = sampleTime(data)
    name_dict = names_dict(data)
    data["location_name"] = data["location_name"].map(name_dict)
    X = featuresEncoding(data, timeSample, name_dict)
    train_set, test_set = train_test_split(X, test_size=0.4, shuffle=False)
    return train_set, test_set

def pandasToTorch(dataFrames):
    """Transform pandas data to torch data"""
    tData = []
    for dataFrame in dataFrames :
        tData.append(DataLoader(dataFrame.values, batch_size=512))
    return tData

if __name__ == '__main__':
    data = loadData("data/preprocessedData.p")
    train_set, test_set = splitData(data)
    pickle.dump(train_set, open( "data/train_2.p", "wb" ))
    pickle.dump(test_set, open( "data/test_2.p", "wb" ))
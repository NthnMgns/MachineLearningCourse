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

"""

#-----------------------------------------------------------------------------#

import torch
import pandas as pd
import numpy as np
import pickle

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def loadData(path):
    """Load data from pickle serialized file"""
    return pd.DataFrame(pickle.load(open( path, "rb" )))

def names_dict(data):
    """Create a dictionnaire of location names"""
    names = data["location_name"].unique()
    name_dict = {name : iter for iter, name in enumerate(names)}
    return name_dict

def detectTimerange(data):
    """Detect the minimal subset of dates and filter the dataset by it"""
    aggdata = data.groupby(['location_name']).agg({'Full Time': join_time})
    n = aggdata.shape[0]
    keep_timerange = [aggdata.iloc[i,0] for i in range(n) if len(aggdata.iloc[i,0])>1]
    keep_timerange.sort(key=len, reverse=True)
    intersect = set.intersection(*keep_timerange[0:33])
    data = data[data['Full Time'].isin(list(intersect))]
    print(f'{len(intersect)} shared dates detected / {len(data.location_name.unique())} locations selected')
    return data

def join_time(times):
    return set(times)

def dropUseless(data):
    """Delete useless features"""
    uselessFeatures = ['Month','Day','Day of Week', 'Year', 'Minute', 'Hour', 'Time']
    for feat in uselessFeatures :
        data.drop(feat, axis=1, inplace=True)
    return data

def featuresEncoding(data, name_dict):
    """Encode informations from data in a correct format :
        - All futur vectors with same dimensions
        - Location_name become index of vectors 
        - Columns 0 and 1 corresponding to labels
        - Other columns are the observations """
    X = []
    timeSample =  data['Full Time'].unique()
    for time in tqdm(timeSample):
        temp = data.loc[data["Full Time"] == time]
        temp = temp.sort_values(by = ["location_name"])
        temp = temp.set_index(["location_name"])
        temp = temp.reindex(pd.RangeIndex(len(name_dict))).fillna(0.)
        temp = dropUseless(temp)
        # print(temp.T.values)
        X.append(torch.tensor(temp.T.values,dtype=torch.float32))
    return X

def splitData(data):
    """Split data in train, test and valid sets"""
    data = detectTimerange(data)
    name_dict = names_dict(data)
    data["location_name"] = data["location_name"].map(name_dict)
    X = featuresEncoding(data, name_dict)
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
    # pickle.dump(train_set, open( "data/train_obj2.p", "wb" ))
    # pickle.dump(test_set, open( "data/test_obj2.p", "wb" ))

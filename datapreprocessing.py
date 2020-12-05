# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 19:51:14 2020

@author: miguens1u

Inspired by : https://towardsdatascience.com/all-about-categorical-variable-encoding-305f3361fd02
and : https://towardsdatascience.com/cyclical-features-encoding-its-about-time-ce23581845ca
"""
import pandas as pd
import numpy as np
import pickle

def preprecessing(file = "data/Radar_Traffic_Counts.csv"):
    """preprocessing main function"""
    data = pd.read_csv(file, header = 0, nrows = 1000)
    data = encodeDirection(data)
    data = encodeTime(data)
    data = aggregateVolume(data)
    data = dropUseless(data)
    serializationData(data)
    return data

def encodeDirection(data):
    """Transform direction string into continuous variables"""
    directionX = {"EB" : 1., "WB" : -1., "SB" : 0., "NB" : 0., "None" : 0.}
    directionY = {"EB" : 0., "WB" : 0., "SB" : -1., "NB" : 1., "None" : 0.}
    data["direction_x"] = data.Direction.map(directionX)
    data["direction_y"] = data.Direction.map(directionY)
    return data.drop(['Direction'], axis=1)

def encodeTime(data):
    """Transform Year, Month, Day, Hours (etc...) variables into continuous variables"""
    data["Year"] = data.Year.astype('float')
    data["Month"] = data.Month.astype('float')
    data["Day"] = data.Day.astype('float')
    data["Day of Week"] = data["Day of Week"].astype('float')
    data["Hour"] = data["Hour"].astype('float')
    data["Minute"] = data["Minute"].astype('float')
    
    data["Time"] = 60*data["Hour"]+data["Minute"]
    
    time_spec = {"Month" : 12., "Day" : 30., "Day of Week" : 7., "Time" : 1440.}
    time_feature = list(time_spec.keys())
    for feature in time_feature:
        data[feature+"_norm"] = 2*np.pi*data[feature]/time_spec[feature]
        data["cos_"+feature] = np.cos(data[feature+"_norm"])
        data["sin_"+feature] = np.sin(data[feature+"_norm"])
        data.drop(feature+"_norm", axis=1, inplace=True)
    return data

def aggregateVolume(data):
    """Sum the volumes of rows describing the same event"""
    aggdata = data.groupby(['location_name','Year','Month','Day','Time Bin', "direction_x", "direction_y"]).agg({'Volume':'sum'})
    data.drop(["Volume"], axis=1, inplace=True)
    data = data.drop_duplicates(subset=['location_name','Year','Month','Day','Time Bin', "direction_x", "direction_y"])
    return pd.merge(data, aggdata, on=['location_name','Year','Month','Day','Time Bin', "direction_x", "direction_y"])

def dropUseless(data):
    """Delete useless features"""
    uselessFeatures = ['location_name', 'Month','Day','Day of Week', 'Hour', 'Minute', 'Time', 'Time Bin']
    for feat in uselessFeatures :
        data.drop(feat, axis=1, inplace=True)
    return data

def serializationData(data):
    """Save data with Pickle library"""
    pickle.dump(data, open( "data/preprocessedData.p", "wb" ))
    return 
if __name__ == '__main__':
    data = preprecessing()
    #print(data.head())
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 19:51:14 2020

@author: Nathan Miguens & Hugues Raguenes

Inspired by : 
    - https://towardsdatascience.com/all-about-categorical-variable-encoding-305f3361fd02
    - https://towardsdatascience.com/cyclical-features-encoding-its-about-time-ce23581845ca
    - https://towardsdatascience.com/data-normalization-with-pandas-and-scikit-learn-7c1cc6ed6475

"""

import pandas as pd
import numpy as np
import pickle

def preprecessing(file = "data/Radar_Traffic_Counts.csv", save = True):
    """preprocessing main function"""
    data = pd.read_csv(file, header = 0)#, nrows = 100000)
    data = encodeDirection(data)
    data = encodeTime(data)
    data = aggregateVolume(data)
    #data = dropUseless(data)
    data, meanStd = scaleData(data)
    if save :
        serializationData(data, meanStd)
    return data

def encodeDirection(data):
    """Transform direction string into continuous variables"""
    data["location_name"] += " " + data['Direction']
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
    month = {1:0,2:31,3:59,4:90,5:120,6:151,7:181,8:212,9:243,10:273,11:304,12:334}
    data["Full Time"] = (365 * (2020 - data["Year"]) + data.Month.map(month) + data["Day"]) * 24 * 60 + data["Time"]
    
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
    aggdata = data.groupby(['location_name','Year','Month','Day','Time Bin']).agg({'Volume':'sum'})
    data.drop(["Volume"], axis=1, inplace=True)
    data = data.drop_duplicates(subset=['location_name','Year','Month','Day','Time Bin'])
    return pd.merge(data, aggdata, on=['location_name','Year','Month','Day','Time Bin'])

def dropUseless(data):
    """Delete useless features"""
    uselessFeatures = ['Month','Day','Day of Week', 'Hour', 'Minute', 'Time', 'Time Bin']
    for feat in uselessFeatures :
        data.drop(feat, axis=1, inplace=True)
    return data

def serializationData(data, meanStd):
    """Save data with Pickle library"""
    pickle.dump(data, open( "data/preprocessedData.p", "wb" ))
    pickle.dump(meanStd, open( "data/preprocessedData_meanStd.p", "wb" ))
    return 

def scaleData(data):
    """ - create a dataFrame to save mean, std of all features
        - Normalize data
    """
    data.drop("Time Bin", axis =1, inplace = True)
    meanStd = pd.DataFrame()
    for feature in data.columns:
        if feature != 'location_name':
            m, std = data[feature].mean(), data[feature].std()
            data[feature] = (data[feature] - m)/std
            meanStd[feature] = {'mean':m, 'std' : std}
    return data, meanStd

if __name__ == '__main__':
    data = preprecessing()
    print(data.head())
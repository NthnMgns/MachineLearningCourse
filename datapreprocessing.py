# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 19:51:14 2020

@author: miguens1u

Inspired by : https://towardsdatascience.com/all-about-categorical-variable-encoding-305f3361fd02
"""
import pandas as pd

def preprecessing(file = "data/Radar_Traffic_Counts.csv"):
    """preprocessing main function"""
    data = pd.read_csv(file, header = 0, nrows = 1000)
    data = encodeDirection(data)
    data = encodeTime(data)
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
    return data

if __name__ == '__main__':
    data = preprecessing()
    print(data.head())
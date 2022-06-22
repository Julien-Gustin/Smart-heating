import numpy as np
import pandas as pd
import os 
from python.data import rooms

def room_as_index(room_name: str):
    for i in range(len(rooms)):
        if room_name == rooms[i]:
            return i 
    return -1
    

def RMSE(y_pred, y):
    return np.math.sqrt(sum(((y_pred - y)**2)/len(y)))


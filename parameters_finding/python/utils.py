import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from data import get_data
from functools import reduce

def RMSE(y_pred, y):
    return np.math.sqrt(sum(((y_pred - y)**2)/len(y)))

def get_X_y_of(rooms: list, dataset:pd.DataFrame):
    X = []

    for room in rooms:
        X.append(dataset["T_m_" + room][:-1])

    for room in rooms:
        X.append(dataset["T_set_" + room][1:])
    
    for room in rooms:
        X.append(dataset["switch_" + room][1:])

    X.append(dataset["T_m_ext"][:-1])
    X.append(dataset["T_m_water"][1:])
    X.append(dataset["Irradiation"][1:])
    X.append(dataset["Occupancy"][1:])

    # Predict t+1 given t
    y = np.array([dataset["T_m_" + room][1:] for room in rooms]) # shift
    X = np.array(X)
    y0 = np.array([dataset["T_m_" + room][0] for room in rooms]) # shift

    return X, y, y0

def load_data(rooms):
    data = get_data()
    return get_X_y_of(rooms, data), data["Datetime"]
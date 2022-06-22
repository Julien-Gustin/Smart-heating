import pandas as pd 
import numpy as np 
from python.room_init import rooms

def get_data():
    data = pd.read_csv("data/data.csv", delimiter=";")
    return data

def load_dataset(dataset:pd.DataFrame):
    column_names = []
    for room in rooms.keys():
        column_names.append("T_m_" + room)

    for room in rooms.keys():
        column_names.append("T_set_" + room)
    
    for room in rooms.keys():
        column_names.append("switch_" + room)

    column_names.append("T_m_ext")
    column_names.append("T_m_water")
    column_names.append("Irradiation")
    column_names.append("Occupancy")

    y = {}
    for room in rooms.keys():
        y[room] = np.array(dataset["T_m_" + room][1:])
    
    X = {}
    for column_name in column_names:
        X[column_name] = np.array(dataset[column_name][:-1])

    y0 = {}
    for room in rooms.keys():
        y0[room] = dataset["T_m_" + room][0]

    datetime = dataset["Datetime"]

    return X, y, y0, datetime


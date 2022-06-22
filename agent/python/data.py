import os
import pandas as pd
from functools import reduce

rooms = ["Kitchen", "Dining", "Living", "Room1", "Bathroom", "Room2", "Room3"]

def get_data():
    data = pd.read_csv(os.path.join("data", "data.csv"), delimiter=";")
    return data


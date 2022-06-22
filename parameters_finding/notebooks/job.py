import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

sys.path.insert(0, '../python/')

from multi_zone import MultiZoneRegression
from utils import RMSE, load_data

import warnings
warnings.filterwarnings('ignore')

rooms = ["Dining", "Kitchen", "Living", "Room1", "Bathroom", "Room2", "Room3"]
(X, y, y0), time = load_data(rooms)
area = [3.94*4.19, 3.87*4.19, 5.53*8, 4.93*4.19, 4*3.88, 5.53*4, 3.15*4]

X_train = X
y_train = y 

X_test = X.T
y_test = y

fit = True
house = np.array([[False, True, False, True, False, False, False],
                [True, False, False, False, True, False, False],
                [False, False , False, False, False, False, False],
                [True, False, False, False, True, False, False],
                [False, True, False, True, False, False, False],
                [False, False, False, False, False, False, True],
                [False, False, False, False, False, True, False]
                ])

# multi_zone = MultiZoneRegression(house, area, initial_values)

min_rmse = 1e10
min_init = []
min_zone = None

for j in range(20):
    try:
        R = np.random.uniform(low=1000000*1e-3, high=1000000*1e-1, size=7)/1000000
        Rab = np.random.uniform(low=1000000*1e-2, high=1000000*10, size=6)/1000000
        C = np.random.uniform(low=1000000*1e5, high=1000000*1e6, size=7)/1000000
        M = np.random.uniform(low=1000000*1e-5, high=1000000*1e-2, size=7)/1000000
        A = np.random.uniform(low=1000000*1e-5, high=1000000*20.0, size=7)/1000000
        K = np.random.uniform(low=1000000*0, high=1000000*1e-6, size=7)/1000000
        GS = np.random.uniform(low=1000000*1e-3, high=1000000*1e-2, size=7)/1000000

        C1 = list(np.concatenate([R[:4], C[:4], M[:4], A[:4], K[:4], GS[:4], Rab[:4]]))
        C2 = list(np.concatenate([R[4:5], C[4:5], M[4:5], A[4:5], K[4:5], GS[4:5]]))
        C3 = list(np.concatenate([R[5:7], C[5:7], M[5:7], A[5:7], K[5:7], GS[5:7], Rab[4:6]]))

        initial_values = [C1, C2, C3]

        multi_zone = MultiZoneRegression(house, area, initial_values)
        multi_zone.fit(X_train, y_train, area, fit=True)

        y_pred = multi_zone.predict_all(X_test, y0)

        total_rmse = 0

        for i in range(7):
            rmse = RMSE(y_pred[i], y_test[i])
            total_rmse += rmse

        if total_rmse < min_rmse:
            min_zone = multi_zone
            min_rmse = total_rmse
            min_init = initial_values
            print(min_rmse, j)
            for i, room in enumerate(min_zone.rooms):
                print("{} = multi_zone.rooms[{}]".format(rooms[i], i))
                print("{}.R = {}".format(rooms[i],room.R))
                print("{}.C = ".format(rooms[i]), room.C,)
                print("{}.m = ".format(rooms[i]), room.m,)
                print("{}.a = ".format(rooms[i]), room.a,)
                print("{}.k = ".format(rooms[i]), room.k,)
                print("{}.gs = ".format(rooms[i]), room.gs,)
                print("{}.Rab = ".format(rooms[i]), room.Rab, "\n")

    except:
        print("Error")

print(min_init)


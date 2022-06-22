import pandas as pd

from python.room_temperature_predictor import Cluster, RoomTemperaturePredictor

def load_rooms(param_csv):
    df_parameters = pd.read_csv(param_csv, delimiter=";")
    rooms_in_file = {}
    for index, parameter_row in df_parameters.iterrows():
        rooms_in_file[parameter_row["name"]] = RoomTemperaturePredictor(
                                        parameter_row["name"], 
                                        parameter_row["area"],
                                        parameter_row["R"],
                                        parameter_row["C"],
                                        parameter_row["m"],
                                        parameter_row["a"],
                                        parameter_row["k"],
                                        parameter_row["gs"])
    return rooms_in_file

def init_cluster(cluster_file):
    cluster_file = open(cluster_file, "r")
    cluster_file = cluster_file.read()
    lines = cluster_file.split("\n")

    room_names = lines[0].split(" ")
    cluster_rooms = [rooms[room_name] for room_name in room_names]

    cluster_R_links = []

    for link in lines[1:]:
        link = link.split(" ")
        cluster_R_links.append((link[0], link[1], float(link[2])))
    
    cluster = Cluster(cluster_rooms, cluster_R_links)
    return cluster

#Global variable representing all the rooms in the thermal model as (key,value) pairs
rooms = load_rooms("data/rooms.csv")
cluster_1 = init_cluster("data/cluster1.txt")
cluster_2 = init_cluster("data/cluster2.txt")
cluster_3 = init_cluster("data/cluster3.txt") #il est useless mais j'ai eu un peu la flemme 




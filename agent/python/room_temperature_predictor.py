import numpy as np
import pandas as pd

#constants
const_dt = 900
const_cp = 4810

class Cluster:
    #rooms: list of RoomTemperaturePredictor instances
    #R_links: list of (room_name1, room_name2, R_room_name1_room_name2) instances
    def __init__(self, rooms: list, R_links: list):
        self._R_links = R_links
        self._rooms = {}

        for room in rooms:
            self._rooms[room.name()] = room            
            room.assignCluster(self)

    def get_R_link(self, room_name1, room_name2):
        for R_link in self._R_links:
            link1, link2, R = R_link
            link = [link1, link2]
            if room_name1 in link and room_name2 in link:
                return R
        return None

    #adjacent_rooms: list of (room_name, prev_temperature) instances
    def getThermalInfluence(self, room_name: str, room_temperature: float, room_capacity: float, adjacent_rooms: list):
        total_thermal_influence = 0.0
        for adjacent_room in adjacent_rooms:
            adjacent_room_name, adjacent_room_temperature = adjacent_room
            R_link = self.get_R_link(room_name, adjacent_room_name)

            if R_link is not None:
                thermal_influence = (room_temperature - adjacent_room_temperature) / (R_link * room_capacity)
                total_thermal_influence += thermal_influence

        return total_thermal_influence

    def room_names(self):
        return self._rooms.keys()

    def simulate(self,
        init_temperature: dict,
        water_temperature: list,
        ambient_temperature: list,
        radiator_switch: dict,
        irradiance: list,
        occupancy: list, 
    ) -> float:

        prev_temperature = {}
        radiator_temperature = {}
        Q_heat = {}

        current_temperature = {}
        predicted_temperature = {}

        for room_name in self._rooms.keys():
            prev_temperature[room_name] = init_temperature[room_name]
            radiator_temperature[room_name] = 0.0
            Q_heat[room_name] = 0.0
            predicted_temperature[room_name] = list()

        for i in range(len(water_temperature)): #using the length of one of the list to know how many predictions must be made
            #apply an iteration prediction for each room of the cluster
            for room_name in self._rooms.keys():
                room = self._rooms[room_name]

                adjacent_rooms_temperature = prev_temperature.copy()
                adjacent_rooms_temperature.pop(room_name)
                adjacent_rooms_temperature = [(key, adjacent_rooms_temperature[key]) for key in adjacent_rooms_temperature.keys()]

                current_temperature[room_name], radiator_temperature[room_name], Q_heat[room_name] = room.predict(prev_temperature[room_name],
                                                                                                               water_temperature[i],
                                                                                                               ambient_temperature[i],
                                                                                                               radiator_switch[room_name][i],
                                                                                                               irradiance[i],
                                                                                                               occupancy[i],
                                                                                                               radiator_temperature[room_name],
                                                                                                               Q_heat[room_name],
                                                                                                               adjacent_rooms_temperature)
                predicted_temperature[room_name].append(current_temperature[room_name])

            #prev/cur maintained because of the adjacent rooms
            prev_temperature = current_temperature.copy()
        return predicted_temperature


#Class that simulates the temperature evolution of a room
class RoomTemperaturePredictor:
    def __init__(self, name: str, area: float, R: float, C: float, m: float, a: float, k: float, gs: float):
        self._cluster = None 
        self._name = name 
        self._area = area
        self._R = R
        self._C = C
        self._m = m
        self._a = a
        self._k = k
        self._gs = gs

    def name(self):
        return self._name
        
    #Assign a cluster to a room
    #Raise an exception if the room is already assigned to a cluster
    def assignCluster(self, cluster: Cluster):
        if self._cluster is not None:
            raise Exception("This room is already in a cluster")
        self._cluster = cluster 

    #adjacent_rooms_prev_temperature: list of (room_name, prev_temperature) instances
    def predict(self,
        prev_temperature: float,
        water_temperature: float,
        ambient_temperature: float,
        radiator_switch: bool,
        irradiance: float,
        occupancy: float, 
        prev_radiator_temperature: float, 
        prev_Q_heat: float,
        adjacent_rooms_prev_temperature: list
    ) -> float:

        if radiator_switch:
            #the radiator is ON
            radiator_temperature = water_temperature
        else:
            #estimate the radiator loss
            radiator_temperature = prev_radiator_temperature - const_dt * self._k * prev_Q_heat
        
        Q_sol = self._area * irradiance * self._gs
        Q_heat = const_cp * self._m * (radiator_temperature-prev_temperature)
        Q_heat = max(Q_heat, 0)
        Q_int = self._a*occupancy

        predicted_temperature = (ambient_temperature-prev_temperature) / (self._R * self._C)
        predicted_temperature += (Q_heat + Q_sol + Q_int) / self._C

        if self._cluster is not None:
            thermal_influence = self._cluster.getThermalInfluence(self._name, prev_temperature, self._C, adjacent_rooms_prev_temperature)
            predicted_temperature -= thermal_influence

        predicted_temperature = predicted_temperature * const_dt + prev_temperature

        return predicted_temperature, radiator_temperature, Q_heat





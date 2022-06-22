import pandas as pd
import numpy as np
from pyomo.environ import *
import sys
if sys.version_info[0] < 3: 
    from StringIO import StringIO
else:
    from io import StringIO

# Big-M parameters
M = 100

# Constants
const_dt = 900
const_cp = 4810
rooms_names = ["Kitchen", "Dining", "Living", "Room1", "Bathroom", "Room2", "Room3"]

def room_as_index(room_name: str):
    for i in range(len(rooms_names)):
        if room_name == rooms_names[i]:
            return i 
    return -1

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


def load_rooms(param_csv):
    df_parameters = pd.read_csv(StringIO(param_csv), delimiter=";")
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
rooms = load_rooms("name;area;R;C;m;a;k;gs\nDining;16.5086;0.08705215583493366;1316623.9101518856;0.0053241419909227615;0.012963508154211736;3.8501678672604e-05;0.003974463503095047\nKitchen;16.215300000000003;0.05511557713764909;2811551.4088770924;0.011111375897285449;4.046409843185994;1.2442186439280044e-05;0.00766535429152145\nLiving;44.24;0.035239438327118615;2287354.4863172104;0.006394006048548277;2.2412215391892496;5.25787058645177e-05;0.0018003693401619496\nRoom1;20.6567;0.05912584436141776;4387092.741697961;0.00902658804998515;30.05523408968866;2.5565690793273325e-05;0.007675218805971147\nBathroom;15.52;0.022874512518973027;1482003.4916086262;0.011127769189433551;8.176050677705527;1.2688267487536067e-05;0.013658242381980096\nRoom2;22.12;0.04810725434440728;8491065.220857458;0.01067938957624795;76.29331197663618;1.532417229708552e-05;0.01329970752206089\nRoom3;12.6;0.018425159385894996;10259884.23196018;0.0029679146022725334;144.51048265097316;6.905486302378876e-07;0.04391116167853753")
cluster_1 = init_cluster("Dining Kitchen Room1 Bathroom\nDining Kitchen 5.84805019e+02\nDining Room1 0.05279355\nKitchen Bathroom 0.00877792\nRoom1 Bathroom 0.01007886")
cluster_2 = init_cluster("Room2 Room3\nRoom2 Room3 0.00970442")
cluster_3 = init_cluster("Living")

class OptimalHeatingController:

    def __init__(self, alpha=1e-4, beta=2, omega=1):
        self.alpha = alpha
        self.beta = np.ones(7) * beta
        self.omega = np.ones(7) * omega

        self.prec_Tr = np.zeros(7)
        self.prec_Q = np.zeros(7)

        self.beta[5] = 1
        self.omega[5] = 1
    
    def model_cluster(self, model: ConcreteModel, cluster: Cluster, set_temp, outside_temp,
                        forecast_irr, forecast_occ):

        for name in cluster.room_names():
            room = cluster._rooms[name]
            model = self.model_room(model, room, set_temp)
            
            Q_int = room._a * np.array(forecast_occ)
            Q_sol = room._area * np.array(forecast_irr) * room._gs

            index = room_as_index(name)
            
            influence = [] #list of (index, R) pairs

            for other_name in cluster._rooms:
                R = cluster.get_R_link(name, other_name)
                if R is not None:
                    influence.append((room_as_index(other_name), R))
                              
            for t in range(self.control_horizon):
                # Temperature of the zone
                model.constraints.add(model.Tz[index,t+1] == ((outside_temp[t] - model.Tz[index, t]) / (room._R * room._C) 
                                                                + (Q_int[t] + Q_sol[t] + model.Q_heat[index, t]) / room._C
                                                                -sum((model.Tz[index, t]-model.Tz[neighbor_index, t])/ (R_link*room._C) for neighbor_index, R_link in influence)
                                                                ) * const_dt + model.Tz[index,t])
        return model 
                 
    def model_room(self, model: ConcreteModel, room: RoomTemperaturePredictor, set_temp):
        index = room_as_index(room.name())
        
        for t in range(self.control_horizon):
            # Temperature of the radiator
            model.constraints.add(model.Tr[index, t+1] <= model.Tw[t+1] + M * (1 - model.Switch[index, t+1]))
            model.constraints.add(model.Tr[index, t+1] >= model.Tw[t+1] - M * (1 - model.Switch[index, t+1]))

            model.constraints.add(model.Tr[index, t+1] >= model.Tr[index, t] - const_dt * room._k * model.Q_heat[index, t] - M * (model.Switch[index, t+1]))
            model.constraints.add(model.Tr[index, t+1] <= model.Tr[index, t] - const_dt * room._k * model.Q_heat[index, t] + M * (model.b1[index, t+1] + model.Switch[index, t+1]))

            model.constraints.add(model.Tr[index, t+1] >= model.Tz[index, t+1])
            model.constraints.add(model.Tr[index, t+1] <= model.Tz[index, t+1] + M * (1 - model.b1[index, t+1]) + M * model.Switch[index, t+1])

            # Q_heat
            model.constraints.add(model.Q_heat[index, t+1] == const_cp * room._m * (model.Tr[index, t+1] - model.Tz[index, t+1])) 

            # x_diff = |Tz - set_temp|
            model.constraints.add(model.x_diff[index, t+1] >= self.omega[index] * (model.Tz[index, t+1] - set_temp[index][t+1])) # orange
            model.constraints.add(model.x_diff[index, t+1] >= self.beta[index] * (set_temp[index][t+1] - model.Tz[index, t+1])) # red
        
        return model 


    def init_variables(self, model, prev_rooms_temp):
        model.constraints.add(model.Tw[0] == 0)
        for room in range(7):
            model.constraints.add(model.Tz[room, 0] == prev_rooms_temp[room])
            model.constraints.add(model.Tr[room, 0] == self.prec_Tr[room])
            model.constraints.add(model.Q_heat[room, 0] == self.prec_Q[room])

        return model

    def save_variables(self, model):
        for room in range(7):
            self.prec_Tr[room] = model.Tr[room, 1].value
            self.prec_Q[room] = model.Q_heat[room, 1].value

        return model

        
    def make_model(self, control_horizon, prev_rooms_temp, set_temp, outside_temp,
                        forecast_irr, forecast_occ):
        model = ConcreteModel(doc="squid game")

        model.nb_rooms = RangeSet(0, 6)
        model.control_horizon = RangeSet(0, control_horizon) 
        
        model.Tw = Var(model.control_horizon, domain=NonNegativeReals, bounds=(0, 70))
        model.Tz = Var(model.nb_rooms, model.control_horizon, domain=NonNegativeReals, bounds=(0, 100))
        model.Tr = Var(model.nb_rooms, model.control_horizon, domain=NonNegativeReals, bounds=(0, 70))
        model.x_diff = Var(model.nb_rooms, model.control_horizon, domain=NonNegativeReals)
        model.Q_heat = Var(model.nb_rooms, model.control_horizon)
        model.Switch = Var(model.nb_rooms, model.control_horizon, domain=Binary)
        model.b1 = Var(model.nb_rooms, model.control_horizon, domain=Binary)

        model.constraints = ConstraintList()
        
        model = self.init_variables(model, prev_rooms_temp)

        model = self.model_cluster(model, cluster_1, set_temp, outside_temp, forecast_irr, forecast_occ)
        model = self.model_cluster(model, cluster_2, set_temp, outside_temp, forecast_irr, forecast_occ)
        model = self.model_cluster(model, cluster_3, set_temp, outside_temp, forecast_irr, forecast_occ)

        model.o = Objective(expr=sum(1e-8 * model.Tw[t] for t in model.control_horizon) + sum(model.x_diff[i, t] + self.alpha * (model.Q_heat[i, t] / 1000) * 0.25 * 0.06 / 0.95 for t in model.control_horizon for i in model.nb_rooms), sense=minimize)
        return model 

    def compute_actions(self, control_horizon, prev_rooms_temp, set_temp, forecast_outside_temp, forecast_ground_temp,
                        forecast_irr, forecast_occ):
        """
        A function that determines the control decisions for the optimal heating of the house and returns the
        temperature settings of the heating water and the switches of the radiators.
        :param control_horizon: Optimization control horizon (96 periods).
        :param prev_rooms_temp: Measured temperatures at the previous time step.
        :param set_temp: Set temperatures during the control horizon.
        :param forecast_outside_temp: Predictions of outside temperature over the control horizon.
        :param forecast_ground_temp: Predictions of ground temperature over the control horizon.
        :param forecast_irr: Predictions of irradiation over the control horizon.
        :param forecast_occ: Predictions of occupancy over the control horizon.
        :return: Control actions on the heaters switches and heater water temperature
        """

        # t = 0: Precedent time step
        # t = 1: Current time step

        # Convert K to °C
        prev_rooms_temp = {room: np.array(temp)-273.15 for room, temp in prev_rooms_temp.items()}
        set_temp = {room: np.array(temp)-273.15 for room, temp in set_temp.items()}
        forecast_outside_temp = np.array(forecast_outside_temp) - 273.15

        # Set controller parameters
        number_of_rooms = 9
        my_control_horizon = int(min(3, control_horizon)) # control_horizon is the max possible value of the look ahead horizon
        self.control_horizon = my_control_horizon

        # Shift forecast
        outside_temp = np.insert(forecast_outside_temp[:-1], 0, prev_rooms_temp[7]) # shift with previous outside temperature

        a = [] # translate dictionnary to a matrix
        for item in set_temp.values():
            a.append(item)
        set_temp = np.insert(a, 0, 0, axis=1)

        # Optimize
        model = self.make_model(my_control_horizon, prev_rooms_temp, set_temp, outside_temp, forecast_irr, forecast_occ)
        solver = SolverFactory("cbc")
        results = solver.solve(model, tee=False) #

        if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
            self.save_variables(model)

        elif (results.solver.termination_condition == TerminationCondition.infeasible):
            print (">>> INFEASIBLE MODEL dumped to tmp.lp")
            model.write("tmp.lp", io_options={'symbolic_solver_labels': True}) # Export the model

        else:
            # Something else is wrong
            print("Solver Status: ",  results.solver.status)
            print (">>> MODEL dumped to strange.lp")
            model.write("strange.lp", io_options={'symbolic_solver_labels': True}) # Export the model

        # Submit
        switch_heaters = dict()
        boiler_temp = [0.0] * control_horizon
        for r in range(number_of_rooms):
            switch_heaters[r] = [0.0] * control_horizon

        for r in range(7):
            switch_heaters[r][0] = model.Switch[r, 1].value

        
        # Convert °C to K
        boiler_temp[0] = model.Tw[1].value + 273.15

        return switch_heaters, boiler_temp
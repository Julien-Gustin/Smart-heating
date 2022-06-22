from room_temperature_predictor import Cluster, RoomTemperaturePredictor
from room_temperature_predictor import const_dt, const_cp
import numpy as np
from room_init import cluster_1, cluster_2, cluster_3

from pyomo.environ import *
import pyomo as pyo

from utils import room_as_index

M1 = 110

class OptimalHeatingController:

    def __init__(self, alpha=1e-2, beta=1, omega=1):
        self.Tw = 0
        self.Tr_prec = np.ones(7) * 0
        self.prec_Q = np.ones(7) * 0

        self.alpha = alpha
        self.beta = beta
        self.omega = omega
    
    def model_cluster(self, model: ConcreteModel, cluster: Cluster, set_temp, outside_temp, forecast_ground_temp,
                        forecast_irr, forecast_occ):

        for name in cluster.room_names():
            room = cluster._rooms[name]
            model = self.model_room(model, room, set_temp)
            
            Q_int = room._a * forecast_occ
            Q_sol = room._area * forecast_irr * room._gs

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
            model.constraints.add(model.Tr[index, t+1] <= model.Tw[t+1] + M1 * (1 - model.Switch[index, t+1]))
            model.constraints.add(model.Tr[index, t+1] >= model.Tw[t+1] - M1 * (1 - model.Switch[index, t+1]))

            # model.constraints.add(model.Tr[index, t+1] <= model.Tr[index, t] - const_dt * room._k * model.Q_heat[index, t] + M1 * (model.Switch[index, t+1] + model.b1[t+1]))
            model.constraints.add(model.Tr[index, t+1] >= model.Tr[index, t] - const_dt * room._k * model.Q_heat[index, t] - M1 * (model.Switch[index, t+1]))
            model.constraints.add(model.Tr[index, t+1] <= model.Tr[index, t] - const_dt * room._k * model.Q_heat[index, t] + M1 * (model.b1[index, t+1] + model.Switch[index, t+1]))

            model.constraints.add(model.Tr[index, t+1] >= model.Tz[index, t+1])
            model.constraints.add(model.Tr[index, t+1] <= model.Tz[index, t+1] + M1 * (1 - model.b1[index, t+1]) + M1 * model.Switch[index, t+1])
            # Q_heat
            model.constraints.add(model.Q_heat[index, t+1] == const_cp * room._m * (model.Tr[index, t+1] - model.Tz[index, t+1])) 

            # x_diff = |Tz - set_temp|
            model.constraints.add(model.x_diff[index, t+1] >= self.omega * (model.Tz[index, t+1] - set_temp[index][t+1]))
            model.constraints.add(model.x_diff[index, t+1] >= self.beta * (set_temp[index][t+1] - model.Tz[index, t+1]))

        
        return model 

    def init_variables(self, model, prev_rooms_temp):
        model.constraints.add(model.Tw[0] == self.Tw)
        for room in range(7):
            model.constraints.add(model.Tz[room, 0] == prev_rooms_temp[room])
            model.constraints.add(model.Tr[room, 0] == max(self.Tr_prec[room], prev_rooms_temp[room]))
            model.constraints.add(model.Q_heat[room, 0] == self.prec_Q[room])

        return model

    def save_variables(self, model):
        self.Tw = model.Tw[1].value
        for room in range(7):
            self.Tr_prec[room] = model.Tr[room, 1].value
            self.prec_Q[room] = model.Q_heat[room, 1].value

        
    def make_model(self, control_horizon, prev_rooms_temp, set_temp, outside_temp, forecast_ground_temp,
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

        model = self.model_cluster(model, cluster_1, set_temp, outside_temp, forecast_ground_temp, forecast_irr, forecast_occ)
        model = self.model_cluster(model, cluster_2, set_temp, outside_temp, forecast_ground_temp, forecast_irr, forecast_occ)
        model = self.model_cluster(model, cluster_3, set_temp, outside_temp, forecast_ground_temp, forecast_irr, forecast_occ)
        
        model.o = Objective(expr= sum(1e-6 * model.Tw[t] for t in model.control_horizon) + sum(model.x_diff[i, t] + self.alpha * (model.Q_heat[i, t] / 1000) * 0.25 * 0.06 / 0.95 for t in model.control_horizon for i in model.nb_rooms), sense=minimize)
        
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

        number_of_rooms = 7
        my_control_horizon = min(100, control_horizon) # control_horizon is the max possible value of the look ahead horizon
        self.control_horizon = my_control_horizon

        outside_temp = np.insert(forecast_outside_temp[:my_control_horizon-1], 0, prev_rooms_temp[7])

        a = []
        for item in set_temp.values():
            a.append(item)

        set_temp = np.insert(a, 0, 0, axis=1)

        model = self.make_model(my_control_horizon, prev_rooms_temp, set_temp, outside_temp, forecast_ground_temp, forecast_irr, forecast_occ)
       
        solver = SolverFactory("cbc")
        solver.options["threads"] = 4
        results = solver.solve(model, tee=False) # tee=True makes the solver verbose
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

        # Initializing return values  
        switch_heaters = dict()
        temperature_rooms = dict()
        temperature_radiator = dict()
        boiler_temperature = model.Tw[1].value
        heat_radiator = dict()

        for r in range(number_of_rooms):
            switch_heaters[r] = model.Switch[r, 1].value
            temperature_rooms[r] = model.Tz[r, 1].value
            temperature_radiator[r] = model.Tr[r, 1].value
            heat_radiator[r] = model.Q_heat[r, 1].value 

        return temperature_rooms, boiler_temperature, temperature_radiator, heat_radiator, switch_heaters, 10
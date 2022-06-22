import numpy as np

from data import get_data, rooms
from optimize import OptimalHeatingController
from tqdm.notebook import tqdm
def get_data_time_t(t: int, control_horizon:int, data):
    """ could be optimize later on """
    prev_rooms_temp ={room_nb: np.array(data["T_m_" + room])[t-1] for room_nb, room in enumerate(rooms)}
    prev_rooms_temp[7] = data["T_m_ext"][t-1]
    # prev_rooms_temp.append(data["T_m_ext"][t-1])
    set_temp = {room_nb:np.array(data["T_set_" + room][t: t+control_horizon]) for room_nb, room in enumerate(rooms)}
    forecast_outside_temp = np.array(data["T_m_ext"][t: t+control_horizon])
    forecast_ground_temp = np.array(data["T_m_ground_post"][t: t+control_horizon])
    forecast_irr = np.array(data["Irradiation"][t: t+control_horizon])
    forecast_occ = np.array(data["Occupancy"][t: t+control_horizon])
    datetime = np.array(data["Datetime"][t: t+control_horizon])

 
    dates = []
    for date in datetime:
        dates.append(date[:-9])

    return prev_rooms_temp, set_temp, forecast_outside_temp, forecast_ground_temp, forecast_irr, forecast_occ, dates

def append_for_each_room(i:dict, o:dict):
    for r in range(len(rooms)):
        if r not in o.keys():
            o[r] = []
        o[r].append(i[r])
    return o


class Simulation():
    def __init__(self, heating_controller:OptimalHeatingController, control_horizon: int):   
        self.heating_controller = heating_controller
        self.data = get_data()
        self.control_horizon = control_horizon

    def simulate(self, T):
        prev_rooms_temp, Ts, forecast_outside_temp, forecast_ground_temp, forecast_irr, forecast_occ, datetime = get_data_time_t(1, T, self.data)
        Tz = {room_nb:[room_temp] for room_nb, room_temp in prev_rooms_temp.items() if room_nb < len(rooms)}
        set_temp = Ts
        Tw = [0]
        Ta = [prev_rooms_temp[7]] + forecast_outside_temp
        Tr = {r: [0] for r in range(7)}
        Q_heat = {r: [0] for r in range(7)}
        Switch = {r: [0] for r in range(7)}
        sum_obj = 0

        for t in tqdm(range(2, T+1)):
            prev_rooms_temp, boiler_temperature, temperature_radiator, heat_radiator, switch, obj_value = self.heating_controller.compute_actions(self.control_horizon, prev_rooms_temp, set_temp, forecast_outside_temp, forecast_ground_temp, forecast_irr, forecast_occ)

            Tz = append_for_each_room(prev_rooms_temp, Tz)
            Tw.append(boiler_temperature)
            Tr = append_for_each_room(temperature_radiator, Tr)
            Q_heat = append_for_each_room(heat_radiator, Q_heat)
            Switch = append_for_each_room(switch, Switch)

            prev_rooms_temp[7] = forecast_outside_temp[0]
            _, set_temp, forecast_outside_temp, forecast_ground_temp, forecast_irr, forecast_occ, _ = get_data_time_t(t, T, self.data)
            sum_obj += obj_value

        return datetime, Ts, Tz, Tw, Tr, Q_heat, Switch, Ta, sum_obj




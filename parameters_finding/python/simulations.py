from numba import jit
import numpy as np
import sys
import warnings
from multi_zone import model, model_t
warnings.filterwarnings('ignore')
sys.path.insert(0, '../python/')

@jit
def simulate_1(theta, params, prec_temp):
    """Predict temperature for cluster [Diningroom, Kitchen, Bedroom1, Bathroom]

    Args:
        theta: Tunable parameters
        params: Given data

    Returns:
        Predictions of rooms temperatures
    """
    R_Dining, C_Dining, M_Dining, A_Dining, K_Dining = theta[:5]
    R_Kitchen , C_Kitchen , M_Kitchen , A_Kitchen , K_Kitchen  = theta[5:10]
    R_Bedroom_1, C_Bedroom_1, M_Bedroom_1, A_Bedroom_1, K_Bedroom_1 = theta[10:15]
    R_Bathroom, C_Bathroom, M_Bathroom, A_Bathroom, K_Bathroom = theta[15:20]
    R_Dining_kitchen, R_Dining_Bedroom1, R_Kitchen_Bathroom, R_Bedroom1_Bathroom = theta[20:-1]

    df = theta[-1]

    T_set_Dining, T_set_Kitchen, T_set_Bedroom1, T_set_Bathroom = params[:4]
    T_outside, T_water = params[4:]
    
    room_temperature_dining = np.zeros(len(T_outside) + 1)
    room_temperature_kitchen = np.zeros(len(T_outside) + 1)
    room_temperature_bedroom1 = np.zeros(len(T_outside) + 1)
    room_temperature_bathroom = np.zeros(len(T_outside) + 1)

    room_temperature_dining[0] = prec_temp[0] 
    room_temperature_kitchen[0] = prec_temp[1]
    room_temperature_bedroom1[0] = prec_temp[2]
    room_temperature_bathroom[0] = prec_temp[3]

    Tr_prec = np.zeros(4)

    for i in range(len(T_outside)):
        room_temperature_dining[i+1], Tr_prec[0] = model(room_temperature_dining[i], T_set_Dining[i], [room_temperature_kitchen[i], room_temperature_bedroom1[i]], T_water[i], T_outside[i], 
            R_Dining, C_Dining, M_Dining, A_Dining, [R_Dining_kitchen, R_Dining_Bedroom1], K_Dining, Tr_prec[0]) 

        room_temperature_kitchen[i+1], Tr_prec[1] = model(room_temperature_kitchen[i], T_set_Kitchen[i], [room_temperature_dining[i], room_temperature_bathroom[i]], T_water[i], T_outside[i], 
            R_Kitchen , C_Kitchen , M_Kitchen , A_Kitchen, [R_Dining_kitchen, R_Kitchen_Bathroom], K_Kitchen, Tr_prec[1]) 

        room_temperature_bedroom1[i+1], Tr_prec[2] = model(room_temperature_bedroom1[i], T_set_Bedroom1[i], [room_temperature_dining[i], room_temperature_bathroom[i]], T_water[i], T_outside[i], 
            R_Bedroom_1, C_Bedroom_1, M_Bedroom_1, A_Bedroom_1, [R_Dining_Bedroom1, R_Bedroom1_Bathroom], K_Bedroom_1, Tr_prec[2]) 

        room_temperature_bathroom[i+1], Tr_prec[3] = model(room_temperature_bathroom[i], T_set_Bathroom[i], [room_temperature_kitchen[i], room_temperature_bedroom1[i]], T_water[i], T_outside[i], 
            R_Bathroom, C_Bathroom, M_Bathroom, A_Bathroom, [R_Kitchen_Bathroom, R_Bedroom1_Bathroom], K_Bathroom, Tr_prec[3]) 

    obs_temperature_dining = room_temperature_dining + np.random.standard_t(df, *room_temperature_dining.shape)
    obs_temperature_kitchen = room_temperature_kitchen + np.random.standard_t(df, *room_temperature_kitchen.shape)
    obs_temperature_bedroom1 = room_temperature_bedroom1 + np.random.standard_t(df, *room_temperature_bedroom1.shape)
    obs_temperature_bathroom = room_temperature_bathroom + np.random.standard_t(df, *room_temperature_bathroom.shape)

    return [room_temperature_dining[1:], room_temperature_kitchen[1:], room_temperature_bedroom1[1:], room_temperature_bathroom[1:]], [obs_temperature_dining[1:], obs_temperature_kitchen[1:], obs_temperature_bedroom1[1:], obs_temperature_bathroom[1:]]

@jit
def simulate_1_vector(theta, params):
    """Predict temperature for cluster [Livingroom]

    Args:
        theta: Tunable parameters
        params: Given data

    Returns:
        Predictions of rooms temperatures
    """
    R_Dining, C_Dining, M_Dining, A_Dining, K_Dining = theta[:5]
    R_Kitchen , C_Kitchen , M_Kitchen , A_Kitchen , K_Kitchen  = theta[5:10]
    R_Bedroom_1, C_Bedroom_1, M_Bedroom_1, A_Bedroom_1, K_Bedroom_1 = theta[10:15]
    R_Bathroom, C_Bathroom, M_Bathroom, A_Bathroom, K_Bathroom = theta[15:20]
    R_Dining_kitchen, R_Dining_Bedroom1, R_Kitchen_Bathroom, R_Bedroom1_Bathroom = theta[20:-1]

    df = theta[-1]

    T_obs_Dining, T_obs_Kitchen, T_obs_BR1, T_obs_Bathroom = params[:4]
    T_set_Dining, T_set_Kitchen, T_set_Bedroom1, T_set_Bathroom = params[4:8]
    T_outside, T_water = params[8:]
    
    room_temperature_dining = np.zeros(len(T_outside) + 1)
    room_temperature_kitchen = np.zeros(len(T_outside) + 1)
    room_temperature_bedroom1 = np.zeros(len(T_outside) + 1)
    room_temperature_bathroom = np.zeros(len(T_outside) + 1)

    room_temperature_dining[0] = room_temperature_kitchen[0] = room_temperature_bedroom1[0] = room_temperature_bathroom[0] = T_outside[0]
    Tr_prec = np.zeros(4)

    Dining_temperature = model_t(T_obs_Dining, T_set_Dining, [T_obs_Kitchen, T_obs_BR1], T_water, T_outside, R_Dining, C_Dining, M_Dining, A_Dining, [R_Dining_kitchen, R_Dining_Bedroom1], K_Dining)
    Kitchen_temperature = model_t(T_obs_Kitchen, T_set_Kitchen, [T_obs_Bathroom, T_obs_Dining], T_water, T_outside, R_Kitchen, C_Kitchen, M_Kitchen, A_Kitchen, [R_Kitchen_Bathroom, R_Dining_kitchen], K_Kitchen) 
    BR1_temperature = model_t(T_obs_BR1, T_set_Bedroom1, [T_obs_Dining, T_obs_Bathroom], T_water, T_outside, R_Bedroom_1, C_Bedroom_1, M_Bedroom_1, A_Bedroom_1, [R_Dining_Bedroom1, R_Bedroom1_Bathroom], K_Bedroom_1) 
    Bathroom_temperature = model_t(T_obs_Bathroom, T_set_Bathroom, [T_obs_Kitchen, T_obs_BR1], T_water, T_outside, R_Bathroom, C_Bathroom, M_Bathroom, A_Bathroom, [R_Kitchen_Bathroom, R_Bedroom1_Bathroom], K_Bathroom)



    obs_temperature_dining = Dining_temperature + np.random.standard_t(df, *Dining_temperature.shape)
    obs_temperature_kitchen = Kitchen_temperature + np.random.standard_t(df, *Kitchen_temperature.shape)
    obs_temperature_bedroom1 = BR1_temperature + np.random.standard_t(df, *BR1_temperature.shape)
    obs_temperature_bathroom = Bathroom_temperature + np.random.standard_t(df, *Bathroom_temperature.shape)

    return [Dining_temperature, Kitchen_temperature, BR1_temperature, Bathroom_temperature], [obs_temperature_dining, obs_temperature_kitchen, obs_temperature_bedroom1, obs_temperature_bathroom]


@jit
def simulate_2(theta, params, temp_prec):
    """Predict temperature for cluster [Livingroom]
    Args:
        theta: Tunable parameters
        params: Given data
    Returns:
        Predictions of rooms temperatures
    """
    R_living, C_living, M_living, A_living, K_living = theta[:-1]
    T_set, T_outside, T_water = params
    

    df = theta[-1]

    room_temperature = np.zeros(len(T_outside) + 1)
    room_temperature[0] = temp_prec
    Tr_prec = 0

    for i in range(len(T_set)):
        room_temperature[i+1], Tr_prec = model(room_temperature[i], T_set[i], [], T_water[i], T_outside[i], R_living, C_living, M_living, A_living, [], K_living, Tr_prec) 

    obs_temperature = room_temperature + np.random.standard_t(df, *room_temperature.shape) 

    return [room_temperature[1:]], [obs_temperature[1:]]

@jit
def simulate_2_vector(theta, params):
    """Predict temperature for cluster [Livingroom]

    Args:
        theta: Tunable parameters
        params: Given data

    Returns:
        Predictions of rooms temperatures
    """
    R_living, C_living, M_living, A_living, K_living = theta[:-1]
    T_obs_living, T_set, T_outside, T_water = params

    df = theta[-1]

    room_temperature = model_t(T_obs_living, T_set, [], T_water, T_outside, R_living, C_living, M_living, A_living, [], K_living) 

    obs_temperature = room_temperature + np.random.standard_t(df, *room_temperature.shape) 

    return [room_temperature], [obs_temperature]



@jit
def simulate_3(theta, params, temp_prec):
    """Predict temperature for cluster [Bedroom2, Bedroom3]

    Args:
        theta: Tunable parameters
        params: Given data

    Returns:
        Predictions of rooms temperatures
    """
    R_Bedroom_2, C_Bedroom_2, M_Bedroom_2, A_Bedroom_2, K_Bedroom_2 = theta[0:5]
    R_Bedroom_3, C_Bedroom_3, M_Bedroom_3, A_Bedroom_3, K_Bedroom_3 = theta[5:10]
    R_Bedroom2_Bedroom3 = theta[10:-1][0]

    df = theta[-1]

    T_set_Bedroom2, T_set_Bedroom3 = params[:2]
    T_outside, T_water = params[2:]
    
    room_temperature_bedroom2 = np.zeros(len(T_outside) + 1)
    room_temperature_bedroom3 = np.zeros(len(T_outside) + 1)

    room_temperature_bedroom2[0] = temp_prec[0]
    room_temperature_bedroom3[0] = temp_prec[1]
    Tr_prec = np.zeros(2)

    for i in range(len(T_outside)):
        room_temperature_bedroom2[i+1], Tr_prec[0] = model(room_temperature_bedroom2[i], T_set_Bedroom2[i], [room_temperature_bedroom3[i]], T_water[i], T_outside[i], 
            R_Bedroom_2, C_Bedroom_2, M_Bedroom_2, A_Bedroom_2, [R_Bedroom2_Bedroom3], K_Bedroom_2, Tr_prec[0]) 

        room_temperature_bedroom3[i+1], Tr_prec[1] = model(room_temperature_bedroom3[i], T_set_Bedroom3[i], [room_temperature_bedroom2[i]], T_water[i], T_outside[i], 
            R_Bedroom_3, C_Bedroom_3, M_Bedroom_3, A_Bedroom_3, [R_Bedroom2_Bedroom3], K_Bedroom_3, Tr_prec[1]) 

    obs_temperature_bedroom2 = room_temperature_bedroom2 + np.random.standard_t(df, *room_temperature_bedroom2.shape)
    obs_temperature_bedroom3 = room_temperature_bedroom3 + np.random.standard_t(df, *room_temperature_bedroom3.shape)

    return [room_temperature_bedroom2[1:], room_temperature_bedroom3[1:]], [obs_temperature_bedroom2[1:], obs_temperature_bedroom3[1:]]


@jit
def simulate_3_vector(theta, params):
    """Predict temperature for cluster [Livingroom]

    Args:
        theta: Tunable parameters
        params: Given data

    Returns:
        Predictions of rooms temperatures
    """
    R_Bedroom_2, C_Bedroom_2, M_Bedroom_2, A_Bedroom_2, K_Bedroom_2 = theta[0:5]
    R_Bedroom_3, C_Bedroom_3, M_Bedroom_3, A_Bedroom_3, K_Bedroom_3 = theta[5:10]
    R_Bedroom2_Bedroom3 = theta[10:-1][0]

    T_obs_BR2, T_obs_BR3, T_set_Bedroom2, T_set_Bedroom3 = params[:4]
    T_outside, T_water = params[4:]

    df = theta[-1]

    BR2_temperature = model_t(T_obs_BR2, T_set_Bedroom2, [T_obs_BR3], T_water, T_outside, R_Bedroom_2, C_Bedroom_2, M_Bedroom_2, A_Bedroom_2, [R_Bedroom2_Bedroom3], K_Bedroom_2) 
    BR3_temperature = model_t(T_obs_BR3, T_set_Bedroom3, [T_obs_BR2], T_water, T_outside, R_Bedroom_3, C_Bedroom_3, M_Bedroom_3, A_Bedroom_3, [R_Bedroom2_Bedroom3], K_Bedroom_3) 

    obs_temperature_bedroom2 = BR2_temperature + np.random.standard_t(df, *BR2_temperature.shape)
    obs_temperature_bedroom3 = BR3_temperature + np.random.standard_t(df, *BR3_temperature.shape)

    return [BR2_temperature, BR3_temperature], [obs_temperature_bedroom2, obs_temperature_bedroom3]


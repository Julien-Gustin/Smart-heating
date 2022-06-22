import numpy as np
from graph import Graph
from scipy.optimize import least_squares, minimize
from numba import jit
import functools
import operator

DT = 900
CP = 4810

# initial_values = [
#     [
#         8e-1,  # R0
#         5e-1,  # R1
#         3,  # R4
#         4e-1,  # R3
#         148500,  # C0
#         244800,  # C1
#         33900,  # C4
#         591300,  # C3
#         5/4810,  # M0
#         8/4810,  # M1
#         0.6/4810,  # M4
#         9/4810,  # M3
#         15,  # A0
#         15,  # A1
#         15,  # A4
#         15,  # A3
#         0.9,  # k0
#         0.9,  # k1
#         0.9,  # k4
#         0.9,  # k3
#         0, # tolerance1
#         0, # tolerance2
#         0, # tolerance3
#         0, # tolerance4
#         0.5, #gs1
#         0.5, #gs2
#         0.5, #gs3
#         0.5, #gs4
#         5e-2,
#         30,
#         5e-1,
#         5e-1,
#     ],
#     [30, 8700, 8e-02/4810, 15, 0.9, 0, 0.5],  # [R2, C2, M2, A2, k2, tolerance2, gs2]
#     [
#         7e-1,  # R5
#         1,  # R6
#         350400,  # C5
#         416400,  # C6
#         2/4810,  # M5
#         2/4810,  # M6
#         15,  # A5
#         15,  # A6
#         0.9,  # k5
#         0.9,  # k6
#         0, # tolerance5
#         0, # tolerance6
#         0.5, # gs5
#         0.5, # gs6
#         2e-1,
#     ],
# ]
initial_values = [
    [
        8e-2,  # R0
        5e-2,  # R1
        3e-1,  # R4
        4e-2,  # R3
        148500,  # C0
        244800,  # C1
        33900,  # C4
        191300,  # C3
        5/4810,  # M0
        8/4810,  # M1
        0.6/4810,  # M4
        9/4810,  # M3
        15,  # A0
        15,  # A1
        15,  # A4
        15,  # A3
        0.9,  # k0
        0.9,  # k1
        0.9,  # k4
        0.9,  # k3
        0, # tolerance1
        0, # tolerance2
        0, # tolerance3
        0, # tolerance4
        0.5, #gs1
        0.5, #gs2
        0.5, #gs3
        0.5, #gs4
        5e-2,
        30,
        5e-1,
        5e-1,
    ],
    [3, 17000, 8e-02/4810, 15, 0.9, 0, 0.5],  # [R2, C2, M2, A2, k2, tolerance2, gs2]
    [
        7e-3,  # R5
        1e-2,  # R6
        350400,  # C5
        416400,  # C6
        2/4810,  # M5
        2/4810,  # M6
        15,  # A5
        15,  # A6
        0.9,  # k5
        0.9,  # k6
        0, # tolerance5
        0, # tolerance6
        0.5, # gs5
        0.5, # gs6
        2e-1,
    ],
]

def model_t(
    Tza: float,
    Ts: float,
    Tzb: np.array,
    Tw: float,
    Ta: float,
    R: float,
    C: float,
    m: float,
    a: float,
    Rab: np.array,
    k,
    switch,
    gs,
    area, 
    irradiance,
    occupancy,
) -> float:
    """A model to predict the temperature at time t+1 work given array for each t or for a single t
    Args:
        Tza (float/np.array): Room temperature at t
        Ts (float/np.array): Temperature set at time t
        Tzb (np.array): Neighboor room temperature at t
        Tw (float/np.array): Water temperature at t
        Ta (float/np.array): Outdoor temperature at t
        R (float): Thermal resistance
        C (float): Thermal capacitor
        m (float): Mass flow rate
        a (float): offset correction
        Rab (np.array): Array of thermal resistance that link zone a and b
    Returns:
        float: predicted temperature
    """

    radiator_on = switch
    pred_t = np.zeros(len(Tza))
    adjacent_dependence = np.zeros(len(Tza))
    Q_heat = 0
    Q_sol = 0
    Tr = 0
    
    for tzb, rab in zip(Tzb, Rab):
        adjacent_dependence += (Tza - tzb) / (rab * C)

    adjacent_dependence

    pred_t_without_q = (
        (Ta - Tza) / (R * C) - adjacent_dependence
    )  # for optimization purpose

    Q_sol = area * irradiance * gs

    Q_int = a * occupancy

    for t in range(len(Tza)):
        if radiator_on[t] and Tza[t] <= Ts[t]:
            Tr = Tw[t]

        else:
            Tr = Tr - DT * k * Q_heat

        if Tr <= Tza[t]:
            Tr = Tza[t]

        Q_heat = CP * m * (Tr - Tza[t])
        # Q_heat = max(0, Q_heat)

        pred_t[t] = pred_t_without_q[t] + (Q_heat + Q_sol[t] + Q_int[t]) / C

    pred_t = (pred_t * DT) + Tza

    return pred_t

@jit
def model(
    Tza: float,
    Ts: float,
    Tzb: np.array,
    Tw: float,
    Ta: float,
    R: float,
    C: float,
    m: float,
    a: float,
    Rab: np.array,
    k,
    switch,
    gs,
    area, 
    irradiance,
    occupancy,
    Tr_prec,
    Q_prec
) -> float:
    """A model to predict the temperature at time t+1 work given array for each t or for a single t
    Args:
        Tza (float/np.array): Room temperature at t
        Ts (float/np.array): Temperature set at time t
        Tzb (np.array): Neighboor room temperature at t
        Tw (float/np.array): Water temperature at t
        Ta (float/np.array): Outdoor temperature at t
        R (float): Thermal resistance
        C (float): Thermal capacitor
        m (float): Mass flow rate
        a (float): offset correction
        Rab (np.array): Array of thermal resistance that link zone a and b
    Returns:
        float: predicted temperature
    """

    radiator_on = switch

    if radiator_on and Tza <= Ts:
        Tr = Tw

    else:
        Tr = Tr_prec - DT * k * Q_prec  # 0 <= k <= 1
        # Tr = max(Tr, Tza)

    if Tr <= Tza:
        Tr = Tza

    adjacent_dependence = 0

    Q_sol = area * irradiance * gs
    Q_heat = CP * m * (Tr - Tza)
    Q_int = a * occupancy

    for tzb, rab in zip(Tzb, Rab):
        adjacent_dependence += (Tza - tzb) / (rab * C)

    pred_t = ((Ta - Tza) / (R * C) + (Q_heat + Q_sol + Q_int) / C - adjacent_dependence) * DT + Tza

    return pred_t, Tr, Q_heat

@jit
def model_train(X: np.array, t: list, Y: list) -> np.array:
    """Find the best parameters in X given data in t according to output Y
    Args:
        X (np.array): Parameters to be estimated [RR..RCC..CCMM..MMAA..AARAB..RAB] each parameters is repeted `number_of_room` time unless for last one 
        t (dict): Given data
        Y (list): Output]
    Returns:
        np.array: Array of residuals
    """
    number_of_room = len(Y)
    R = X[:number_of_room]
    C = X[number_of_room : 2 * number_of_room]
    M = X[2 * number_of_room : 3 * number_of_room]
    A = X[3 * number_of_room : 4 * number_of_room]
    K = X[4 * number_of_room : 5 * number_of_room]
    GS = X[5 * number_of_room : 6 * number_of_room]
    RAB = X[6 * number_of_room :]

    DICO, Ta, Tw, irradiation, occupancy = t

    Tza = DICO["Tza"]
    Ts = DICO["Ts"]
    Tzb = DICO["Tzb"]
    Rab = DICO["Rab"]
    AREA = DICO["Area"]
    SWITCH = DICO["Switch"]

    residuals = []
    for r, c, m, a, tza, ts, tzb, rab, k, gs, area, y, switch in zip(R, C, M, A, Tza, Ts, Tzb, Rab, K, GS, AREA, Y, SWITCH):
        Rab_in = np.ones(len(rab))
        for i, rb in enumerate(rab):
            Rab_in[i] = RAB[rb]

        pred_t = model_t(tza, ts, tzb, Tw, Ta, r, c, m, a, Rab_in, k, switch, gs, area, irradiation, occupancy)
        residual = pred_t - y
        residuals.append(residual)

    residuals = functools.reduce(operator.iconcat, residuals, [])
    return np.array(residuals)

def set_bounds(lower_bounds, upper_bounds, size):
    for i in range(size):
        # gs
        upper_bounds[4 * size + i] = 1
        upper_bounds[5 * size + i] = 0.9

    return (lower_bounds, upper_bounds)


class Room:
    def __init__(self, number) -> None:
        self.number = number
        self.R = 0.0
        self.C = 0.0
        self.m = 0.0
        self.a = 0.0
        self.k = 0.0
        self.gs = 0.0
        self.area = 0.0
        self.Rab = np.zeros(0)
        self.neighbours = []


class MultiZoneRegression:
    def __init__(self, neighbour_matrix: np.array, area, initial_values):
        if not np.allclose(neighbour_matrix, neighbour_matrix.T):
            raise (Exception("Matrix should be diagonal"))

        self.initial_values = initial_values

        graph = Graph(neighbour_matrix.shape[0])
        graph.matrixToGraph(neighbour_matrix)

        self.cluster = graph.connectedComponents()
        self.rooms = [Room(i) for i in range(len(neighbour_matrix))]

        for i, row in enumerate(neighbour_matrix):  # connect neighbour together
            neighbours = []
            for j, elem in enumerate(row):
                if elem:
                    neighbours.append(self.rooms[j])

            self.rooms[i].neighbours = neighbours
            self.rooms[i].area = area[i]

    def _create_dico(self, cluster, data):
        input_dico = {  # dico that will be given to model_train via least_square
            "Tza": [],
            "Ts": [],
            "Tzb": [],
            "Rab": {},
            "Area": [],
            "Switch": []
        }

        Rab = {}
        rab_count = 0
        rab = []
        for room_nb in cluster:
            input_dico["Tza"].append(
                data[room_nb]
            )  # append the temperature of the room
            input_dico["Ts"].append(
                data[room_nb + len(self.rooms)]
            )  # append the setpoint of the room
            input_dico["Area"].append(
                self.rooms[room_nb].area
            )
            input_dico["Switch"].append(
                data[room_nb + 2 * len(self.rooms)]
            )
            room_rab = []
            tzb = []
            for neighbour in self.rooms[
                room_nb
            ].neighbours:  # store where the corresponding Rab is located in {Rab}
                if (room_nb, neighbour.number) not in Rab.keys() and (
                    neighbour.number,
                    room_nb,
                ) not in Rab.keys():
                    Rab[(room_nb, neighbour.number)] = rab_count
                    Rab[(neighbour.number, room_nb)] = rab_count

                    rab_count += 1

                room_rab.append(
                    Rab[(room_nb, neighbour.number)]
                )  # save to which Rab it corresponds
                tzb.append(data[neighbour.number])  # store id of neighbour

            input_dico["Tzb"].append(tzb)

            rab.append(room_rab)
        input_dico["Rab"] = rab
        return input_dico, Rab

    def fit(self, X: np.array, y: np.array, area, fit=True):
        """Fit the model using data and expected outcome
        Args:
            X (np.array): Input data
            y (np.array): Expected outcome
        """
        for cluster, x0 in zip(
            self.cluster, self.initial_values
        ):  # compute best parameters for each cluster

            # Create dico that will be given to least square
            input_dico, Rab = self._create_dico(cluster, X)

            # Init parameters
            x0 = np.array(x0)
            # Given data
            input_ls = [input_dico, X[-4], X[-3], X[-2], X[-1]]  # {dico}, Ta, Tw, irradiation, occupancy

            # Set upper and lower bounds
            lower_bounds = [0] * len(x0)
            upper_bounds = [1e8] * len(x0)
            bounds = set_bounds(lower_bounds, upper_bounds, len(cluster))

            # Learning parameters
            if fit:
                ls = least_squares(
                    model_train,
                    x0,
                    args=(input_ls, y[cluster]),
                    # loss="cauchy",
                    bounds=bounds,
                )

                X_opt = ls.x
                # print(cluster)
                # print(ls)
            else:
                X_opt = x0


            # Set learned parameters
            R = X_opt[: len(cluster)]
            C = X_opt[len(cluster) : 2 * len(cluster)]
            M = X_opt[2 * len(cluster) : 3 * len(cluster)]
            A = X_opt[3 * len(cluster) : 4 * len(cluster)]
            K = X_opt[4 * len(cluster) : 5 * len(cluster)]
            GS = X_opt[5 * len(cluster) : 6 * len(cluster)]
            RAB = X_opt[6 * len(cluster) :]

            for i, room_nb in enumerate(cluster):
                self.rooms[room_nb].R = R[i]
                self.rooms[room_nb].C = C[i]
                self.rooms[room_nb].m = M[i]
                self.rooms[room_nb].a = A[i]
                self.rooms[room_nb].k = K[i]
                self.rooms[room_nb].gs = GS[i]

                for neighbour in self.rooms[room_nb].neighbours:
                    self.rooms[room_nb].Rab = np.append(
                        self.rooms[room_nb].Rab, RAB[Rab[room_nb, neighbour.number]]
                    )

    def predict_all(self, X: np.array, Z_prec) -> np.array:
        """ Predict the outcome given `X`
        Args:
            X (np.array): Input data
        Returns:
            np.array: predicted value
        """
        # Z_prec = np.ones(len(self.rooms)) * 20
        y_pred = np.ones((len(self.rooms), len(X)))
        Tr_prec = np.zeros(len(self.rooms))
        Q_prec = np.zeros(len(self.rooms))


        for cluster in self.cluster:
            for i, xi in enumerate(X):
                for room_nb in cluster:
                    neighbour_temp_prec = []
                    for neighbour in self.rooms[room_nb].neighbours:
                        neighbour_temp_prec.append(Z_prec[neighbour.number])

                    neighbour_temp_prec = np.array(neighbour_temp_prec)

                    y_pred[room_nb][i], Tr_prec[room_nb], Q_prec[room_nb] = model(
                        Z_prec[room_nb],
                        xi[room_nb],
                        neighbour_temp_prec,
                        xi[-3],
                        xi[-4],
                        self.rooms[room_nb].R,
                        self.rooms[room_nb].C,
                        self.rooms[room_nb].m,
                        self.rooms[room_nb].a,
                        self.rooms[room_nb].Rab,
                        self.rooms[room_nb].k,
                        xi[room_nb + 2 * len(self.rooms)],
                        self.rooms[room_nb].gs,
                        self.rooms[room_nb].area,
                        xi[-2],
                        xi[-1],
                        Tr_prec[room_nb],
                        Q_prec[room_nb]
                    )

                for room_nb in cluster:
                    Z_prec[room_nb] = y_pred[room_nb][i]

        return y_pred
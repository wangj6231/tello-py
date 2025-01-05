import numpy as np


def set_enemies():

    ENEMY=[
        [0.8750, 0.0500, 0.0300],
        [0.4250, 0.1000, 0.0450],
        [0.3000, 0.2000, 0.0500],
        [0.4350, 0.2750, 0.0500],
        [0.9250, 0.3000, 0.0500],
        [0.6000, 0.5000, 0.0550],
        [0.6750, 0.6500, 0.0250],
        [0.8500, 0.6500, 0.0250],
        [0.7500, 0.8000, 0.0625],
        [0.5500, 0.8250, 0.0500]
        ]
    
    return ENEMY

def get_mobile_demands():
    # positions = [
    #         [0.71, 0.22, 1],
    #         [0.23, 0.55, 1],
    #         [0.38, 0.48, 0.5],
    #         [0.5, 0.71, 1]]
    positions = np.random.rand(6, 3)
    # positions = positions.tolist()
    print(positions)
    return positions
    
    
    
def get_config():
    config = {
            'N': 20,
            'N2': 50,
            'K': 20,
            'M': 0.05,
            'service_rate':0.1,
            'step_way': 0.02,
            'observe_r': 0.2,
            'epo': 20 * 20
     }
    return config

def norm(x):
    res = np.sqrt(np.square(x[0]) + np.square(x[1]))
    return res

def calculate_distance(point1, point2):
    return np.sqrt(np.square(point1[0] - point2[0]) + np.square(point1[1] - point2[1]))

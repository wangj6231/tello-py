import numpy as np

class stereoCamera(object):
    def __init__(self):
        # 左
        self.cam_matrix_left = np.array([   [830.5873,   -3.0662,  658.1007],
                                            [       0,  830.8116,  482.9859],
                                            [       0,         0,         1]
                                        ])
        # 右
        self.cam_matrix_right = np.array([  [830.4255,   -3.5852,  636.8418],
                                            [       0,  830.7571,  476.0664],
                                            [       0,         0,         1]
                                        ])

        # [k1, k2, p1, p2, k3]
        self.distortion_l = np.array([[-0.0806, 0.3806, -0.0033, 0.0005148, -0.5229]])
        self.distortion_r = np.array([[-0.0485, 0.2200, -0.002,  0.0017,    -0.2876]])

        # 矩陣
        self.R = np.array([ [      1,  0.0017, -0.0093],
                            [-0.0018,  1.0000, -0.0019],
                            [ 0.0093,  0.0019,  1.0000]   
                            ])

        self.T = np.array([[-119.9578], [0.1121], [-0.2134]])

        self.focal_length = 859.367 

        self.baseline = 119.9578 

        



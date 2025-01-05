# 這個檔案可以用來定義一些輔助函數或類別，以支援路線規劃和避障功能的實作。

import numpy as np

def calculate_route(start, end):
    # 計算從起點到終點的路線
    route = []  # 儲存路線的列表
    # 在這裡加入路線計算的邏輯
    return route

def obstacle_avoidance(frame):
    # 使用影像處理技術進行避障
    obstacles = []  # 儲存障礙物的列表
    # 在這裡加入避障檢測的邏輯
    return obstacles

def get_drone_position():
    # 獲取無人機當前的位置
    position = (0, 0)  # 假設的座標
    # 在這裡加入獲取位置的邏輯
    return position

def norm(x):
    res = np.sqrt(np.square(x[0]) + np.square(x[1]))
    return res

def calculate_distance(point1, point2):
    return np.sqrt(np.square(point1[0] - point2[0]) + np.square(point1[1] - point2[1]))
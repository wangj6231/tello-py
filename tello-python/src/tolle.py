import cv2
import mediapipe as mp
from djitellopy import Tello
import pywifi 
from pywifi import const
import comtypes
import threading
import time
import os
import numpy as np
import heapq
from utils import calculate_distance, obstacle_avoidance, get_drone_position
import drone_camera_distance 
import stereoconfig

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils
tello = Tello()
tello.connect()
print(f"Battery: {tello.get_battery()}%")

drone_in_air = False
shutdown_event = threading.Event()

class PriorityQueue:
    def __init__(self):
        self.elements = []

    def empty(self):
        return len(self.elements) == 0

    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        return heapq.heappop(self.elements)[1]

class SquareGrid:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.walls = []

    def in_bounds(self, id):
        (x, y) = id
        return 0 <= x < self.width and 0 <= y < self.height

    def neighbors(self, id):
        (x, y) = id
        results = [(x+1, y), (x+1, y-1), (x, y-1), (x-1, y-1), (x-1, y), (x-1, y+1), (x, y+1), (x+1, y+1)]
        if (x + y) % 2 == 0:
            results.reverse()
        return results

class GridWithWeights(SquareGrid):
    def __init__(self, width, height):
        super(GridWithWeights, self).__init__(width, height)
        self.weights = {}

    def cost(self, service_list, M, N, r, now_node, E, K, to_node):
        demand = demand_matrix(to_node, service_list, M, N, r)
        A = danger_matrix(now_node, N, E, K, to_node) - demand
        self.weights[to_node] = A
        return A

def connect_to_open_wifi(ssid): 

    wifi = pywifi.PyWiFi() 

    iface = wifi.interfaces()[0] 

    iface.disconnect() 

    time.sleep(2) 

    profile = pywifi.Profile() 

    profile.ssid = ssid   

    profile.auth = const.AUTH_ALG_OPEN 

    profile.akm.append(const.AKM_TYPE_NONE) 

    profile.cipher = const.CIPHER_TYPE_NONE 

    iface.remove_all_network_profiles() 

    temp_profile = iface.add_network_profile(profile) 

    iface.connect(temp_profile) 

    time.sleep(10) 

 

    if iface.status() == const.IFACE_CONNECTED: 

        print(f"已連接 {ssid}") 

        return True 

    else: 

        print(f"連接到 {ssid} 失敗") 

        return False 

def demand_matrix(s, service_list, M, N, r):
    pos = np.array(s) / N
    demand = 0
    for server in service_list:
        server_pos = np.array([server[0], server[1]])
        server_demand = server[2]
        if server_demand <= 0:
            demand = 0
        elif calculate_distance(pos, server_pos) < r:
            demand += server_demand * M
    return demand

def danger_matrix(s, N, E, K, to_node):
    x, y = to_node
    N2 = E.shape[0]
    step_dis = 1 / N2
    s1 = np.array(s) / N
    s2 = np.array([(x) / N, (y) / N])
    delta = 0
    dis = calculate_distance(s1, s2)
    weight_A = dis
    while dis > 0.000001:
        s1, move = next_step(s1, s2, step_dis)
        dis -= move
        delta += move * E[int(np.ceil(s1[0] * N2)), int(np.ceil(s1[1] * N2))]
    weight_A += K * delta
    return weight_A

def next_step(s1, s2, step_dis):
    move = calculate_distance(s1, s2)
    if move <= step_dis:
        s3 = s2
    else:
        s3 = s1 + (s2 - s1) * step_dis / move
        move = step_dis
    return s3, move

def heuristic(a, b):
    (x1, y1) = a
    (x2, y2) = b
    return abs(x1 - x2) + abs(y1 - y2)

def a_star_search(graph, start, goal, E, N, K, M, service_list, r):
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0
    while not frontier.empty():
        current = frontier.get()
        if current == goal:
            break
        if (current >= (N, N)).any():
            print('out of indexs!')
            break
        for next in graph.neighbors(current):
            if next[0] < 0 or next[1] < 0 or next[0] > 19 or next[1] > 19:
                continue
            new_cost = cost_so_far[current] + graph.cost(service_list, M, N, r, current, E, K, next)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + heuristic(next, goal)/N
                frontier.put(next, priority)
                came_from[next] = current
    return came_from, cost_so_far

def reconstruct_path(came_from, start, goal):
    current = goal
    path = [current]
    while current != start:
        current = came_from[current]
        path.append(current)
    path.append(start)
    path.reverse()
    return path

def planning(pos, tar, E, N, K, M, service_list, r):
    start, end = tuple(pos), tuple(tar)
    graph = GridWithWeights(N, N)
    came_from, _ = a_star_search(graph, start, end, E, N, K, M, service_list, r)
    path = reconstruct_path(came_from, start, end)
    return path

def move_drone_along_path(drone, path):
    for step in path:
        print(f"Moving to {step}")
        # 假設每一步都是一個單位距離
        drone.move_forward(20)  # 根據實際情況調整
        time.sleep(1)

def process_video_stream(drone):
    cap = cv2.VideoCapture(0)
    while not shutdown_event.is_set():
        ret, frame = cap.read()
        if not ret:
            continue
        # 用於障礙物檢測的工藝框架
        obstacles = obstacle_avoidance(frame)
        if obstacles:
            print("Obstacles detected:", obstacles)
            # 根據障礙物調整路徑的邏輯
            pos = get_drone_position()
            tar = (5, 5)  # 目標位置可以根據需要調整
            E = np.zeros((20, 20))
            N = 20
            K = 1.0
            M = 1.0
            service_list = []
            r = 1.0
            path = planning(pos, tar, E, N, K, M, service_list, r)
            print("Adjusted Path:", path)
            move_drone_along_path(drone, path)
        cv2.imshow('Drone Camera', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def main():
    ssid = "TELLO-634F52"
    if connect_to_open_wifi(ssid):
        time.sleep(5)
        global drone
        drone = Tello()
        drone.connect()
        # 啟動無人機鏡頭
        drone.streamon()
        threading.Thread(target=process_video_stream, args=(drone,)).start()
        pos = (0, 0)
        tar = (5, 5)
        E = np.zeros((20, 20))
        N = 20
        K = 1.0
        M = 1.0
        service_list = []
        r = 1.0
        path = planning(pos, tar, E, N, K, M, service_list, r)
        print("Planned Path:", path)
        move_drone_along_path(drone, path)
        
        # 使用 drone_camera_distance 模組測量距離
        frame_read = drone.get_frame_read()
        while True:
            frame = frame_read.frame
            distance = drone_camera_distance.measure_distance(frame)
            if distance:
                print(f"Distance to obstacle: {distance:.2f} cm")
            cv2.imshow("Tello Camera", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        drone.streamoff()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
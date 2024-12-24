import cv2
import numpy as np
import torch
from djitellopy import Tello

# 載入 YOLOv5 模型
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

def measure_distance(frame):
    # 使用 YOLOv5 模型檢測障礙物
    results = model(frame)
    labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    
    # 假設第一個檢測到的物體是障礙物
    if len(labels) > 0:
        x1, y1, x2, y2, conf = cord[0]
        width = x2 - x1
        distance = 1000 / width  # 假設障礙物的實際寬度已知，這裡用1000作為比例因子
        return distance
    return None

def main():
    tello = Tello()
    tello.connect()
    print(f"Battery: {tello.get_battery()}%")
    tello.streamon()
    frame_read = tello.get_frame_read()

    while True:
        frame = frame_read.frame
        distance = measure_distance(frame)
        if distance:
            print(f"Distance to obstacle: {distance:.2f} cm")

        cv2.imshow("Tello Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    tello.streamoff()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
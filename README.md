# 無人機控制專案

這個專案旨在使用 Python 控制無人機，並實現路線規劃和鏡頭避障功能。專案使用 OpenCV 和 MediaPipe 進行影像處理，並透過 Tello API 控制無人機的操作。

## 專案結構

```
tello-python
├── src
│   ├── dianyuntu_yolo.py  # 圖像處理和立體匹配相關函數
│   ├── stereoconfig.py    # 立體相機配置
│   ├── tolle.py           # 主要程式碼，控制無人機的起飛、降落和其他操作
│   ├── utils
│   └── yolov5-fpn.yaml # 模型訓練
│       └── __init__.py    # 輔助函數或類別，用於路線規劃和避障功能
├── requirements.txt       # 專案所需的 Python 套件及其版本
└── README.md              # 專案說明和使用指南
```

## 安裝指南

1. 確保已安裝 Python 3.12。
2. 下載或克隆此專案。
3. 在專案根目錄下，使用以下命令安裝所需的套件：

   ```sh
   pip install -r requirements.txt
   ```

## 使用指南

1. 連接無人機並確保其電源開啟。
2. 執行 `src/tolle.py` 以啟動無人機控制程式：

   ```sh
   python src/tolle.py
   ```

3. 使用程式中的指令控制無人機的起飛、降落及其他操作。

## 功能

- **控制無人機的起飛和降落**：使用 Tello API 控制無人機的基本操作。
- **路線規劃功能**：使用 A* 演算法進行路線規劃，確保無人機能夠避開障礙物到達目標位置。
- **鏡頭避障功能**：使用 OpenCV 和 MediaPipe 進行影像處理，檢測並避開障礙物。
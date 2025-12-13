# IGVC-Object-Detection

This repository contains scripts for training and evaluating a custom YOLOv11 model on our own dataset. The model is used for the IGVC 2025 self-driving car model.

## 📂 Requirements

- `Python 3.10`
- `ultralytics`  
- `opencv-python`  
- `PyYAML`
- `uv`  

Install dependencies with:
```bash
pip install -r requirements.txt
```
### 📂 Dataset

You can download the dataset from the following link:

👉 [**Download Dataset (Google Drive)**](https://drive.google.com/drive/folders/18tpDPkw0tC9mf_HfYlu_ZCCebQhCNo6M?usp=drive_link)

## After downloading, unzip using the following command:
``` unzip Dataset.zip```

## The dataset folder structure should look as follows:
```bash
Dataset/
yolo
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
data.yaml
```
### 📂 Pre-trained Model Download

You can download the trained model from the following link:

👉 [**Download Models (Google Drive)**](https://drive.google.com/drive/folders/1nU-36Mtn90B-VykBUzdMBfgZOreif3Is?usp=drive_link)

### 📂 Training (If needed)

Use train.py to train your model:
```bash
python train.py \
  --data Dataset/data.yaml \
  --epochs 300 \
  --imgsz 640 \
  --batch 4 \
  --project results/object_detection \
  --name results
```
### 📂 Evaluation
Use evaluate.py to evaluate the trained model:

## Evaluate with Single Image
```bash
python evaluate.py --weights Pretrained_weights/weights/best.pt --image path/to/image.jpg
```

## Evaluate with Full Test Dataset
```bash
python evaluate.py \
  --weights Pretrained_weights/weights/best.pt \
  --dataset Dataset/data.yaml
```
## Real-time Object Detection Evaluation

```bash
python evaluate.py --weights Pretrained_weights/weights/best.pt --camera

```
## Generate an ONNX model to evaluate on edge devices

## Export (For Dynamic settings, please look into the comments on convert_to_onnx.py)

```bash
python convert_to_onnx.py --weights Pretrained_weights/weights/best.pt --imgsz 640 --half
```

## Evaluate ONNX Model

```bash
python evaluate_onnx_model.py --data Dataset/data.yaml -weights Pretrained_weights/weights/best.onnx
```

### 📊 Evaluation Results

## Evaluation Results on Test Dataset

The trained YOLO model was evaluated directly on the test dataset.  
Below are the summarized detection metrics:

| Model      | mAP@0.5 | mAP@0.5:0.95 | Precision (μ) | Recall (μ) | Average inference time (ms/im) | FPS
|-----------|:--------:|:------------:|:-------------:|:----------:|:----------:|:----------:|
| YOLOv11x  | **0.887** | **0.692**   | **0.861**     | **0.842**  |**22.58** | **44.28** |
| YOLOv11s  | **0.572** | **0.293**   | **0.756**     | **0.469**  | **10.09** | **99.10** |
| YOLOv5s   | **0.517** | **0.244**   | **0.600**     | **0.579**  |**9.21** | **108.62** |
| YOLOv5x   | **0.630** | **0.313**   | **0.682**     | **0.563**  |**50.82** | **19.79** |
---

## Per-Class Performance (Yolov11x)

| Class | Images | Instances | Precision (P) | Recall (R) | mAP@0.5 | mAP@0.5:0.95 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| **stop-sign** | 49 | 49 | 0.978 | 0.888 | 0.977 | 0.706 |
| **stop-sign-vandalized** | 16 | 16 | 0.863 | 0.786 | 0.799 | 0.633 |
| **traffic barrel** | 101 | 198 | 0.802 | 0.859 | 0.886 | 0.689 |
| **traffic cone** | 86 | 86 | 0.989 | 1.000 | 0.995 | 0.982 |
| **car-tire** | 103 | 154 | 0.976 | 0.777 | 0.922 | 0.647 |
| **pedestrian** | 6 | 27 | 0.558 | 0.741 | 0.741 | 0.497 |
| **All** | 361 | 530 | 0.861 | 0.842 | 0.887 | 0.692 |
---

## Per-Class Performance (Yolov11s)
| Class                   | Images | Instances | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|------------------------|:------:|:---------:|:---------:|:------:|:-------:|:------------:|
| **stop-sign**             |   49   |     49    |  0.964    | 0.469  |  0.779  |    0.496     |
| **stop-sign-vandalized** |   16   |     16    |  0.718    | 0.250  |  0.351  |    0.158     |
| **traffic barrel**        |  101   |    198    |  0.696    | 0.578  |  0.683  |    0.353     |
| **traffic cone**         |   86   |     86    |  0.701    | 0.709  |  0.682  |    0.266     |
|  **car-tire**               |  103   |    154    |  0.936    | 0.435  |  0.575  |    0.331     |
| **pedestrian**             |    6   |     27    |  0.524    | 0.370  |  0.365  |    0.150     |
| **All**        |  361   |    530    |  0.756    | 0.469  |  0.572  |    0.293     |
---
## Per-Class Performance (Yolov5s)
| Class                   | Images | Instances | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|------------------------|:------:|:---------:|:---------:|:------:|:-------:|:------------:|
| **All**        |  361   |    530    |   0.600   |  0.579 |  0.517  |    0.244     |
| **Stop Sign**              |   49   |     49    |   0.813   |  0.711 |  0.711  |    0.421     |
| **Stop Sign (Vandalized)** |   16   |     16    |   0.548   |  0.375 |  0.384  |    0.191     |
| **Traffic Barrel**         |  101   |    198    |   0.713   |  0.702 |  0.683  |    0.291     |
| **Traffic Cone**           |   86   |     86    |   0.550   |  0.710 |  0.571  |    0.204     |
|**Car Tire**               |  103   |    154    |   0.694   |  0.456 |  0.467  |    0.245     |
| **Pedestrian**             |    6   |     27    |   0.281   |  0.522 |  0.288  |    0.112     |

## Inference Performance

Average Inference Time: 22.58 ms/image \
Frames Per Second (FPS): 44.28

##  Real-time Evaluation

| Metric | Value |
|:--|:--:|
| **Total Frames Processed** | **522** |
| **Total Time** | **54.84 seconds** |
| **Average Inference Time** | **21.06 ms** |
| **Average FPS** | **47.48** |
| **Minimum Inference Time** | **13.00 ms** |
| **Maximum Inference Time** | **1546.20 ms** |


## ONNX Model

The trained YOLOv11 ONNX model was evaluated on the test dataset.  
Below are the summarized detection metrics:

| Metric | Score |
|:--|:--:|
| **mAP@0.5** | **0.8778** |
| **mAP@0.5:0.95** | **0.7092** |
| **Mean Precision** | **0.8741** |
| **Mean Recall** | **0.8288** |
---

## Per-Class Performance

| Class | Images | Instances | Precision (P) | Recall (R) | mAP@0.5 | mAP@0.5:0.95 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| **stop-sign** | 49 | 49 | 0.977 | 0.878 | 0.937 | 0.688 |
| **stop-sign-vandalized** | 16 | 16 | 0.857 | 0.750 | 0.824 | 0.664 |
| **traffic barrel** | 101 | 198 | 0.829 | 0.843 | 0.882 | 0.723 |
| **traffic cone** | 86 | 86 | 1.000 | 1.000 | 0.995 | 0.983 |
| **car-tire** | 103 | 154 | 0.975 | 0.761 | 0.871 | 0.650 |
| **pedestrian** | 6 | 27 | 0.606 | 0.741 | 0.758 | 0.546 |
| **All** | 361 | 530 | 0.878 | 0.773 | 0.849 | 0.666|
---


## Per class performance with competition scenarios
## W/o finetuning

| Class                  | Images | Instances | Precision (P) | Recall (R) | mAP@50 | mAP@50–95 |
|------------------------|--------|-----------|--------------|-----------|--------|-----------|
| **stop-sign**          | 53     | 53        | 0.937        | 0.844     | 0.947  | 0.686     |
| **stop-sign-fake**   | 7      | 8         | 0.549        | 0.460     | 0.657  | 0.488     |
| **stop-sign-vandalized** | 16 | 16        | 0.801        | 0.757     | 0.789  | 0.623     |
| **traffic barrel**    | 101    | 198       | 0.729        | 0.859     | 0.869  | 0.679     |
| **traffic cone**       | 86     | 86        | 0.989        | 1.000     | 0.995  | 0.982     |
| **car-tire**           | 107    | 158       | 0.976        | 0.768     | 0.922  | 0.638     |
| **pedestrian**         | 10     | 31        | 0.554        | 0.710     | 0.680  | 0.455     |
| **All**                | 379    | 550       | 0.791        | 0.771     | 0.837  | 0.650     |
---
## W finetuning
| Class | Images | Instances | Precision (P) | Recall (R) | mAP@50 | mAP@50–95 |
|------|--------|-----------|---------------|------------|--------|-----------|
| **stop-sign** | 53 | 53 | 0.959 | 0.892 | 0.964 | 0.757 |
| **stop-sign-fake** | 7 | 8 | 0.654 | 1.000 | 0.831 | 0.746 |
| **stop-sign-vandalized** | 16 | 16 | 0.844 | 0.675 | 0.792 | 0.663 |
| **traffic barrel** | 101 | 198 | 0.858 | 0.828 | 0.897 | 0.705 |
| **traffic cone** | 86 | 86 | 0.992 | 1.000 | 0.995 | 0.986 |
| **car-tire** | 107 | 158 | 0.983 | 0.770 | 0.954 | 0.649 |
| **pedestrian** | 10 | 31 | 0.700 | 0.676 | 0.741 | 0.519 |
| **All** | 379 | 550 | 0.801 | 0.834 | 0.868 | 0.717 |




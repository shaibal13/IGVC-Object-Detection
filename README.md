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

The trained YOLOv11 model (`.pt` version) was evaluated directly on the test dataset.  
Below are the summarized detection metrics:

| Metric | Score |
|:--|:--:|
| **mAP@0.5** | **0.887** |
| **mAP@0.5:0.95** | **0.692** |
| **Mean Precision (μ)** | **0.861** |
| **Mean Recall (μ)** | **0.842** |
---

## Per-Class Performance

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
## Inference Performance
Average Inference Time: 21.58 ms/image \
Frames Per Second (FPS): 46.34


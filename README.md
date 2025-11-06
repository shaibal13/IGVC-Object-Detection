# IGVC-Object-Detection

This repository contains scripts for training and evaluating a custom YOLOv11 model on our own dataset.

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
## Evaluate with Test Data
Use test.py to evaluate the trained model:
```bash
python test.py \
  --data Dataset/data.yaml \
  --weights Pretrained_weights/weights/best.pt \
  --imgsz 720 1280 \
  --batch 2 \
  --conf 0.4 \
  --iou 0.3
```
## Real-time Object Detection Evaluation
Run test_from_camera.py for real-time detection
```bash
python test_from_camera.py
```
## Generate an ONNX model to evaluate on edge devices

Run export_onnx.py to generate an ONNX file
```bash
python export_onnx.py
```

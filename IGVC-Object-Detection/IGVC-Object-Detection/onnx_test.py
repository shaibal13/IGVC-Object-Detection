from ultralytics import YOLO

#  Load your ONNX model
model = YOLO("runs/train2/custom_run/weights/best.onnx")

#  Run inference on a folder of 640×640 test images
#    and save annotated results in runs/detect/onnx_preds/
results = model.predict(
    source="/media/shaibal/FILES/Yolov11/Dataset/yolo/images/test",
    imgsz=(720, 1280),   # height, width of your camera frames
    conf=0.25,
    iou=0.45,
    save=True,           # save annotated images
    project="runs",
    name="onnx_preds"
)

#  inspect one result in code
for r in results:
    print(r.scores, r.boxes.xyxy, r.boxes.cls)
    # r.plot() returns an OpenCV image

from ultralytics import YOLO
#Load the YOLO11 model
model = YOLO("runs/train2/custom_run/weights/best.pt")

# # # Export the model to ONNX format
# # model.export(format="onnx")  # creates 'yolo11n.onnx'

model.export(
    format="onnx",
    dynamic=True,   
    simplify=True,    
)

model.export(format="engine")

# Load the exported TensorRT model
tensorrt_model = YOLO("yolo11n.engine")

# Run inference
results = tensorrt_model("/media/shaibal/FILES/Yolov11/Dataset/yolo/images/test/6-jpg_0000_0056_0044_0122_0110_png.rf.425cca0f1386c90ad57c708dbd14b701.jpg")
print(results)
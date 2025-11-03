import cv2
from ultralytics import YOLO

# Load
model = YOLO("runs/train2/custom_run/weights/best.pt")

# Open camera at 1280×720
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Loop over frames, detect & display
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 4) Run inference, telling it our frame is (720,1280)
    results = model.predict(
        source=frame,
        imgsz=(720, 1280),    # (height, width)
        conf=0.4,
        iou=0.3,
        device=0,
        verbose=False
    )

    r = results[0]

    # Draw boxes on the ORIGINAL frame size
    for box in r.boxes:
        # box.xyxy is in original image coords
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf   = box.conf[0]
        label  = f"{model.names[cls_id]} {conf:.2f}"
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, label, (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    #Show the annotated frame
    cv2.imshow("YOLOv11 Live", frame)
    if cv2.waitKey(1) == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()

import os
import yaml
from ultralytics import YOLO

# ───────────────────────────────────────────────────────────────────────────────
#  PATHS: adjust these to match your setup
# ───────────────────────────────────────────────────────────────────────────────

# Path to your custom YOLOv11 checkpoint (the .pt file you trained)
CUSTOM_WEIGHTS = "/media/shaibal/FILES/Yolov11/runs/train2/custom_run/weights/best.pt"

# Directory containing your new, unlabeled images
NEW_IMG_DIR = "/media/shaibal/FILES/IGVC/Datasets/zip_files/dataset_tire_sign/resized_images"

# Directory where you want to save the auto‐generated .txt files
NEW_LABEL_DIR = "/media/shaibal/FILES/Yolov11/new_labels2"

# Confidence threshold for including a detection in the label file
CONF_THRESHOLD = 0.30  # keep only boxes with score ≥ 0.50

# ───────────────────────────────────────────────────────────────────────────────
#  VERIFY FOLDERS EXIST (or create NEW_LABEL_DIR if missing)
# ───────────────────────────────────────────────────────────────────────────────

if not os.path.isdir(NEW_IMG_DIR):
    raise FileNotFoundError(f"Could not find the image folder: {NEW_IMG_DIR}")

os.makedirs(NEW_LABEL_DIR, exist_ok=True)

print(" Images folder:", NEW_IMG_DIR)
print(" Labels folder:", NEW_LABEL_DIR)

# ───────────────────────────────────────────────────────────────────────────────
# 3) LOAD YOUR YOLOv11 MODEL
# ───────────────────────────────────────────────────────────────────────────────

if not os.path.isfile(CUSTOM_WEIGHTS):
    raise FileNotFoundError(f"Could not find your weights at: {CUSTOM_WEIGHTS}")

model = YOLO(CUSTOM_WEIGHTS)
model.conf = CONF_THRESHOLD
print(f" Loaded YOLOv11 weights from: {CUSTOM_WEIGHTS}")
print(f" Set confidence threshold = {model.conf:.2f}")

# ───────────────────────────────────────────────────────────────────────────────
# 4) RUN INFERENCE & SAVE .TXT FOR EACH IMAGE
# ───────────────────────────────────────────────────────────────────────────────

all_files = sorted(os.listdir(NEW_IMG_DIR))
images = [f for f in all_files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
num_images = len(images)

if num_images == 0:
    raise RuntimeError(f"No .jpg/.png images found in {NEW_IMG_DIR}")

print(f" Found {num_images} images to label…")

for idx, img_file in enumerate(images, start=1):
    img_path = os.path.join(NEW_IMG_DIR, img_file)
    results = model(img_path)[0]  # single‐image inference

    # results.boxes.xywhn → numpy array of shape (N, 4): [x_center, y_center, w, h], all normalized to [0,1]
    # results.boxes.cls   → numpy array of shape (N,) with class IDs (as floats)
    # results.boxes.conf  → numpy array of shape (N,) with confidence scores
    boxes_xywhn = results.boxes.xywhn.cpu().numpy()
    class_ids   = results.boxes.cls.cpu().numpy()
    confidences = results.boxes.conf.cpu().numpy()

    yolo_lines = []
    for (x_c, y_c, w, h), cls, conf in zip(boxes_xywhn, class_ids, confidences):
        if conf < model.conf:
            continue
        # Each line: "<class_id> <x_center> <y_center> <width> <height>"
        # All coords are already normalized to [0,1].
        line = f"{int(cls)} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}"
        yolo_lines.append(line)

    # Write .txt with the same base name, e.g. "img001.txt"
    txt_name = os.path.splitext(img_file)[0] + ".txt"
    txt_path = os.path.join(NEW_LABEL_DIR, txt_name)
    with open(txt_path, "w") as f:
        f.write("\n".join(yolo_lines))

    print(f"  [{idx:03d}/{num_images:03d}] {img_file} → {len(yolo_lines)} boxes")

print("\n Done! All new images have YOLO‐format labels in:")
print("   ", NEW_LABEL_DIR)

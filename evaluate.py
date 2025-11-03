# import argparse, random, yaml, cv2
# from pathlib import Path
# from ultralytics import YOLO

# def main(args):
#     # Handle rectangular or square inference size
#     # args.imgsz comes in as a list: [sz] or [h, w]
#     if len(args.imgsz) == 1:
#         imgsz = args.imgsz[0]
#     elif len(args.imgsz) == 2:
#         imgsz = (args.imgsz[0], args.imgsz[1])
#     else:
#         raise ValueError("--imgsz must be one int or two ints (height width)")

#     # Load data.yaml 
    
#     data = yaml.safe_load(open(args.data))
#     test_dir = Path(data['path']) / data['test']
#     names    = data['names']

#     # Load model
#     model = YOLO(args.weights)

#     # Evaluate
#     print("Evaluation")
#     print(args.imgsz)
#     results = model.val(
#         data=args.data,
#         split='test',      
#         imgsz=args.imgsz,
#         batch=args.batch,
#         verbose=True,
#         save=True,         # save annotated test images
#         project=args.project,
#         name=args.name
#     )

#     # Extract detection metrics 
#     precs    = results.box.p            # per-class precision list
#     recs     = results.box.r            # per-class recall list
#     map50    = results.box.map50        
#     map50_95 = results.box.map          

#     print("\n— Test set metrics —")
#     print(f" mAP@0.5       : {map50:.3f}")
#     print(f" mAP@0.5:0.95  : {map50_95:.3f}")
#     print(f" Precision: {sum(precs)/len(precs):.3f}")
#     print(f" Recall: {sum(recs)/len(recs):.3f}")

#     # Gather all test images 
#     all_imgs = [p for p in test_dir.rglob('*') if p.suffix.lower() in ('.jpg', '.jpeg', '.png')]
#     print(f"Found {len(all_imgs)} images in {test_dir}")
#     if not all_imgs:
#         return

#     # Process all images
#     sampled = all_imgs
#     print(f"\n Processing {len(sampled)} test images …")

#     # Directory for visualizations
#     out_dir = Path("runs/predict/sample_test_x3")
#     out_dir.mkdir(parents=True, exist_ok=True)
#     for img_path in sampled:
#         img = cv2.imread(str(img_path))
#         if img is not None:
#             h, w, c = img.shape
#             print(f"Processing {img_path.name}: original shape = ({h}, {w}, {c})")
#         else:
#             print(f"Could not load {img_path.name}")
#         # Inference + save annotated samples
#         for img_path in sampled:
#             res = model.predict(
#                 source=str(img_path),
#                 imgsz=args.imgsz,
#                 conf=args.conf,
#                 iou=args.iou,
#                 verbose=False
#             )[0]

#             vis = res.plot()  # annotated OpenCV image
#             save_path = out_dir / f"{img_path.stem}.jpg"
#             cv2.imwrite(str(save_path), vis)
#             print(f"   • {img_path.name} → {save_path}")

#     print(f"\n Done!")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Evaluate and visualize YOLOv11 test set outputs")
#     parser.add_argument("--data",    type=str, required=True,
#                         help="Path to your data.yaml")
#     parser.add_argument("--weights", type=str, required=True,
#                         help="Path to your trained .pt weights file")
#     parser.add_argument(
#         "--imgsz",
#         nargs='+',
#         type=int,
#         default=[640],
#         help="Inference image size: one int for square or two ints 'height width' for rectangular"
#     )
#     parser.add_argument("--batch",   type=int, default=8,
#                         help="Batch size for evaluation")
#     parser.add_argument("--conf",    type=float, default=0.25,
#                         help="Confidence threshold for predictions")
#     parser.add_argument("--iou",     type=float, default=0.45,
#                         help="NMS IoU threshold for predictions")
#     parser.add_argument("--project", type=str, default="runs",
#                         help="Base folder for saved images")
#     parser.add_argument("--name",    type=str, default="test_preds",
#                         help="Subfolder name under project for saved images")
#     args = parser.parse_args()
#     main(args)


"""
Evaluate a YOLOv11 model on your test split using camera-resolution inference,
then visualize detections on upscaled test images so you can ensure performance
on 1280×720 frames, even though the model was trained on 640×640.

Usage:
    python test_up.py \
      --data Dataset/data.yaml \
      --weights runs/train2/custom_run/weights/best.pt \
      --imgsz 720 1280 \
      --batch 8 \
      --conf 0.25 \
      --iou 0.45 \
      --num 20 \
      --project runs \
      --name sample_test_x2
"""
import argparse
import yaml
import cv2
import random
from pathlib import Path
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="Test YOLOv11 camera pipeline and visualize detections")
    parser.add_argument("--data",    type=str, required=True, help="Path to data.yaml")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained .pt weights")
    parser.add_argument("--imgsz", nargs='+', type=int, default=[640],
                        help="Inference size: one int or two ints 'height width'")
    parser.add_argument("--batch",   type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--conf",    type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou",     type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--num",     type=int, default=10, help="Number of samples to visualize")
    parser.add_argument("--project", type=str, default="runs", help="Base folder for outputs")
    parser.add_argument("--name",    type=str, default="sample_test_x2", help="Sub-folder name under project")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # determine inference shape
    if len(args.imgsz) == 1:
        H = W = args.imgsz[0]
        imgsz = H
        rect = False
    elif len(args.imgsz) == 2:
        H, W = args.imgsz
        imgsz = (H, W)
        rect = True
    else:
        raise ValueError("--imgsz must be one int or two ints: height width")

    # load dataset config and test directory
    data = yaml.safe_load(open(args.data))
    test_dir = Path(data['path']) / data['test']

    # load model
    model = YOLO(args.weights)

    # 1) quantitative evaluation on test split with camera resolution
    print(f" Evaluating TEST split at imgsz={imgsz}, rect={rect} …")
    results = model.val(
        data=args.data,
        split='test',
        imgsz=imgsz,
        rect=rect,
        batch=args.batch,
        verbose=True,
        save=False
    )
    # extract metrics
    precs = results.box.p
    recs  = results.box.r
    mAP50 = results.box.map50
    mAP   = results.box.map
    print("\n— TEST metrics —")
    print(f" mAP@0.5      : {mAP50:.3f}")
    print(f" mAP@0.5:0.95 : {mAP:.3f}")
    print(f" Precision μ : {sum(precs)/len(precs):.3f}")
    print(f" Recall    μ : {sum(recs)/len(recs):.3f}\n")

    # 2) visual check: sample and upscale to camera size
    all_imgs = [p for p in test_dir.rglob('*') if p.suffix.lower() in ('.jpg', '.jpeg', '.png')]
    print(f" Found {len(all_imgs)} test images in {test_dir}")
    if not all_imgs:
        exit(0)

    sampled = random.sample(all_imgs, min(len(all_imgs), args.num))

    out_dir = Path("runs/predict/sample_test_x5")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f" Visualizing {len(sampled)} upscaled frames to {out_dir} …")

    for img_path in sampled:
        orig = cv2.imread(str(img_path))
        if orig is None:
            print(f" Cannot read {img_path}")
            continue
        # upscale to camera resolution
        frame = cv2.resize(orig, (W, H))  # (width, height)
        print(f"Processing {img_path.name}: upscaled to ({H},{W},{frame.shape[2]})")

        # inference
        res = model.predict(
            source=frame,
            imgsz=imgsz,
            conf=args.conf,
            iou=args.iou,
            verbose=False
        )[0]

        # draw boxes and labels
        for box in res.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0])
            conf_score = float(box.conf[0])
            label = f"{model.names[cls_id]} {conf_score:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
            cv2.putText(frame, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        out_path = out_dir / img_path.name
        cv2.imwrite(str(out_path), frame)
        print(f"   • saved {out_path}")

    print("\n Done!")

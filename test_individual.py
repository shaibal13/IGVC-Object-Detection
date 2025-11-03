import argparse, random, yaml, cv2
from pathlib import Path
from ultralytics import YOLO

def main(args):
    #  Load data.yaml to get test folder & class names
    data = yaml.safe_load(open(args.data))
    test_dir = Path(data['path']) / data['test']

    #  Load your trained model
    model = YOLO(args.weights)

    #  Gather all test images recursively
    all_imgs = [p for p in test_dir.rglob('*') if p.suffix.lower() in ('.jpg', '.jpeg', '.png')]
    print(f"🔍 Found {len(all_imgs)} images in {test_dir}")
    if not all_imgs:
        return

    #  Sample images 
    if len(all_imgs) >= args.num:
        sampled = random.sample(all_imgs, args.num)
    else:
        sampled = random.choices(all_imgs, k=args.num)

    # Directory for sampled visualizations
    out_dir = Path(args.project) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 5) Inference + save annotated samples
    print(f"\n Running detection on {len(sampled)} images …")
    for img_path in sampled:
        res = model.predict(
            source=str(img_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            verbose=False
        )[0]
        vis = res.plot()  # annotated OpenCV image
        save_path = out_dir / f"{img_path.stem}.jpg"
        cv2.imwrite(str(save_path), vis)
        print(f"   • {img_path.name} → {save_path}")

    print(f"\n Done! Annotated images saved in {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLOv11 detection on images and save results")
    parser.add_argument("--data",    type=str, required=True,
                        help="Path to your data.yaml")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to your trained .pt weights file")
    parser.add_argument("--imgsz",   type=int, default=640,
                        help="Inference image size (square)")
    parser.add_argument("--conf",    type=float, default=0.5,
                        help="Confidence threshold for predictions")
    parser.add_argument("--iou",     type=float, default=0.45,
                        help="NMS IoU threshold for predictions")
    parser.add_argument("--num",     type=int, default=100,
                        help="Number of random samples to process")
    parser.add_argument("--project", type=str, default="runs",
                        help="Base folder for saved images")
    parser.add_argument("--name",    type=str, default="detections2",
                        help="Subfolder name under project for saved images")
    args = parser.parse_args()
    main(args)

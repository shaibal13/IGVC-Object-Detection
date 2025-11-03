import argparse, random, yaml, cv2
from pathlib import Path
from ultralytics import YOLO

def main(args):
    #  Load data.yaml to get test folder & class names
    data = yaml.safe_load(open(args.data))
    test_dir = Path(data['path']) / data['test']
    names    = data['names']

    #  Load your trained model
    model = YOLO(args.weights)

    #  Evaluate on the full test set and print metrics
    print(" Running full test evaluation …")
    results = model.val(
        data=args.data,
        split='test',      # use the 'test:' entry in data.yaml
        imgsz=args.imgsz,
        batch=args.batch,
        verbose=True,
        save=True,         # save annotated test images
        project=args.project,
        name=args.name
    )

    # Extract detection metrics using updated Metric API
    precs    = results.box.p            # per-class precision list
    recs     = results.box.r            # per-class recall list
    map50    = results.box.map50        # mAP @ IoU=0.5 (numpy float)
    map50_95 = results.box.map          # mAP @ IoU=0.5:0.95 (numpy float)

    print("\n— Test set metrics —")
    print(f" mAP@0.5       : {map50:.3f}")
    print(f" mAP@0.5:0.95  : {map50_95:.3f}")
    print(f" Precision: {sum(precs)/len(precs):.3f}")
    print(f" Recall: {sum(recs)/len(recs):.3f}")

    #  Gather all test images recursively
    all_imgs = [p for p in test_dir.rglob('*') if p.suffix.lower() in ('.jpg', '.jpeg', '.png')]
    print(f" Found {len(all_imgs)} images in {test_dir}")
    if not all_imgs:
        return

    #  Sample images (with replacement if needed)
    if len(all_imgs) >= args.num:
        sampled = random.sample(all_imgs, args.num)
    else:
        sampled = random.choices(all_imgs, k=args.num)

    # Directory for sampled visualizations
    out_dir = Path("runs/predict/sample_test_x2")
    out_dir.mkdir(parents=True, exist_ok=True)

    #  Inference + save annotated samples
    print(f"\n Inference on {len(sampled)} random test images …")
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

    print(f"\n Done! Annotated samples in {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate and visualize YOLOv11 test set outputs")
    parser.add_argument("--data",    type=str, required=True,
                        help="Path to your data.yaml")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to your trained .pt weights file")
    parser.add_argument("--imgsz",   type=int, default=640,
                        help="Inference image size (square)")
    parser.add_argument("--batch",   type=int, default=8,
                        help="Batch size for evaluation")
    parser.add_argument("--conf",    type=float, default=0.25,
                        help="Confidence threshold for predictions")
    parser.add_argument("--iou",     type=float, default=0.45,
                        help="NMS IoU threshold for predictions")
    parser.add_argument("--num",     type=int, default=100,
                        help="Number of random samples to visualize")
    parser.add_argument("--project", type=str, default="runs",
                        help="Base folder for saved images")
    parser.add_argument("--name",    type=str, default="test_preds",
                        help="Subfolder name under project for saved images")
    args = parser.parse_args()
    main(args)

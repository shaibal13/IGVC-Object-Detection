import argparse
import time
import random
import yaml
import cv2
from pathlib import Path
from ultralytics import YOLO


def evaluate_single_image(model, image_path, args):
    """Evaluate on a single image."""
    print(f"\n{'='*60}")
    print(f"Evaluating single image: {image_path}")
    print(f"{'='*60}")
    
    img_path = Path(image_path)
    if not img_path.exists():
        print(f"Error: Image not found at {image_path}")
        return
    
    # Run inference with timing
    start_time = time.time()
    results = model.predict(
        source=str(img_path),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        verbose=False
    )[0]
    inference_time = time.time() - start_time
    
    # Get detection info
    num_detections = len(results.boxes) if results.boxes is not None else 0
    print(f"\nInference Time: {inference_time*1000:.2f} ms")
    print(f"Number of detections: {num_detections}")
    
    if num_detections > 0:
        print("\nDetections:")
        for i, box in enumerate(results.boxes):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]
            print(f"  {i+1}. {label}: {conf:.3f}")
    
    # Save annotated image
    out_dir = Path(args.output_dir) if args.output_dir else Path("runs/predict/single_image")
    out_dir.mkdir(parents=True, exist_ok=True)
    vis = results.plot()
    save_path = out_dir / f"{img_path.stem}_result.jpg"
    cv2.imwrite(str(save_path), vis)
    print(f"\nAnnotated image saved to: {save_path}")
    
    return inference_time


def evaluate_dataset(model, data_path, args):
    """Evaluate on dataset using model.val() and show metrics."""
    print(f"\n{'='*60}")
    print(f"Evaluating on dataset: {data_path}")
    print(f"{'='*60}")
    
    # Load data.yaml to get test folder & class names
    data = yaml.safe_load(open(data_path))
    test_dir = Path(data['path']) / data['test']
    names = data['names']
    
    # Evaluate on the full test set and print metrics
    print("\nRunning full test evaluation...")
    start_time = time.time()
    results = model.val(
        data=data_path,
        split='test',
        imgsz=args.imgsz,
        batch=args.batch,
        verbose=True,
        save=True,
        project=args.project,
        name=args.name
    )
    total_eval_time = time.time() - start_time
    
    # Extract detection metrics
    precs = results.box.p
    recs = results.box.r
    map50 = results.box.map50
    map50_95 = results.box.map
    
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Total Evaluation Time: {total_eval_time:.2f} seconds")
    print(f"mAP@0.5       : {map50:.3f}")
    print(f"mAP@0.5:0.95  : {map50_95:.3f}")
    print(f"Precision (μ) : {sum(precs)/len(precs):.3f}")
    print(f"Recall (μ)    : {sum(recs)/len(recs):.3f}")
    
    # Per-class metrics
    if len(names) > 0:
        print(f"\nPer-class metrics:")
        for i, name in enumerate(names):
            if i < len(precs):
                print(f"  {name}: Precision={precs[i]:.3f}, Recall={recs[i]:.3f}")
    
    # Gather all test images recursively
    all_imgs = [p for p in test_dir.rglob('*') if p.suffix.lower() in ('.jpg', '.jpeg', '.png')]
    print(f"\nFound {len(all_imgs)} images in {test_dir}")
    
    if all_imgs and args.sample_images > 0:
        # Sample images for visualization
        if len(all_imgs) >= args.sample_images:
            sampled = random.sample(all_imgs, args.sample_images)
        else:
            sampled = random.choices(all_imgs, k=args.sample_images)
        
        # Directory for sampled visualizations
        out_dir = Path(args.project) / args.name / "samples"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Inference + save annotated samples with timing
        print(f"\nProcessing {len(sampled)} sample images for visualization...")
        inference_times = []
        
        for img_path in sampled:
            start_time = time.time()
            res = model.predict(
                source=str(img_path),
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                verbose=False
            )[0]
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            vis = res.plot()
            save_path = out_dir / f"{img_path.stem}.jpg"
            cv2.imwrite(str(save_path), vis)
            print(f"  • {img_path.name} → {save_path} ({inference_time*1000:.2f} ms)")
        
        avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0
        print(f"\nAverage inference time per image: {avg_inference_time*1000:.2f} ms")
        print(f"FPS: {1.0/avg_inference_time:.2f}" if avg_inference_time > 0 else "N/A")
    
    print(f"\nDone! Results saved in {args.project}/{args.name}")
    
    return {
        'map50': map50,
        'map50_95': map50_95,
        'precision': sum(precs)/len(precs),
        'recall': sum(recs)/len(recs),
        'total_time': total_eval_time
    }


def evaluate_camera(model, args):
    """Evaluate on live camera feed."""
    print(f"\n{'='*60}")
    print("Starting live camera evaluation")
    print(f"{'='*60}")
    print("Press ESC to quit")
    
    # Open camera
    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera_index}")
        return
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
    
    # Determine imgsz
    if isinstance(args.imgsz, (list, tuple)) and len(args.imgsz) == 2:
        imgsz = tuple(args.imgsz)
    elif isinstance(args.imgsz, (list, tuple)) and len(args.imgsz) == 1:
        imgsz = args.imgsz[0]
    else:
        imgsz = args.imgsz
    
    print(f"Camera resolution: {args.camera_width}x{args.camera_height}")
    print(f"Inference size: {imgsz}")
    
    frame_count = 0
    inference_times = []
    fps_start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from camera")
                break
            
            # Run inference with timing
            start_time = time.time()
            results = model.predict(
                source=frame,
                imgsz=imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                verbose=False
            )[0]
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            frame_count += 1
            
            # Draw boxes on the frame
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = f"{model.names[cls_id]} {conf:.2f}"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
                cv2.putText(frame, label, (x1, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Display FPS and inference time
            current_fps = 1.0 / inference_time if inference_time > 0 else 0
            avg_inference_time = sum(inference_times[-30:]) / len(inference_times[-30:]) if len(inference_times) > 0 else 0
            avg_fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
            
            info_text = f"FPS: {current_fps:.1f} | Avg FPS: {avg_fps:.1f} | Time: {inference_time*1000:.1f}ms"
            cv2.putText(frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Show the annotated frame
            cv2.imshow("YOLOv11 Live Detection", frame)
            
            # ESC to quit
            if cv2.waitKey(1) == 27:
                break
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # Print statistics
        if inference_times:
            avg_time = sum(inference_times) / len(inference_times)
            total_time = time.time() - fps_start_time
            print(f"\n{'='*60}")
            print("CAMERA EVALUATION STATISTICS")
            print(f"{'='*60}")
            print(f"Total frames processed: {frame_count}")
            print(f"Total time: {total_time:.2f} seconds")
            print(f"Average inference time: {avg_time*1000:.2f} ms")
            print(f"Average FPS: {1.0/avg_time:.2f}")
            print(f"Min inference time: {min(inference_times)*1000:.2f} ms")
            print(f"Max inference time: {max(inference_times)*1000:.2f} ms")


def main():
    parser = argparse.ArgumentParser(
        description=" YOLOv11 evaluation tool - supports single image, dataset, and live camera",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image
  python evaluate.py --weights model.pt --image path/to/image.jpg
  
  # Dataset evaluation
  python evaluate.py --weights model.pt --dataset Dataset/data.yaml
  
  # Live camera
  python evaluate.py --weights model.pt --camera

        """
    )
    
    # Required arguments
    parser.add_argument("--weights", type=str, required=True,
                       help="Path to trained .pt weights file")
    
    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--image", type=str,
                           help="Path to single image for evaluation")
    mode_group.add_argument("--dataset", type=str,
                           help="Path to data.yaml for dataset evaluation")
    mode_group.add_argument("--camera", action="store_true",
                           help="Use live camera feed")
    
    # Model parameters
    parser.add_argument("--imgsz", type=int, nargs='+', default=[640],
                       help="Inference image size: one int for square or two ints 'height width' for rectangular")
    parser.add_argument("--conf", type=float, default=0.25,
                       help="Confidence threshold for predictions")
    parser.add_argument("--iou", type=float, default=0.45,
                       help="NMS IoU threshold for predictions")
    parser.add_argument("--device", type=str, default="",
                       help="Device to run inference on (e.g., '0' for GPU, 'cpu' for CPU)")
    
    # Dataset-specific arguments
    parser.add_argument("--batch", type=int, default=8,
                       help="Batch size for dataset evaluation")
    parser.add_argument("--sample_images", type=int, default=10,
                       help="Number of sample images to visualize for dataset evaluation")
    parser.add_argument("--project", type=str, default="runs",
                       help="Base folder for saved results")
    parser.add_argument("--name", type=str, default="evaluation",
                       help="Subfolder name under project for saved results")
    
    # Camera-specific arguments
    parser.add_argument("--camera_index", type=int, default=0,
                       help="Camera index (default: 0)")
    parser.add_argument("--camera_width", type=int, default=1280,
                       help="Camera frame width")
    parser.add_argument("--camera_height", type=int, default=720,
                       help="Camera frame height")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for single image results")
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.weights}...")
    model = YOLO(args.weights)
    print("Model loaded successfully!")
    
    # Process imgsz argument
    if isinstance(args.imgsz, list):
        if len(args.imgsz) == 1:
            args.imgsz = args.imgsz[0]
        elif len(args.imgsz) == 2:
            args.imgsz = tuple(args.imgsz)
        else:
            raise ValueError("--imgsz must be one int or two ints (height width)")
    
    # Route to appropriate evaluation function
    if args.camera:
        evaluate_camera(model, args)
    elif args.dataset:
        evaluate_dataset(model, args.dataset, args)
    elif args.image:
        evaluate_single_image(model, args.image, args)
    else:
        parser.error("Must specify one of: --image, --dataset, or --camera")


if __name__ == "__main__":
    main()


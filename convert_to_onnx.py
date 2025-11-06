"""
Export YOLO PyTorch model to ONNX format

This script converts a trained YOLO .pt model to ONNX format for deployment.

Usage Examples:
    # Basic usage with required arguments
    python convert_to_onnx.py --weights /path/to/model.pt
    
    # Full example with all options
    python convert_to_onnx.py \
        --weights runs/train2/custom_run/weights/best.pt \
        --imgsz 640 \
        --dynamic \
        --simplify \
        --half
"""

from ultralytics import YOLO
import argparse
from pathlib import Path


def main(args):
    # Load the YOLO model
    print(f"Loading model from: {args.weights}")
    model = YOLO(args.weights)
    
    # Prepare export arguments
    export_kwargs = {
        "format": "onnx",
    }
    
    # Add optional parameters
    if args.imgsz:
        export_kwargs["imgsz"] = args.imgsz
    
    if args.dynamic:
        export_kwargs["dynamic"] = True
    
    if args.simplify:
        export_kwargs["simplify"] = True
    
    if args.half:
        export_kwargs["half"] = True
    
    if args.workspace:
        export_kwargs["workspace"] = args.workspace
    
    # Export to ONNX
    print("Exporting model to ONNX format...")
    model.export(**export_kwargs)
    
    # Determine output filename
    weights_path = Path(args.weights)
    output_name = weights_path.stem + ".onnx"
    output_path = weights_path.parent / output_name
    
    print(f"\n ONNX model exported successfully!")
    print(f"   Output: {output_path}")
    
    if args.verify:
        print("\nVerifying ONNX model...")
        try:
            onnx_model = YOLO(str(output_path))
            print(" ONNX model loaded successfully!")
        except Exception as e:
            print(f" Error loading ONNX model: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export YOLO PyTorch model to ONNX format")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to PyTorch model file (.pt)")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="Image size for export (default: model's training size)")
    parser.add_argument("--dynamic", action="store_true",
                        help="Enable dynamic axes for variable input sizes")
    parser.add_argument("--simplify", action="store_true",
                        help="Simplify ONNX model using onnxsim")
    parser.add_argument("--half", action="store_true",
                        help="Export FP16 half-precision model")
    parser.add_argument("--workspace", type=int, default=None,
                        help="TensorRT workspace size in GB (not used for ONNX)")
    parser.add_argument("--verify", action="store_true",
                        help="Verify the exported ONNX model by loading it")
    
    args = parser.parse_args()
    main(args)
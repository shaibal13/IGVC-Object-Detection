"""
ONNX Model Evaluation and Visualization Script

This script evaluates an ONNX YOLO model on a test dataset.

Usage Examples:
    # Basic usage with required arguments
    python evaluate_onnx_model.py --data /path/to/data.yaml --weights /path/to/model.onnx
    
    # Full example with all options
    python evaluate_onnx_model.py \
        --data /path/to/data.yaml \
        --weights /path/to/model.onnx \
        --imgsz 640 \
        --batch 8 \
        --conf 0.25 \
        --iou 0.45 \
        --split test \
        --project runs \
        --name onnx_eval \
        --output-dir runs/onnx_eval \
        --show

"""

from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
import yaml
import argparse
from pathlib import Path


def main(args):
    # Load data configuration
    data = yaml.safe_load(open(args.data))
    class_names = data['names']

    # Load ONNX model
    print(f"Loading ONNX model from: {args.weights}")
    model = YOLO(args.weights)

    # Evaluate ONNX model on test set
    print(f"Evaluating ONNX model on {args.split} set...")
    results = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        verbose=True,
        save=True,
        project=args.project,
        name=args.name
    )

    # Extract metrics
    precs = results.box.p  # per-class precision
    recs = results.box.r   # per-class recall
    map50 = results.box.map50  # mAP @ IoU=0.5
    map50_95 = results.box.map  # mAP @ IoU=0.5:0.95

    # Extract speed metrics
    speed = results.speed
    # Speed dict contains: 'preprocess', 'inference', 'postprocess' in ms
    inference_time_ms = speed.get('inference', 0)  # Average inference time per image in ms
    total_time_ms = speed.get('preprocess', 0) + speed.get('inference', 0) + speed.get('postprocess', 0)
    avg_inference_time_ms = inference_time_ms if inference_time_ms > 0 else total_time_ms
    
    # Calculate FPS
    fps = 1000.0 / avg_inference_time_ms if avg_inference_time_ms > 0 else 0

    # Convert to numpy arrays if needed
    precs = np.array(precs) if not isinstance(precs, np.ndarray) else precs
    recs = np.array(recs) if not isinstance(recs, np.ndarray) else recs

    # Print metrics
    print("\n" + "="*50)
    print("ONNX Model Test Results")
    print("="*50)
    print(f"mAP@0.5      : {map50:.4f}")
    print(f"mAP@0.5:0.95 : {map50_95:.4f}")
    print(f"Mean Precision: {np.mean(precs):.4f}")
    print(f"Mean Recall    : {np.mean(recs):.4f}")
    print("-"*50)
    print("Inference Performance")
    print(f"Average Inference Time: {avg_inference_time_ms:.2f} ms/image")
    print(f"Frames Per Second (FPS): {fps:.2f}")
    print("="*50)

    # Create visualization graphs
    fig = plt.figure(figsize=(16, 10))

    # Overall Metrics Bar Chart
    ax1 = plt.subplot(2, 3, 1)
    metrics = ['mAP@0.5', 'mAP@0.5:0.95', 'Mean Precision', 'Mean Recall']
    values = [map50, map50_95, np.mean(precs), np.mean(recs)]
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    bars = ax1.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax1.set_title('Overall Metrics', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 1])
    ax1.grid(axis='y', alpha=0.3)
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

    # Per-Class Precision
    ax2 = plt.subplot(2, 3, 2)
    class_indices = list(range(len(precs)))
    bars2 = ax2.barh(class_indices, precs, color='#e74c3c', alpha=0.7, edgecolor='black')
    ax2.set_yticks(class_indices)
    ax2.set_yticklabels([class_names.get(i, f'Class {i}') for i in class_indices])
    ax2.set_xlabel('Precision', fontsize=12, fontweight='bold')
    ax2.set_title('Per-Class Precision', fontsize=14, fontweight='bold')
    ax2.set_xlim([0, 1])
    ax2.grid(axis='x', alpha=0.3)
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars2, precs)):
        ax2.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                 f'{val:.3f}', va='center', fontweight='bold')

    # Per-Class Recall
    ax3 = plt.subplot(2, 3, 3)
    bars3 = ax3.barh(class_indices, recs, color='#3498db', alpha=0.7, edgecolor='black')
    ax3.set_yticks(class_indices)
    ax3.set_yticklabels([class_names.get(i, f'Class {i}') for i in class_indices])
    ax3.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax3.set_title('Per-Class Recall', fontsize=14, fontweight='bold')
    ax3.set_xlim([0, 1])
    ax3.grid(axis='x', alpha=0.3)
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars3, recs)):
        ax3.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                 f'{val:.3f}', va='center', fontweight='bold')

    # Precision vs Recall Comparison
    ax4 = plt.subplot(2, 3, 4)
    x_pos = np.arange(len(class_indices))
    width = 0.35
    bars4a = ax4.bar(x_pos - width/2, precs, width, label='Precision', color='#e74c3c', alpha=0.7, edgecolor='black')
    bars4b = ax4.bar(x_pos + width/2, recs, width, label='Recall', color='#3498db', alpha=0.7, edgecolor='black')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([class_names.get(i, f'C{i}') for i in class_indices], rotation=45, ha='right')
    ax4.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax4.set_title('Precision vs Recall by Class', fontsize=14, fontweight='bold')
    ax4.set_ylim([0, 1])
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)

    # F1-Score per Class
    ax5 = plt.subplot(2, 3, 5)
    f1_scores = 2 * (precs * recs) / (precs + recs + 1e-6)  # F1 = 2 * (P * R) / (P + R)
    bars5 = ax5.barh(class_indices, f1_scores, color='#2ecc71', alpha=0.7, edgecolor='black')
    ax5.set_yticks(class_indices)
    ax5.set_yticklabels([class_names.get(i, f'Class {i}') for i in class_indices])
    ax5.set_xlabel('F1-Score', fontsize=12, fontweight='bold')
    ax5.set_title('Per-Class F1-Score', fontsize=14, fontweight='bold')
    ax5.set_xlim([0, 1])
    ax5.grid(axis='x', alpha=0.3)
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars5, f1_scores)):
        ax5.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                 f'{val:.3f}', va='center', fontweight='bold')

    # Summary Statistics Table
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    table_data = [
        ['Metric', 'Value'],
        ['mAP@0.5', f'{map50:.4f}'],
        ['mAP@0.5:0.95', f'{map50_95:.4f}'],
        ['Mean Precision', f'{np.mean(precs):.4f}'],
        ['Mean Recall', f'{np.mean(recs):.4f}'],
        ['Mean F1-Score', f'{np.mean(f1_scores):.4f}'],
        ['Best Precision', f'{np.max(precs):.4f}'],
        ['Best Recall', f'{np.max(recs):.4f}'],
        ['Worst Precision', f'{np.min(precs):.4f}'],
        ['Worst Recall', f'{np.min(recs):.4f}'],
        ['Avg Inference Time', f'{avg_inference_time_ms:.2f} ms'],
        ['FPS', f'{fps:.2f}'],
    ]
    table = ax6.table(cellText=table_data, cellLoc='left', loc='center',
                      colWidths=[0.5, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    for i in range(len(table_data)):
        for j in range(2):
            cell = table[(i, j)]
            if i == 0:  # Header row
                cell.set_facecolor('#34495e')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#ecf0f1' if i % 2 == 0 else 'white')
    ax6.set_title('Summary Statistics', fontsize=14, fontweight='bold', pad=20)

    plt.suptitle('ONNX Model Test Results - Performance Metrics', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Save the figure
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "test_results_graph.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nGraphs saved to: {output_path}")

    # Show the plot
    if args.show:
        plt.show()
    else:
        plt.close()

    # Print per-class metrics
    print("\n" + "="*50)
    print("Per-Class Metrics")
    print("="*50)
    print(f"{'Class':<25} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-"*50)
    for i in range(len(precs)):
        class_name = class_names.get(i, f'Class {i}')
        print(f"{class_name:<25} {precs[i]:<12.4f} {recs[i]:<12.4f} {f1_scores[i]:<12.4f}")
    print("="*50)
    print("\n" + "="*50)
    print("Inference Performance Summary")
    print("="*50)
    print(f"Average Inference Time: {avg_inference_time_ms:.2f} ms/image")
    print(f"Frames Per Second (FPS): {fps:.2f}")
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ONNX model and generate performance graphs")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to data.yaml file")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to ONNX model file (.onnx)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Image size for inference (default: 640)")
    parser.add_argument("--batch", type=int, default=8,
                        help="Batch size for evaluation (default: 8)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold (default: 0.25)")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="IoU threshold for NMS (default: 0.45)")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"],
                        help="Dataset split to evaluate (default: test)")
    parser.add_argument("--project", type=str, default="runs",
                        help="Project directory for saving results (default: runs)")
    parser.add_argument("--name", type=str, default="onnx_eval",
                        help="Name for the evaluation run (default: onnx_eval)")
    parser.add_argument("--output-dir", type=str, default="runs/onnx_eval",
                        help="Directory to save graphs (default: runs/onnx_eval)")
    parser.add_argument("--show", action="store_true",
                        help="Display the plot (default: False)")
    
    args = parser.parse_args()
    main(args)

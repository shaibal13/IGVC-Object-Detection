import argparse
from ultralytics import YOLO
import subprocess
def parse_args():
    p = argparse.ArgumentParser(description="Train YOLOv11 on a custom dataset")
    p.add_argument(
        "-d", "--data",
        type=str,
        required=True,
        help="Path to your data.yaml (e.g. Dataset/data.yaml)"
    )
    p.add_argument(
        "-e", "--epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    p.add_argument(
        "-s", "--imgsz",
        type=int,
        default=640,
        help="Image size for training (square) or height if rectangular"
    )
    p.add_argument(
        "--rect",
        action="store_true",
        help="Enable rectangular training (for non-square imgsz)"
    )
    p.add_argument(
        "-b", "--batch",
        type=int,
        default=8,
        help="Batch size"
    )
    p.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA device index (or use 'cpu')"
    )
    p.add_argument(
        "-p", "--project",
        type=str,
        default="runs/train3",
        help="Directory to save training results"
    )
    p.add_argument(
        "-n", "--name",
        type=str,
        default="yolo_custom",
        help="Sub-folder name under project"
    )
    return p.parse_args()


# 1) Define your “after-epoch” callback
def after_epoch(trainer):
    """
    This function is called at the end of each full fit epoch
    (train+val). You can inspect trainer.epoch, locate the saved
    checkpoint, then launch any other script you like.
    """
    epoch = trainer.epoch  # current epoch number (1-based)
    # build path to the checkpoint saved by save_period=1
    ckpt = trainer.save_dir / 'weights' / f'epoch{epoch}.pt'
    if ckpt.exists():
        # call your external script, passing the checkpoint path
        subprocess.run([
            'python', 'visualize_outputs.py',
            '--weights', str(ckpt),
            '--epoch', str(epoch)
        ])

if __name__ == "__main__":
    args = parse_args()

    # 1) Load pretrained YOLOv11-Nano
    model = YOLO("yolo11x.pt")
    model.add_callback("on_fit_epoch_end", after_epoch)

    # 2) Train
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=(args.imgsz, args.imgsz) if not args.rect else (args.imgsz, int(args.imgsz * 1280/720)),
        batch=args.batch,
        rect=args.rect,
        device=args.device,
        project=args.project,
        name=args.name
    )

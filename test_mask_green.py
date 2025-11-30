import time
from datetime import datetime
from pathlib import Path
import torch
import os
import sys

CURRENT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.join(CURRENT_DIR, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from ai_tracking import run_yolo_on_video

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"./data/output/{timestamp}_mask_green")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print("Testing MASK_GREEN strategy with transparent PNGs\n")
    
    input_video = r"./data/input/raw-video.mp4"
    
    start_time = time.time()
    frame_count = 0
    
    for idx, result in enumerate(run_yolo_on_video(
        input_path=input_video,
        output_dir=str(output_dir),
        frames=[1, 10, 216],
        device=device,
        debug=True,
        clustering_k_min=2,
        clustering_k_max=5,
        hsv_strategy="mask_green",  # NEW STRATEGY
    )):
        frame_count = idx + 1
        num_detections = len(result.boxes) if result.boxes else 0
        if idx % 30 == 0:
            print(f"Frame {idx}: {num_detections} person(s) detected")
    
    elapsed_ms = (time.time() - start_time) * 1000
    fps = frame_count / (elapsed_ms / 1000) if elapsed_ms > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Frames processed: {frame_count}")
    print(f"Time elapsed: {elapsed_ms:.2f} ms ({elapsed_ms/1000:.2f} seconds)")
    print(f"Processing speed: {fps:.2f} FPS")
    print(f"Output saved to {output_dir}")
    print(f"Check crops - they should be PNG with transparency!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

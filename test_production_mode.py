"""Test script to verify histogram visualization is disabled in production mode (frames=None)."""
import time
from datetime import datetime
from pathlib import Path

import torch
from ai_tracking import run_yolo_on_video


def main():
    # Auto-detect GPU
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"./data/output/production_test_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Define paths
    input_video = r"./data/input/raw-video.mp4"
    
    # Production mode: Process only first 100 frames without specifying frames parameter
    # This should compute histograms but NOT save visualizations
    start_time = time.time()
    frame_count = 0
    
    for idx, result in enumerate(run_yolo_on_video(
        input_path=input_video,
        output_dir=str(output_dir),
        save_annotated=False,
        save_crops=True,
        frames=None,  # Production mode - no frames filter
        device=device
    )):
        frame_count = idx + 1
        if idx % 30 == 0:  # Progress update every ~1 second
            num_detections = len(result.boxes) if result.boxes else 0
            print(f"Frame {idx}: {num_detections} person(s) detected")
        
        # Stop after 100 frames for quick test
        if idx >= 99:
            break
    
    elapsed_ms = (time.time() - start_time) * 1000
    fps = frame_count / (elapsed_ms / 1000) if elapsed_ms > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Frames processed: {frame_count}")
    print(f"Time elapsed: {elapsed_ms:.2f} ms ({elapsed_ms/1000:.2f} seconds)")
    print(f"Processing speed: {fps:.2f} FPS")
    print(f"Output saved to {output_dir}")
    
    # Check if histogram visualizations were created (they shouldn't be)
    hist_files = list(output_dir.glob("**/*_hist.png"))
    print(f"\nHistogram visualization files found: {len(hist_files)}")
    if len(hist_files) == 0:
        print("✓ Production mode working correctly - no histogram visualizations saved")
    else:
        print("✗ Warning: Histogram visualizations were created in production mode")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

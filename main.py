import time
from datetime import datetime
from pathlib import Path

import torch
from ai_tracking import run_yolo_on_video, run_yolo_tracking_mode


def main():
    # Auto-detect GPU
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"./data/output/{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Define paths
    input_video = r"./data/input/raw-video.mp4"
    tracked_output = output_dir / "tracked-video.mp4"
    
    # Option 1: Frame-by-frame inference with annotations
    start_time = time.time()
    frame_count = 0
    
    for idx, result in enumerate(run_yolo_on_video(
        input_path=input_video,
        output_dir=str(output_dir),
        frames=list(range(1, 25)),
        device=device,
        debug=True,
        clustering_k_min=2,
        clustering_k_max=5,
    )):
        frame_count = idx + 1
        num_detections = len(result.boxes) if result.boxes else 0
        if idx % 30 == 0:  # Progress update every ~1 second
            print(f"Frame {idx}: {num_detections} person(s) detected")
    
    elapsed_ms = (time.time() - start_time) * 1000
    fps = frame_count / (elapsed_ms / 1000) if elapsed_ms > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Frames processed: {frame_count}")
    print(f"Time elapsed: {elapsed_ms:.2f} ms ({elapsed_ms/1000:.2f} seconds)")
    print(f"Processing speed: {fps:.2f} FPS")
    print(f"Output saved to {output_dir}")
    print(f"{'='*60}")
    
    # Option 2: Tracking mode with persistent IDs
    # run_yolo_tracking_mode(
    #     input_path=input_video,
    #     output_path=str(tracked_output),
    #     device=device
    # )
    # print(f"Tracking complete! Video saved to {tracked_output}")

if __name__ == "__main__":
    main()

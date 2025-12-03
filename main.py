"""Command line interface for executing the soccer tracking pipeline."""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Sequence

import torch

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ai_tracking import RunnerConfig, YoloVideoRunner  # noqa: E402


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the tracking application."""
    parser = argparse.ArgumentParser(description="Run YOLO-based soccer tracking")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("./data/input/raw-video.mp4"),
        help="Path to the input video",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Directory to store artifacts (defaults to timestamped folder)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Ultralytics model weights to load",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for inference (auto-detect when omitted)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        nargs="*",
        default=None,
        help="Optional list of frame indices to process",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug exports (crops, histograms, YOLO video)",
    )
    parser.add_argument(
        "--hsv-strategy",
        choices=["crop_torso", "mask_green"],
        default="crop_torso",
        help="Histogram strategy to isolate jerseys",
    )
    parser.add_argument(
        "--torso-height-ratio",
        type=float,
        default=0.4,
        help="Fraction of crop height used to focus on torsos",
    )
    parser.add_argument(
        "--torso-width-ratio",
        type=float,
        default=0.6,
        help="Fraction of crop width used to focus on torsos",
    )
    return parser.parse_args(argv)


def build_output_dir(root: Path | None) -> Path:
    """Return an output directory, creating a timestamped folder when needed."""
    if root is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = Path("./data/output") / timestamp
    root.mkdir(parents=True, exist_ok=True)
    return root


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_arguments(argv)
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    output_dir = build_output_dir(args.output)

    config = RunnerConfig(
        model_source=args.model,
        device=device,
        debug=args.debug,
        output_dir=output_dir,
        torso_height_ratio=args.torso_height_ratio,
        torso_width_ratio=args.torso_width_ratio,
        hsv_strategy=args.hsv_strategy,
    )

    runner = YoloVideoRunner(config=config)
    start_time = time.perf_counter()
    frames_processed = 0

    for frames_processed, result in enumerate(runner.run(args.input, frames=args.frames), start=1):
        if frames_processed % 30 == 0:
            detections = len(result.boxes) if result.boxes is not None else 0
            print(f"Frame {frames_processed}: {detections} detections")

    elapsed = time.perf_counter() - start_time
    fps = frames_processed / elapsed if elapsed > 0 else 0.0

    print("\n" + "=" * 60)
    print("Processing complete")
    print(f"Frames processed: {frames_processed}")
    print(f"Wall time: {elapsed:.2f}s | Throughput: {fps:.2f} FPS")
    print(f"Artifacts saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

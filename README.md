# Soccer Tracking AI

Minimal scaffolding to run a pretrained YOLO model on every frame of a video, detecting only people by default.

## Installation

```bash
uv pip install -e .
```

## Debugging

To debug the project in VS Code:

1. Start the debug server:
```bash
uv run python -m debugpy --listen 5678 --wait-for-client .\main.py
```

2. Add this configuration to your `.vscode/launch.json`:
```json
{
  "name": "Python Debugger: Attach to uv debugpy",
  "type": "debugpy",
  "request": "attach",
  "connect": {
    "host": "localhost",
    "port": 5678
  },
  "justMyCode": true
}
```

3. Set breakpoints and attach the debugger using "Python Debugger: Attach to uv debugpy" from the Run and Debug panel.

## Usage

```python
from ai_tracking import run_yolo_on_video

for frame_idx, result in enumerate(run_yolo_on_video("/path/to/video.mp4")):
    print(f"Frame {frame_idx}: {result}")
```

The iterator yields the raw `ultralytics.engine.results.Results` objects returned by YOLO. By default, only **person detections** (class 0) are returned. You can inspect bounding boxes, confidences, and classes directly or chain extra processing later.

To detect other classes, override the default filtering:

```python
for result in run_yolo_on_video("/path/to/video.mp4", yolo_kwargs={"classes": [0, 2, 7]}):
    pass  # detects persons, cars, and trucks
```

### Saving annotated videos

To persist an annotated video using YOLO's built-in `plot()` overlays, enable the optional flags:

```python
from ai_tracking import run_yolo_on_video

for result in run_yolo_on_video("/path/to/video.mp4", save_annotated=True):
    pass  # the annotated clip is saved beside the original as `/path/to/video_annotated.mp4`
```

You can customize the output suffix or path:

```python
for result in run_yolo_on_video(
    "/path/to/video.mp4",
    save_annotated=True,
    annotated_suffix="_processed",
    annotated_path="/custom/output.mp4"
):
    pass
```

### Extracting person crops with HSV histograms

Enable crop extraction to save person bounding boxes as individual images with metadata including HSV color histograms:

```python
from ai_tracking import run_yolo_on_video

for result in run_yolo_on_video(
    "/path/to/video.mp4",
    output_dir="./data/output/my_run",
    save_crops=True,
    frames=[24, 48, 72]  # Optional: process specific frames only
):
    pass
```

This creates the following structure:

```
data/output/my_run/
 crops_metadata.json          # Complete metadata for all crops
 annotated-video.mp4           # Optional annotated video
 frame_0024/
    frame_0024.jpg            # Full frame image
    crop_0001.jpg             # Person crop 1
    crop_0001_hist.png        # HSV histogram visualization (debug mode only)
    crop_0002.jpg
    crop_0002_hist.png
    ...
 frame_0048/
 frame_0072/
```

The `crops_metadata.json` includes:

```json
{
  "frames": {
    "frame_0024": {
      "frame_idx": 24,
      "crops": [
        {
          "bbox": [x1, y1, x2, y2],
          "class_name": "person",
          "confidence": 0.90,
          "filename": "crop_0001.jpg",
          "hsv_hist": [0.0, 0.0, ..., 0.05]  // 128-dim normalized feature vector
        }
      ]
    }
  }
}
```

**HSV Histogram Features:**
- Computed from the **upper 50% of each crop** (shirt region)
- Default bins: **8 Hue  4 Saturation  4 Value = 128 dimensions**
- Normalized for comparability across different crop sizes
- Ideal for clustering similar players/objects

**Histogram Visualizations:**
- Automatically enabled in **debug mode** (when `frames` parameter is specified)
- Automatically disabled in **production mode** (when `frames=None`)
- Override with `visualize_histograms=True/False` parameter

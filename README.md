# Soccer Tracking AI

Minimal scaffolding to run a pretrained YOLO model on every frame of a video, detecting only people by default.

## Installation

```bash
uv pip install -e .
```

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

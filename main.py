from ai_tracking import run_yolo_on_video


def main():
    for idx, result in enumerate(run_yolo_on_video(r"C:\\temp\\raw-video.mp4", save_annotated=True)):
        print(f"Frame {idx}: {result}")

if __name__ == "__main__":
    main()

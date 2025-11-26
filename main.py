from ai_tracking import run_yolo_on_video, run_yolo_tracking_mode


def main():
    # Define paths
    input_video = r"./data/input/raw-video.mp4"
    annotated_output = r"./data/output/annotated-video.mp4"
    tracked_output = r"./data/output/tracked-video.mp4"
    
    # Option 1: Frame-by-frame inference with annotations
    for idx, result in enumerate(run_yolo_on_video(
        input_path=input_video,
        output_path=annotated_output,
        save_annotated=True
    )):
        print(f"Frame {idx}: {result}")
    print(f"Annotated video saved to {annotated_output}")
    
    # Option 2: Tracking mode with persistent IDs
    # run_yolo_tracking_mode(
    #     input_path=input_video,
    #     output_path=tracked_output
    # )
    # print(f"Tracking complete! Video saved to {tracked_output}")

if __name__ == "__main__":
    main()

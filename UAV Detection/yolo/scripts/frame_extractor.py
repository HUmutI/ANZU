import os
import cv2
import shutil
from tqdm import tqdm

def extract_frames_from_videos(video_folder, output_folder, frame_interval):
    # Ensure output folder exists, if not create it
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    # List all video files in the folder
    video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.webm'))]

    # Iterate through all video files with tqdm for progress tracking
    for file_name in tqdm(video_files, desc="Processing videos", unit="video"):
        video_path = os.path.join(video_folder, file_name)
        video_name = os.path.splitext(file_name)[0]

        # Open the video file
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        saved_frame_count = 0

        # Get total frame count for progress tracking
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        with tqdm(total=total_frames, desc=f"Extracting frames from {file_name}", unit="frame") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()

                if not ret:
                    break

                if frame_count % int(fps * frame_interval) == 0:
                    
                    frame_file_name = f"{video_name}_frame_{saved_frame_count:05d}.jpg"
                    frame_file_path = os.path.join(output_folder, frame_file_name)
                    cv2.imwrite(frame_file_path, frame)
                    saved_frame_count += 1

                frame_count += 1
                pbar.update(1)

            cap.release()

    print(f"Frames extracted and saved to: {output_folder}")

# Example usage
video_folder = "../data/videos"
output_folder = "../data/raw_frames"
frame_interval = 2  # Interval in seconds

extract_frames_from_videos(video_folder, output_folder, frame_interval)

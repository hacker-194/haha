import cv2
import os
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Base directory where your shared folder contents are located
# Change 'Celeb-DF' to the actual folder name if different
base_dir = '/content/drive/My Drive/Celeb-DF'

# Input video folders
celeb_real_dir = os.path.join(base_dir, 'Celeb-real')
celeb_synthesis_dir = os.path.join(base_dir, 'Celeb-synthesis')
youtube_real_dir = os.path.join(base_dir, 'YouTube-real')

# Output folders for extracted frames
celeb_real_output_dir = os.path.join(base_dir, 'celeb-fake-output')
celeb_synthesis_output_dir = os.path.join(base_dir, 'celeb-synthesis-output')
youtube_real_output_dir = os.path.join(base_dir, 'yt-output')

# Create output directories if they don't exist
os.makedirs(celeb_real_output_dir, exist_ok=True)
os.makedirs(celeb_synthesis_output_dir, exist_ok=True)
os.makedirs(youtube_real_output_dir, exist_ok=True)

def extract_frames_from_videos(video_dir, output_dir, label):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    for video_name in os.listdir(video_dir):
        video_path = os.path.join(video_dir, video_name)
        if not os.path.isfile(video_path):
            continue

        video_capture = cv2.VideoCapture(video_path)
        count = 0
        extracted = 0
        while video_capture.isOpened():
            success, frame = video_capture.read()
            if not success:
                break

            frame_filename = f"{os.path.splitext(video_name)[0]}_{label}_frame{count}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)

            # Skip if this frame already exists
            if os.path.exists(frame_path):
                count += 1
                continue

            cv2.imwrite(frame_path, frame)
            count += 1
            extracted += 1

        video_capture.release()
        print(f"Extracted {extracted} new frames from {video_name} (skipped existing).")

# Extract frames from videos in all folders
extract_frames_from_videos(celeb_real_dir, celeb_real_output_dir, 'celeb_real')
extract_frames_from_videos(celeb_synthesis_dir, celeb_synthesis_output_dir, 'celeb_synthesis')
extract_frames_from_videos(youtube_real_dir, youtube_real_output_dir, 'youtube_real')

"""
Frame Extraction Pipeline
------------------------
A robust, production-ready script for extracting frames from videos.

Features:
    - Batch extraction from all videos in a directory
    - Skips already-extracted videos or resumes partial extraction
    - Configurable extraction interval (every_n_frames)
    - Robust logging and error handling
    - CLI with argparse
    - Summary report at the end

Usage:
    python frame_extraction.py --manipulated_dir ./manipulated --original_dir ./original --output_manipulated ./frames_manip --output_original ./frames_orig --every_n_frames 10
"""

import os
import cv2
import logging
from typing import List, Optional
from tqdm import tqdm
import argparse
import sys

def setup_logging(level: int = logging.INFO, log_file: str = None):
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s", handlers=handlers)

def list_video_files(directory: str, extensions: Optional[List[str]] = None) -> List[str]:
    if extensions is None:
        extensions = ['.mp4', '.avi', '.mov', '.mkv']
    return [
        f for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f)) and os.path.splitext(f)[1].lower() in extensions
    ]

def check_partial_extraction(output_dir: str, frame_prefix: str, label: str, expected_frames: int, img_ext: str) -> bool:
    images = [
        f for f in os.listdir(output_dir)
        if f.startswith(f"{frame_prefix}_{label}_frame") and f.endswith(img_ext)
    ]
    return len(images) >= expected_frames

def extract_frames_from_videos(
    video_dir: str,
    output_dir: str,
    label: str,
    every_n_frames: int = 1,
    overwrite: bool = False,
    extensions: Optional[List[str]] = None,
    img_ext: str = ".jpg"
):
    logger = logging.getLogger()
    if not os.path.exists(video_dir):
        logger.error(f"Video directory does not exist: {video_dir}")
        return 0, 0
    os.makedirs(output_dir, exist_ok=True)
    video_files = list_video_files(video_dir, extensions)
    if not video_files:
        logger.warning(f"No video files found in {video_dir}.")
        return 0, 0

    total_videos = len(video_files)
    total_frames_saved = 0

    for idx, video_name in enumerate(video_files, 1):
        video_path = os.path.join(video_dir, video_name)
        frame_prefix = os.path.splitext(video_name)[0]
        cap = cv2.VideoCapture(video_path)
        try:
            if not cap.isOpened():
                logger.warning(f"Failed to open video: {video_path}")
                continue

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                logger.warning(f"Zero frames in video: {video_path}")
                continue
            expected_frames = (total_frames + every_n_frames - 1) // every_n_frames
            first_frame_path = os.path.join(output_dir, f"{frame_prefix}_{label}_frame0{img_ext}")

            already_exists = os.path.exists(first_frame_path)
            partial = False
            if already_exists and not overwrite:
                partial = not check_partial_extraction(output_dir, frame_prefix, label, expected_frames, img_ext)
                if not partial:
                    logger.info(f"[{idx}/{len(video_files)}] Frames for '{video_name}' already exist. Skipping.")
                    continue
                else:
                    logger.warning(f"[{idx}/{len(video_files)}] Partial extraction detected for '{video_name}'. Will resume extraction.")

            count, saved = 0, 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            pbar = tqdm(total=total_frames, desc=f"Extracting {video_name}", unit="frame", leave=False)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if count % every_n_frames == 0:
                    frame_path = os.path.join(output_dir, f"{frame_prefix}_{label}_frame{count}{img_ext}")
                    if not overwrite and os.path.exists(frame_path):
                        saved += 1
                        count += 1
                        pbar.update(1)
                        continue
                    try:
                        success = cv2.imwrite(frame_path, frame)
                        if not success:
                            logger.error(f"Failed to write frame {count} of '{video_name}' to {frame_path}")
                        else:
                            saved += 1
                    except Exception as e:
                        logger.error(f"Exception saving frame {count} of '{video_name}': {e}")
                count += 1
                pbar.update(1)
            pbar.close()
            logger.info(f"[{idx}/{len(video_files)}] '{video_name}': {saved} frames saved (Total: {total_frames})")
            total_frames_saved += saved
        except Exception as e:
            logger.error(f"Exception processing video {video_path}: {e}")
        finally:
            cap.release()
    return total_videos, total_frames_saved

def print_summary(summary_dict):
    logger = logging.getLogger()
    logger.info("----- Extraction Summary -----")
    for key, value in summary_dict.items():
        logger.info(f"{key}: {value} files / {value[1]} frames extracted")

def main():
    parser = argparse.ArgumentParser(description="Extract frames from video files in a directory.")
    parser.add_argument("--manipulated_dir", required=True, help="Directory containing manipulated videos")
    parser.add_argument("--original_dir", required=True, help="Directory containing original videos")
    parser.add_argument("--output_manipulated", required=True, help="Output directory for manipulated frames")
    parser.add_argument("--output_original", required=True, help="Output directory for original frames")
    parser.add_argument("--every_n_frames", type=int, default=1, help="Extract every n-th frame")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing frames")
    parser.add_argument("--img_ext", default=".jpg", help="Image extension for saved frames")
    parser.add_argument("--log_file", default=None, help="Optional log file")
    args = parser.parse_args()

    setup_logging(log_file=args.log_file)
    logger = logging.getLogger()

    logger.info(f"Starting frame extraction with every_n_frames={args.every_n_frames}")

    summary = {}

    v1, f1 = extract_frames_from_videos(
        args.manipulated_dir, args.output_manipulated, label="manipulated",
        every_n_frames=args.every_n_frames, overwrite=args.overwrite, img_ext=args.img_ext
    )
    summary["manipulated"] = (v1, f1)
    v2, f2 = extract_frames_from_videos(
        args.original_dir, args.output_original, label="original",
        every_n_frames=args.every_n_frames, overwrite=args.overwrite, img_ext=args.img_ext
    )
    summary["original"] = (v2, f2)

    print_summary(summary)

if __name__ == "__main__":
    main()

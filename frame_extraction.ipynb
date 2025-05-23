import cv2
import os
import logging
from typing import List, Optional
from tqdm import tqdm

def setup_logging(level: int = logging.INFO):
    """Set up console logging."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s"
    )


def list_video_files(directory: str, extensions: Optional[List[str]] = None) -> List[str]:
    """Return a list of video files in a directory (filtered by extensions)."""
    if extensions is None:
        extensions = ['.mp4', '.avi', '.mov', '.mkv']
    return [
        f for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f))
        and os.path.splitext(f)[1].lower() in extensions
    ]


def check_partial_extraction(output_dir: str, frame_prefix: str, label: str, expected_frames: int, img_ext: str) -> bool:
    """Check if the number of extracted frames matches the expected count."""
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
    """
    Extract frames from videos in a directory and save them as images.

    Args:
        video_dir: Directory with video files.
        output_dir: Directory to save extracted frames.
        label: Label to append in filenames.
        every_n_frames: Extract every Nth frame.
        overwrite: If True, overwrite existing frames.
        extensions: List of allowed video file extensions.
        img_ext: Image extension for saved frames (e.g. '.jpg', '.png').
    """
    if not os.path.exists(video_dir):
        logging.error(f"Video directory does not exist: {video_dir}")
        return
    os.makedirs(output_dir, exist_ok=True)
    video_files = list_video_files(video_dir, extensions)

    if not video_files:
        logging.warning(f"No video files found in {video_dir}.")
        return

    for idx, video_name in enumerate(video_files, 1):
        video_path = os.path.join(video_dir, video_name)
        frame_prefix = os.path.splitext(video_name)[0]
        cap = cv2.VideoCapture(video_path)
        try:
            if not cap.isOpened():
                logging.warning(f"Failed to open video: {video_path}")
                continue

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            expected_frames = (total_frames + every_n_frames - 1) // every_n_frames
            first_frame_path = os.path.join(output_dir, f"{frame_prefix}_{label}_frame0{img_ext}")

            already_exists = os.path.exists(first_frame_path)
            partial = False
            if already_exists and not overwrite:
                partial = not check_partial_extraction(output_dir, frame_prefix, label, expected_frames, img_ext)
                if not partial:
                    logging.info(f"[{idx}/{len(video_files)}] Frames for '{video_name}' already exist. Skipping.")
                    continue
                else:
                    logging.warning(f"[{idx}/{len(video_files)}] Partial extraction detected for '{video_name}'. Will resume extraction.")

            count, saved = 0, 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            pbar = tqdm(total=total_frames, desc=f"Extracting {video_name}", unit="frame", leave=False)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if count % every_n_frames == 0:
                    frame_path = os.path.join(output_dir, f"{frame_prefix}_{label}_frame{count}{img_ext}")
                    try:
                        success = cv2.imwrite(frame_path, frame)
                        if not success:
                            logging.error(f"Failed to write frame {count} of '{video_name}' to {frame_path}")
                        else:
                            saved += 1
                    except Exception as e:
                        logging.error(f"Exception saving frame {count} of '{video_name}': {e}")
                count += 1
                pbar.update(1)
            pbar.close()
            logging.info(
                f"[{idx}/{len(video_files)}] '{video_name}': {saved} frames saved (Total: {total_frames}, FPS: {fps:.2f})"
            )
        finally:
            cap.release()


def print_summary(*dirs):
    """Print a summary of file counts for each directory."""
    for directory in dirs:
        if os.path.exists(directory):
            count = sum(1 for _ in os.scandir(directory))
            logging.info(f"{directory}: {count} items")
        else:
            logging.info(f"{directory}: Directory does not exist")


# --- USAGE EXAMPLE ---

if __name__ == "__main__":
    setup_logging()

    # Set your directories as needed
    manipulated_dir = r"D:\DFD_manipulated_sequences\DFD_manipulated_sequences"
    original_dir = r"D:\DFD_original_sequences"
    output_manipulated = r"D:\DFD_extracted_frames\manipulated"
    output_original = r"D:\DFD_extracted_frames\original"
    every_n_frames = 1
    overwrite = False
    img_ext = ".jpg"  # or ".png"

    extract_frames_from_videos(
        manipulated_dir, output_manipulated, label="manipulated",
        every_n_frames=every_n_frames, overwrite=overwrite, img_ext=img_ext
    )
    extract_frames_from_videos(
        original_dir, output_original, label="original",
        every_n_frames=every_n_frames, overwrite=overwrite, img_ext=img_ext
    )

    print_summary(manipulated_dir, original_dir, output_manipulated, output_original)

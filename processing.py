from utils import setup_logging, get_detector, detect_and_crop_face
import os
import cv2
import logging
from tqdm import tqdm

DATA_PATH = os.environ.get("DATA_PATH", r"D:\DFD_extracted_frames")
FRAMES_DIR = os.path.join(DATA_PATH, "frames")
PROCESSED_DIR = os.path.join(DATA_PATH, "processed_frames")
LABELS = ["real", "fake"]
IMG_SIZE = 128

def preprocess_directory(src_dir, dst_dir, batch_size=1000, detector=None):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir, exist_ok=True)
    files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]
    total_batches = (len(files) + batch_size - 1) // batch_size
    for i in range(0, len(files), batch_size):
        batch = files[i:i+batch_size]
        for fname in tqdm(batch, desc=f"Preprocessing {src_dir}", leave=False):
            src_path = os.path.join(src_dir, fname)
            dst_path = os.path.join(dst_dir, fname)
            try:
                img = cv2.imread(src_path)
                if img is None:
                    logging.warning(f"Failed to read image: {src_path}")
                    continue
                cropped, _ = detect_and_crop_face(img, detector)
                cropped = cv2.resize(cropped, (IMG_SIZE, IMG_SIZE))
                cv2.imwrite(dst_path, cropped)
            except Exception as e:
                logging.warning(f"Error processing {src_path}: {e}")
        logging.info(f"Processed batch {i // batch_size + 1}/{total_batches} in {src_dir}")

if __name__ == "__main__":
    setup_logging()
    detector = get_detector()
    for label in LABELS:
        src_dir = os.path.join(FRAMES_DIR, label)
        dst_dir = os.path.join(PROCESSED_DIR, label)
        if not os.path.exists(src_dir):
            logging.warning(f"Source directory does not exist: {src_dir}")
            continue
        preprocess_directory(src_dir, dst_dir, batch_size=1000, detector=detector)

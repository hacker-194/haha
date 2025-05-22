import os
import cv2
import numpy as np
from mtcnn import MTCNN
import logging
from tqdm import tqdm

# === CONFIGURATION ===
DATA_PATH = r"D:\DFD_extracted_frames"
FRAMES_DIR = os.path.join(DATA_PATH, "frames")
PROCESSED_DIR = os.path.join(DATA_PATH, "processed_frames")
LABELS = ["real", "fake"]
IMG_SIZE = 128

def setup_logging(level=logging.INFO):
    """Set up console logging."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s: %(message)s"
    )

def get_detector():
    """Create an MTCNN detector instance. This prevents GPU memory leak if run repeatedly in notebooks."""
    return MTCNN()

def detect_and_crop_face(image_array, detector):
    """Detect and crop the largest face in the image. If no face, return the original image."""
    image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(image_rgb)
    if faces:
        # Use the most confident, largest face
        faces = sorted(faces, key=lambda f: (f['confidence'], f['box'][2]*f['box'][3]), reverse=True)
        x, y, w, h = faces[0]['box']
        x, y = max(0, x), max(0, y)
        x2, y2 = min(x + w, image_rgb.shape[1]), min(y + h, image_rgb.shape[0])
        cropped = image_rgb[y:y2, x:x2]
        return cropped
    return image_rgb

def preprocess_image(image_array, detector):
    """Detect, crop, resize, and normalize an image array."""
    cropped = detect_and_crop_face(image_array, detector)
    resized = cv2.resize(cropped, (IMG_SIZE, IMG_SIZE))
    normalized = resized / 255.0
    return normalized

def preprocess_directory(directory_path, save_to_dir, batch_size=64, detector=None):
    """Preprocess images in a directory and save them to another."""
    os.makedirs(save_to_dir, exist_ok=True)
    files = sorted([f for f in os.listdir(directory_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    for i in range(0, len(files), batch_size):
        batch_files = files[i:i+batch_size]
        for fname in tqdm(batch_files, desc=f"{os.path.basename(directory_path)} batch {i//batch_size+1}", leave=False):
            src = os.path.join(directory_path, fname)
            dst = os.path.join(save_to_dir, fname)
            if os.path.exists(dst):
                continue
            try:
                img = cv2.imread(src)
                if img is not None:
                    proc = preprocess_image(img, detector)
                    out_img = (proc * 255).astype(np.uint8)
                    success = cv2.imwrite(dst, cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
                    if not success:
                        logging.error(f"Failed to save {dst}")
                else:
                    logging.warning(f"Could not read image: {src}")
            except Exception as e:
                logging.error(f"Exception processing {src}: {e}")
        logging.info(f"Processed batch {i // batch_size + 1}/{(len(files) + batch_size - 1) // batch_size} in {directory_path}")

if __name__ == "__main__":
    setup_logging()
    detector = get_detector()
    for label in LABELS:
        src_dir = os.path.join(FRAMES_DIR, label)
        dst_dir = os.path.join(PROCESSED_DIR, label)
        if not os.path.exists(src_dir):
            logging.warning(f"Source directory does not exist: {src_dir}")
            continue
        preprocess_directory(
            src_dir,
            dst_dir,
            batch_size=1000,
            detector=detector
        )

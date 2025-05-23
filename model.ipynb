import os
import sys
import numpy as np
import time
import uuid
import hashlib
import logging
from typing import Optional, List
from datetime import datetime
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image

import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForImageClassification

# === CONFIGURATION ===
DATA_PATH = os.environ.get("DEEFAKE_DATA_PATH", r"D:\deepfake_data")
PROCESSED_DIR = os.path.join(DATA_PATH, "processed_frames")
FEEDBACK_DIR = os.path.join(DATA_PATH, "feedback")
FEEDBACK_LOG = os.path.join(DATA_PATH, "feedback_log.txt")
LOCKFILE = os.path.join(FEEDBACK_DIR, "feedback.lock")
FEEDBACK_IMAGE_FORMAT = ".jpg"

HF_MODELS = [
    {
        "name": "prithivMLmods/AI-vs-Deepfake-vs-Real-v2.0",
        "processor": None,
        "model": None,
        "label_map": {0: "AI", 1: "Deepfake", 2: "Real"}
    },
    {
        "name": "microsoft/resnet-50",
        "processor": None,
        "model": None,
        "label_map": None
    }
]

def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s: %(message)s")

def get_unique_filename(original_path: str, ext: Optional[str] = None) -> str:
    name, _ = os.path.splitext(os.path.basename(original_path))
    unique_suffix = datetime.now().strftime('%Y%m%d_%H%M%S') + "_" + str(uuid.uuid4())[:8]
    ext = ext if ext else os.path.splitext(original_path)[1]
    return f"{name}_{unique_suffix}{ext}"

def ensure_dir(path: str) -> None:
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

def hash_image(image_path: str) -> Optional[str]:
    try:
        with open(image_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception as e:
        logging.warning(f"Hashing failed for {image_path}: {e}")
        return None

def log_feedback(logfile: str, image_path: str, predicted_label: str, true_label: str, user: Optional[str] = None, img_hash: Optional[str] = None) -> None:
    try:
        with open(logfile, "a", encoding="utf-8") as f:
            line = f"{datetime.now().isoformat()}\t{image_path}\t{predicted_label}\t{true_label}\t{user or 'unknown'}\t{img_hash or ''}\n"
            f.write(line)
    except Exception as e:
        logging.warning(f"Could not log feedback: {e}")

def load_processed_hashes(feedback_log: str) -> set:
    hashes = set()
    if not os.path.exists(feedback_log):
        return hashes
    with open(feedback_log, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 6:
                hashes.add(parts[5])
    return hashes

class FileLock:
    def __init__(self, lockfile: str):
        self.lockfile = lockfile
    def acquire(self) -> None:
        while os.path.exists(self.lockfile):
            time.sleep(0.1)
        with open(self.lockfile, 'w') as f:
            f.write(str(os.getpid()))
    def release(self) -> None:
        if os.path.exists(self.lockfile):
            os.remove(self.lockfile)

def collect_and_store_feedback(
    image_path: str, 
    predicted_label: str, 
    feedback_dir: str, 
    true_label: Optional[str] = None, 
    user: Optional[str] = None, 
    logfile: str = FEEDBACK_LOG,
    img_hash: Optional[str] = None,
    no_input: bool = False,
    feedback_image_format: str = FEEDBACK_IMAGE_FORMAT
) -> str:
    lock = FileLock(LOCKFILE)
    try:
        lock.acquire()
        if true_label is None and not no_input:
            while True:
                inp = input(f"Model predicted {predicted_label}. Enter correct label for {os.path.basename(image_path)} (AI, Deepfake, Real, or class index): ").strip()
                if inp in {"AI", "Deepfake", "Real"} or inp.isdigit():
                    true_label = inp
                    break
                else:
                    print("Please enter 'AI', 'Deepfake', 'Real', or a class index (0-999).")
        elif true_label is None and no_input:
            true_label = predicted_label
        label_folder = true_label
        dest_dir = os.path.join(feedback_dir, label_folder)
        ensure_dir(dest_dir)
        unique_name = get_unique_filename(image_path, ext=feedback_image_format)
        dest_path = os.path.join(dest_dir, unique_name)
        try:
            img = load_img(image_path)
            arr = img_to_array(img).astype(np.uint8)
            Image.fromarray(arr).save(dest_path, "JPEG")
        except Exception as e:
            logging.warning(f"Could not convert and save {image_path} as .jpg: {e}")
            try:
                import shutil
                shutil.copy(image_path, dest_path)
            except Exception as e2:
                logging.warning(f"Could not copy file {image_path}: {e2}")
                dest_path = image_path
        ensure_dir(os.path.dirname(logfile))
        log_feedback(logfile, dest_path, predicted_label, true_label, user=user, img_hash=img_hash)
        return dest_path
    finally:
        lock.release()

def image_files_in_dir(folder: str) -> List[str]:
    IMG_EXTS = ('.jpg', '.jpeg', '.png')
    return [f for f in os.listdir(folder) if f.lower().endswith(IMG_EXTS)]

def hf_predict(hf, pil_image: Image.Image, device: str = "cpu") -> str:
    processor = hf["processor"]
    model = hf["model"].to(device)
    inputs = processor(images=pil_image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)
        predicted_class_idx = int(torch.argmax(probs, dim=-1).item())
        if hf["label_map"]:
            label = hf["label_map"].get(predicted_class_idx, str(predicted_class_idx))
            return label
        else:
            return str(predicted_class_idx)

def ensemble_predict(
    hf_models: List[dict],
    pil_image: Image.Image,
    device: str = "cpu"
) -> List[str]:
    preds = []
    for hf in hf_models:
        value = hf_predict(hf, pil_image, device=device)
        preds.append(value)
    return preds

def batch_predict_on_folder_ensemble(
    folder: str, 
    hf_models: List[dict],
    feedback_dir: str, 
    label_hint: Optional[str] = None, 
    batch_size: int = 32, 
    feedback_log: str = FEEDBACK_LOG, 
    user: Optional[str] = None,
    no_input: bool = False,
    feedback_image_format: str = FEEDBACK_IMAGE_FORMAT,
    device: str = "cpu"
) -> None:
    if not os.path.exists(folder):
        logging.error(f"Directory does not exist: {folder}")
        return

    processed_hashes = load_processed_hashes(feedback_log)
    image_files = image_files_in_dir(folder)
    image_info = []
    for filename in image_files:
        image_path = os.path.join(folder, filename)
        img_hash = hash_image(image_path)
        if img_hash is None or img_hash in processed_hashes:
            continue
        image_info.append((image_path, img_hash))

    for i in range(0, len(image_info), batch_size):
        batch_info = image_info[i:i+batch_size]
        pil_images = []
        paths, hashes = [], []
        for image_path, img_hash in batch_info:
            try:
                pil_img = load_img(image_path, target_size=(224, 224))
                pil_images.append(pil_img)
                paths.append(image_path)
                hashes.append(img_hash)
            except Exception as e:
                logging.warning(f"Could not load image {image_path}: {e}")
        if pil_images:
            for pil_img, path, img_hash in zip(pil_images, paths, hashes):
                preds = ensemble_predict(hf_models, pil_img, device=device)
                pred_info = ", ".join(preds)
                logging.info(f"Predictions for {os.path.basename(path)}: {pred_info}")
                for pred in preds:
                    collect_and_store_feedback(
                        path, pred, feedback_dir,
                        true_label=label_hint,
                        user=user,
                        logfile=feedback_log,
                        img_hash=img_hash,
                        no_input=no_input,
                        feedback_image_format=feedback_image_format
                    )

def load_hf_models(hf_model_configs: List[dict], device: str = "cpu") -> List[dict]:
    loaded = []
    for hf in hf_model_configs:
        try:
            processor = AutoImageProcessor.from_pretrained(hf["name"])
            model = AutoModelForImageClassification.from_pretrained(hf["name"]).to(device)
            hf["processor"] = processor
            hf["model"] = model
            loaded.append(hf)
            logging.info(f"Loaded HuggingFace model: {hf['name']}")
        except Exception as e:
            logging.warning(f"Failed to load HuggingFace model {hf['name']}: {e}")
    return loaded

def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    setup_logging()
    device = get_device()
    hf_models = load_hf_models(HF_MODELS, device=device)
    if not hf_models:
        logging.critical("No HuggingFace models were loaded. Exiting.")
        sys.exit(1)

    user = os.environ.get("USER", "piphahaha")
    batch_size = 32
    no_input = True

    for label_hint in ["AI", "Deepfake", "Real"]:
        folder = os.path.join(PROCESSED_DIR, label_hint)
        batch_predict_on_folder_ensemble(
            folder, hf_models, FEEDBACK_DIR,
            label_hint=label_hint, batch_size=batch_size, feedback_log=FEEDBACK_LOG,
            user=user, no_input=no_input, device=device
        )
    logging.info("All batches completed.")

import time
import logging
from pathlib import Path
import shutil
import os
import tensorflow as tf
from tensorflow.keras.models import load_model, save_model

# === CONFIGURATION ===
DATA_PATH = r"D:\deepfake_data"
FEEDBACK_LOG = Path(os.path.join(DATA_PATH, "feedback_log.txt"))
FEEDBACK_DIR = Path(os.path.join(DATA_PATH, "feedback"))
PROCESSED_HASHES_FILE = Path(os.path.join(DATA_PATH, "processed_hashes.txt"))
MODEL_PATH = Path(os.path.join(DATA_PATH, "current_model.h5"))
RETRAIN_THRESHOLD = 50
CHECK_INTERVAL = 3600

def load_processed_hashes(feedback_log):
    hashes = set()
    if not feedback_log.exists():
        return hashes
    with open(feedback_log, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 6:
                hashes.add(parts[5])
    return hashes

def get_num_new_feedback():
    hashes = load_processed_hashes(FEEDBACK_LOG)
    if PROCESSED_HASHES_FILE.exists():
        with open(PROCESSED_HASHES_FILE, "r") as f:
            old_hashes = set(f.read().splitlines())
    else:
        old_hashes = set()
    new_hashes = hashes - old_hashes
    return len(new_hashes), new_hashes

def update_processed_hashes(hashes):
    with open(PROCESSED_HASHES_FILE, "a") as f:
        for h in hashes:
            f.write(h + "\n")

def backup_and_update_model(new_model: tf.keras.Model, model_path: Path):
    backup_path = model_path.with_name(f"backup_{int(time.time())}.h5")
    if model_path.exists():
        shutil.copy(str(model_path), str(backup_path))
        logging.info(f"Backup of old model saved to {backup_path}")
    save_model(new_model, model_path)
    logging.info(f"Updated model saved to {model_path}")

def retrain_model_with_feedback(feedback_dir: Path, base_model_path: Path) -> tf.keras.Model:
    print(f"Pretending to retrain model with new data in {feedback_dir}...")
    model = load_model(base_model_path)
    # Add your actual retraining code here!
    return model

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    while True:
        num_new, new_hashes = get_num_new_feedback()
        logging.info(f"Found {num_new} new feedback samples.")
        if num_new >= RETRAIN_THRESHOLD:
            logging.info("Retraining model with new feedback...")
            try:
                new_model = retrain_model_with_feedback(FEEDBACK_DIR, MODEL_PATH)
                backup_and_update_model(new_model, MODEL_PATH)
                update_processed_hashes(new_hashes)
                logging.info("Retraining complete and model updated.")
            except Exception as e:
                logging.error(f"Retraining failed: {e}", exc_info=True)
        else:
            logging.info(f"Waiting for more feedback before retraining. ({num_new}/{RETRAIN_THRESHOLD})")
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()

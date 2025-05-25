"""
Self-Learning Model Retrainer for Deepfake Detection
---------------------------------------------------
Automatically retrains the current model when enough new user feedback is collected.

Features:
    - Monitors feedback directory for new labeled samples (hash-based deduplication)
    - Retrains model when RETRAIN_THRESHOLD is reached
    - Backs up old models before overwriting
    - Logs all major events and errors
    - Configurable paths, batch size, and intervals

Usage:
    python selflearn.py

Tips:
    - Set DATA_PATH via environment variable DEEFAKE_DATA_PATH or edit below.
    - Tune RETRAIN_THRESHOLD and EPOCHS as needed for your deployment.
"""

import time
import logging
from pathlib import Path
import shutil
import os
import tensorflow as tf
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATA_PATH = os.environ.get("DEEFAKE_DATA_PATH", r"D:\deepfake_data")
FEEDBACK_LOG = Path(os.path.join(DATA_PATH, "feedback_log.txt"))
FEEDBACK_DIR = Path(os.path.join(DATA_PATH, "feedback"))
PROCESSED_HASHES_FILE = Path(os.path.join(DATA_PATH, "processed_hashes.txt"))
MODEL_PATH = Path(os.path.join(DATA_PATH, "current_model.h5"))
RETRAIN_THRESHOLD = 50
CHECK_INTERVAL = 3600
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 2  # Increase this for real retraining

def load_processed_hashes(feedback_log):
    """Load image hashes from feedback log file."""
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
    """Return number and set of new image hashes not already used for retraining."""
    hashes = load_processed_hashes(FEEDBACK_LOG)
    if PROCESSED_HASHES_FILE.exists():
        with open(PROCESSED_HASHES_FILE, "r") as f:
            old_hashes = set(f.read().splitlines())
    else:
        old_hashes = set()
    new_hashes = hashes - old_hashes
    return len(new_hashes), new_hashes

def update_processed_hashes(hashes):
    """Append newly used hashes to the processed hashes file."""
    with open(PROCESSED_HASHES_FILE, "a") as f:
        for h in hashes:
            f.write(h + "\n")

def backup_and_update_model(new_model: tf.keras.Model, model_path: Path):
    """Backup the current model and overwrite with the new one."""
    backup_path = model_path.with_name(f"backup_{int(time.time())}.h5")
    if model_path.exists():
        shutil.copy(str(model_path), str(backup_path))
        logging.info(f"Backup of old model saved to {backup_path}")
    save_model(new_model, model_path)
    logging.info(f"Updated model saved to {model_path}")

def retrain_model_with_feedback(feedback_dir: Path, base_model_path: Path) -> tf.keras.Model:
    """Retrain model using feedback samples and return updated model."""
    model = load_model(base_model_path)
    datagen = ImageDataGenerator(
        rescale=1.0/255,
        validation_split=0.2
    )
    train_gen = datagen.flow_from_directory(
        feedback_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    val_gen = datagen.flow_from_directory(
        feedback_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=True
    )

    # Infer loss based on model output
    loss = 'categorical_crossentropy'
    if model.output_shape[-1] == 1:
        loss = 'binary_crossentropy'

    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        verbose=1
    )
    return model

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.info("Self-learning retrainer started. Monitoring feedback for new samples.")
    while True:
        num_new, new_hashes = get_num_new_feedback()
        logging.info(f"Found {num_new} new feedback samples.")
        if num_new >= RETRAIN_THRESHOLD:
            logging.info("Retraining model with new feedback...")
            try:
                model = retrain_model_with_feedback(FEEDBACK_DIR, MODEL_PATH)
                backup_and_update_model(model, MODEL_PATH)
                update_processed_hashes(new_hashes)
                logging.info("Retraining and update complete.")
            except Exception as e:
                logging.error(f"Retraining failed: {e}")
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()

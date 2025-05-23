import os
import sys
import argparse
import logging
import random
from typing import List, Optional

import numpy as np
import cv2
from tensorflow.keras.models import load_model

try:
    from preprocessing_pipeline import preprocess_image as custom_preprocess_image
except ImportError:
    custom_preprocess_image = None

def setup_logging():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO
    )

def sample_images_from_directory(directory: str, num_samples: int, seed: Optional[int] = None) -> List[np.ndarray]:
    """Randomly samples images from a directory and loads them as numpy arrays."""
    if seed is not None:
        random.seed(seed)
    if not os.path.isdir(directory):
        logging.error(f"Directory does not exist: {directory}")
        return []
    all_images = [
        os.path.join(directory, img_name)
        for img_name in sorted(os.listdir(directory))
        if os.path.isfile(os.path.join(directory, img_name))
    ]
    if len(all_images) < num_samples:
        logging.warning(f"Not enough images in {directory} to sample {num_samples} images.")
        return []
    sampled_paths = random.sample(all_images, num_samples)
    images = []
    for img_path in sampled_paths:
        image = cv2.imread(img_path)
        if image is not None:
            images.append(image)
        else:
            logging.warning(f"Failed to load image: {img_path}")
    return images

def preprocess_image_lstm(image: np.ndarray) -> np.ndarray:
    image = cv2.resize(image, (128, 128))
    return image / 255.0

def preprocess_image_efficientnet(image: np.ndarray) -> np.ndarray:
    from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = effnet_preprocess(image.astype(np.float32))
    return image

def preprocess_image_default(image: np.ndarray) -> np.ndarray:
    image = cv2.resize(image, (224, 224))
    return image / 255.0

def get_preprocessor(model_type: str):
    if model_type == "lstm":
        return custom_preprocess_image if custom_preprocess_image else preprocess_image_lstm, (128, 128)
    elif model_type == "efficientnet":
        return preprocess_image_efficientnet, (224, 224)
    else:
        return preprocess_image_default, (224, 224)

def predict_deepfake(sequence_of_images: List[np.ndarray], model, model_type: str) -> Optional[float]:
    preprocessor, _ = get_preprocessor(model_type)
    try:
        preprocessed_images = []
        for img in sequence_of_images:
            processed = preprocessor(img)
            if processed is None:
                logging.warning("A preprocessed image is None. Skipping prediction.")
                return None
            preprocessed_images.append(processed)
        preprocessed_images = np.array(preprocessed_images)
        if model_type == "lstm":
            preprocessed_sequence = np.expand_dims(preprocessed_images, axis=0)
        elif model_type == "efficientnet":
            preprocessed_sequence = preprocessed_images
        else:
            preprocessed_sequence = np.expand_dims(preprocessed_images, axis=0)
        prediction = model.predict(preprocessed_sequence)
        if prediction.shape[-1] == 1:
            return float(prediction[0][0])
        else:
            return float(prediction[0][1])  # assume index 1 is "fake"
    except Exception as e:
        logging.error(f"Prediction error: {e}", exc_info=True)
        return None

def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Deepfake Sequence Predictor (multi-model support)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to Keras model (.h5 file)")
    parser.add_argument("--model_type", type=str, required=True, choices=["lstm", "efficientnet", "other"],
                        help="Model type: 'lstm', 'efficientnet', or 'other'")
    parser.add_argument("--real_dir", type=str, required=True, help="Directory with real images")
    parser.add_argument("--fake_dir", type=str, required=True, help="Directory with synthetic images")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples per set")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    try:
        model = load_model(args.model_path)
        logging.info(f"Model loaded from {args.model_path}")
    except Exception as e:
        logging.error(f"Could not load model: {e}")
        sys.exit(1)

    real_images = sample_images_from_directory(args.real_dir, args.num_samples, seed=args.seed)
    fake_images = sample_images_from_directory(args.fake_dir, args.num_samples, seed=args.seed)
    combined_images = real_images + fake_images

    if combined_images:
        deepfake_prob = predict_deepfake(combined_images, model, args.model_type)
        if deepfake_prob is not None:
            print(f"Combined Images | Deepfake Probability: {deepfake_prob:.4f}")
        else:
            print("Failed to process combined images.")
    else:
        print("No valid images to process.")

if __name__ == "__main__":
    main()

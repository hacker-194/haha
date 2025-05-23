import os
import random
import logging
from typing import List, Optional

import numpy as np
import cv2
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Try to import a custom preprocessor if available (for LSTM support)
try:
    from preprocessing_pipeline import preprocess_image as custom_preprocess_image
except ImportError:
    custom_preprocess_image = None

def setup_logging(level=logging.INFO):
    """Set up notebook-friendly logging."""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=level
    )

def sample_images_from_directory(directory: str, num_samples: int, seed: Optional[int] = None, show_samples: bool = True) -> List[np.ndarray]:
    """Randomly sample images from a directory and load them as numpy arrays. Optionally display samples."""
    if seed is not None:
        random.seed(seed)
    if not os.path.isdir(directory):
        logging.error(f"Directory does not exist: {directory}")
        return []
    all_images = [
        os.path.join(directory, img_name)
        for img_name in sorted(os.listdir(directory))
        if os.path.isfile(os.path.join(directory, img_name)) and img_name.lower().endswith(('.jpg', '.jpeg', '.png'))
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
            if show_samples:
                plt.figure()
                plt.axis('off')
                plt.title(os.path.basename(img_path))
                plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                plt.show()
        else:
            logging.warning(f"Failed to load image: {img_path}")
    return images

def preprocess_image_lstm(image: np.ndarray) -> np.ndarray:
    # Match processing.py: resize to 128x128, normalize to [0, 1]
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
        # Use custom pipeline if available
        return custom_preprocess_image if custom_preprocess_image else preprocess_image_lstm, (128, 128)
    elif model_type == "efficientnet":
        return preprocess_image_efficientnet, (224, 224)
    else:
        return preprocess_image_default, (224, 224)

def predict_deepfake(sequence_of_images: List[np.ndarray], model, model_type: str, show_shape: bool = True) -> Optional[float]:
    """
    Preprocess the sequence and predict deepfake probability.
    """
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
        if show_shape:
            print(f"Input shape to model: {preprocessed_sequence.shape}")
        prediction = model.predict(preprocessed_sequence)
        print(f"Raw model prediction: {prediction}")
        if prediction.shape[-1] == 1:
            return float(prediction[0][0])
        else:
            # Assume index 1 is "fake" as in your other scripts
            return float(prediction[0][1])
    except Exception as e:
        logging.error(f"Prediction error: {e}", exc_info=True)
        return None

def predict_on_directories(
    model_path: str,
    model_type: str,
    real_dir: str,
    fake_dir: str,
    num_samples: int = 5,
    seed: Optional[int] = None,
    show_samples: bool = True
):
    """Main notebook-friendly prediction function."""
    setup_logging()
    try:
        model = load_model(model_path)
        logging.info(f"Model loaded from {model_path}")
    except Exception as e:
        logging.error(f"Could not load model: {e}")
        return None

    real_images = sample_images_from_directory(real_dir, num_samples, seed=seed, show_samples=show_samples)
    fake_images = sample_images_from_directory(fake_dir, num_samples, seed=seed, show_samples=show_samples)
    combined_images = real_images + fake_images

    if combined_images:
        deepfake_prob = predict_deepfake(combined_images, model, model_type)
        if deepfake_prob is not None:
            print(f"\nCombined Images | Deepfake Probability: {deepfake_prob:.4f}")
            return deepfake_prob
        else:
            print("Failed to process combined images.")
            return None
    else:
        print("No valid images to process.")
        return None

# Example usage in Jupyter Notebook:
    predict_on_directories(
   model_path="path/to/model.h5",
     model_type="lstm",  # or "efficientnet" or "other"
     real_dir=r"D:\DFD_extracted_frames\frames\real",
     fake_dir=r"D:\DFD_extracted_frames\frames\fake",
     num_samples=5,
     seed=42,
     show_samples=True
 )

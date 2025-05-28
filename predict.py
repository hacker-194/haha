from utils import setup_logging, preprocess_image
import numpy as np
import cv2
import logging
from tensorflow.keras.models import load_model

def predict_image(img_path, model, model_type="lstm"):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")
    img, _ = preprocess_image(img, model_type=model_type, use_face_crop=False, output_size=224)
    img_batch = np.expand_dims(img, axis=0)
    pred = model.predict(img_batch)
    if pred.shape[-1] == 1:
        prob = float(pred[0][0])
        label = "FAKE" if prob > 0.5 else "REAL"
    else:
        prob = float(pred[0][1])
        label = "FAKE" if np.argmax(pred[0]) == 1 else "REAL"
    return {"prob": prob, "label": label}

if __name__ == "__main__":
    import argparse
    setup_logging()
    parser = argparse.ArgumentParser(description="Predict deepfake/real from an input image")
    parser.add_argument("image", type=str, help="Path to the image file")
    parser.add_argument("--model", type=str, required=True, help="Path to the model file (.h5)")
    parser.add_argument("--model_type", type=str, default="lstm", choices=["lstm", "efficientnet"], help="Model type (lstm or efficientnet)")
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        logging.error(f"Image file does not exist: {args.image}")
        exit(1)
    if not os.path.isfile(args.model):
        logging.error(f"Model file does not exist: {args.model}")
        exit(1)

    try:
        model = load_model(args.model)
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        exit(1)

    try:
        result = predict_image(args.image, model, model_type=args.model_type)
        print(f"Prediction: {result['label']}, Probability: {result['prob']:.4f}")
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        exit(1)

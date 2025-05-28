import os
import cv2
import numpy as np
import logging
from mtcnn import MTCNN

def setup_logging(log_file=None, level=logging.INFO):
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s", handlers=handlers)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def is_image_file(fname, exts=None):
    if exts is None:
        exts = [".jpg", ".jpeg", ".png"]
    return fname.lower().endswith(tuple(exts))

_detector = None
def get_detector():
    global _detector
    if _detector is None:
        _detector = MTCNN()
    return _detector

def detect_and_crop_face(image_array, detector=None):
    detector = detector or get_detector()
    if image_array.shape[-1] == 3:
        try:
            image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        except Exception:
            image_rgb = image_array
    else:
        image_rgb = image_array
    faces = detector.detect_faces(image_rgb)
    if faces:
        faces = sorted(faces, key=lambda f: f['confidence'] * f['box'][2] * f['box'][3], reverse=True)
        x, y, w, h = faces[0]['box']
        x, y = max(0, x), max(0, y)
        cropped = image_array[y:y+h, x:x+w]
        return cropped, True
    return image_array, False

def preprocess_image(img, model_type="efficientnet", use_face_crop=True, output_size=224, detector=None):
    img = img.copy()
    face_found = False
    if use_face_crop:
        img, face_found = detect_and_crop_face(img, detector)
    img = cv2.resize(img, (output_size, output_size))
    if model_type == "efficientnet":
        from tensorflow.keras.applications.efficientnet import preprocess_input
        img = preprocess_input(img.astype(np.float32))
    else:
        img = img.astype(np.float32) / 255.0
    return img, face_found

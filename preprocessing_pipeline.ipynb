import cv2
import numpy as np
from mtcnn import MTCNN

# Singleton for MTCNN detector to avoid re-initialization
_detector = None
def get_detector():
    global _detector
    if _detector is None:
        _detector = MTCNN()
    return _detector

def detect_and_crop_face(image_array: np.ndarray, detector=None) -> np.ndarray:
    """
    Detect the most confident, largest face in a BGR (or RGB) image and crop to it.
    Returns cropped face (RGB) or original image if no face detected.
    """
    detector = detector or get_detector()
    # Accept BGR or RGB input
    if image_array.shape[-1] == 3:
        try:
            # If already RGB, this does nothing harmful, otherwise converts
            image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        except Exception:
            image_rgb = image_array
    else:
        image_rgb = image_array
    faces = detector.detect_faces(image_rgb)
    if faces:
        faces = sorted(faces, key=lambda f: (f['confidence'], f['box'][2]*f['box'][3]), reverse=True)
        x, y, w, h = faces[0]['box']
        x, y = max(0, x), max(0, y)
        x2, y2 = min(x + w, image_rgb.shape[1]), min(y + h, image_rgb.shape[0])
        cropped = image_rgb[y:y2, x:x2]
        return cropped
    return image_rgb

def preprocess_image(
    image_array: np.ndarray,
    use_face_crop: bool = False,
    output_size: int = 128,
    model_type: str = "lstm"
) -> np.ndarray:
    """
    Preprocess image for model input:
    - Optionally crop to face (MTCNN)
    - Resize to output_size (default 128)
    - Normalize for model_type:
        - "efficientnet": EfficientNet preprocess_input
        - else: [0, 1] float
    Returns RGB numpy array.
    """
    img = image_array.copy()
    if use_face_crop:
        img = detect_and_crop_face(img)
    img = cv2.resize(img, (output_size, output_size))
    if model_type == "efficientnet":
        from tensorflow.keras.applications.efficientnet import preprocess_input
        img = preprocess_input(img.astype(np.float32))
    else:
        img = img.astype(np.float32) / 255.0
    return img

def get_face_confidence(image_array: np.ndarray) -> float:
    """
    Compute a heuristic confidence score for face detection [0, 1].
    Considers MTCNN face confidence and relative face area.
    """
    detector = get_detector()
    image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(image_rgb)
    if not faces:
        return 0.0
    best = max(faces, key=lambda f: f['confidence'])
    conf = best['confidence']
    area = best['box'][2] * best['box'][3]
    rel_area = area / float(image_array.shape[0] * image_array.shape[1])
    return min(1.0, conf * rel_area)

def image_quality_score(image_array: np.ndarray) -> float:
    """
    Sharpness-based quality score [0, 1] using Laplacian variance.
    """
    if image_array.ndim == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_array
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return min(1.0, score / 1000)

# Utility for consistent trust score calculation
def trust_score(prob, face_conf, quality_score, model_weight=0.5, face_weight=0.3, quality_weight=0.2):
    """
    Combines model probability, face confidence, and image quality into a trust score [0,1].
    """
    base = 1.0 - prob if prob > 0.5 else prob
    score = (
        model_weight * base +
        face_weight * face_conf +
        quality_weight * quality_score
    )
    return np.clip(score, 0, 1)

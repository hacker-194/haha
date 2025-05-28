from utils import preprocess_image
import gradio as gr
import numpy as np
import cv2
import tempfile
from tensorflow.keras.models import load_model
from datetime import datetime
import os

from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
import torch.nn.functional as F
from PIL import Image

DATA_PATH = os.environ.get("DEEFAKE_DATA_PATH", "./data")
MODEL_PATH = os.path.join(DATA_PATH, "current_model.h5")
IMG_SIZE = 224
TRUST_WEIGHT_FACE = 0.3
TRUST_WEIGHT_QUALITY = 0.2
TRUST_WEIGHT_MODEL = 0.5

model = None
hf_model = None
hf_processor = None
hf_label_map = {0: "AI", 1: "Deepfake", 2: "Real"}

def load_detection_model():
    global model
    if model is None:
        model = load_model(MODEL_PATH)
    return model

def load_hf_model():
    global hf_model, hf_processor
    if hf_model is None or hf_processor is None:
        hf_model_name = "prithivMLmods/AI-vs-Deepfake-vs-Real-v2.0"
        hf_processor = AutoImageProcessor.from_pretrained(hf_model_name)
        hf_model = AutoModelForImageClassification.from_pretrained(hf_model_name)
    return hf_model, hf_processor

def predict_keras(img: np.ndarray):
    model = load_detection_model()
    proc, _ = preprocess_image(img, model_type="efficientnet", use_face_crop=False, output_size=IMG_SIZE)
    pred = model.predict(np.expand_dims(proc, axis=0))
    prob = float(pred[0][0]) if pred.shape[-1] == 1 else float(pred[0][1])
    label = "FAKE" if prob > 0.5 else "REAL"
    return prob, label

def predict_hf(img: np.ndarray):
    hf_model, hf_processor = load_hf_model()
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert("RGB")
    inputs = hf_processor(images=img_pil, return_tensors="pt")
    with torch.no_grad():
        outputs = hf_model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)
        pred_idx = int(torch.argmax(probs, dim=-1).item())
        prob = float(probs[0][pred_idx])
        label = hf_label_map.get(pred_idx, str(pred_idx))
    if label == "Real":
        label_simple = "REAL"
    else:
        label_simple = "FAKE"
    return prob, label_simple, label

def grad_cam(img: np.ndarray, model=None, pred_index=None):
    import tensorflow as tf
    if model is None:
        model = load_detection_model()
    image, _ = preprocess_image(img, model_type="efficientnet", use_face_crop=False, output_size=IMG_SIZE)
    image = np.expand_dims(image, axis=0)
    last_conv = [l for l in model.layers if "conv" in l.name][-1]
    grad_model = tf.keras.models.Model([model.inputs], [last_conv.output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0].numpy()
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]
    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-10
    heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(
        cv2.cvtColor(cv2.resize(img, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_RGB2BGR), 0.6,
        heatmap, 0.4, 0
    )
    return overlay

def face_detect_confidence(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = detector.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        return 0.0
    area = max([w*h for (x, y, w, h) in faces])
    rel_area = area / float(img.shape[0] * img.shape[1])
    conf = min(1.0, rel_area * 3 / len(faces))
    return conf

def image_quality_score(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return min(1.0, score / 1000)

def trust_score(prob, face_conf, quality_score):
    if prob > 0.5:
        base = 1.0 - prob
    else:
        base = prob
    score = (
        TRUST_WEIGHT_MODEL * base +
        TRUST_WEIGHT_FACE * face_conf +
        TRUST_WEIGHT_QUALITY * quality_score
    )
    return np.clip(score, 0, 1)

leaderboard = []

def ensemble_label_and_prob(keras_prob, keras_label, hf_prob, hf_label):
    votes = [keras_label, hf_label]
    label = "REAL" if votes.count("REAL") > votes.count("FAKE") else "FAKE"
    prob = (keras_prob + hf_prob) / 2
    return prob, label

def process_upload(image=None, video=None, feedback=None):
    report = []
    result_imgs = []
    leaderboard_entry = None
    heatmap = None
    if image is not None:
        arr = np.array(image)
        keras_prob, keras_label = predict_keras(arr)
        hf_prob, hf_label_simple, hf_label_orig = predict_hf(arr)
        ens_prob, ens_label = ensemble_label_and_prob(keras_prob, keras_label, hf_prob, hf_label_simple)
        face_conf = face_detect_confidence(arr)
        quality = image_quality_score(arr)
        trust = trust_score(ens_prob, face_conf, quality)
        cam = grad_cam(arr, model)
        result_imgs.append(cam)
        report.append(f"Prediction (Ensemble): {ens_label}")
        report.append(f" Keras: {keras_label} (Prob: {keras_prob:.3f})")
        report.append(f" HuggingFace: {hf_label_simple} ['{hf_label_orig}'] (Prob: {hf_prob:.3f})")
        report.append(f"Deepfake Probability (Ensemble): {ens_prob:.3f}")
        report.append(f"Face Confidence: {face_conf:.2f}")
        report.append(f"Image Quality: {quality:.2f}")
        report.append(f"Trust Score: {trust:.2f}")
        if trust < 0.4:
            report.append("⚠️ Low trust score: consider a better face image or clearer video frame.")
        leaderboard_entry = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "label": ens_label,
            "keras_label": keras_label,
            "keras_prob": keras_prob,
            "hf_label": hf_label_simple,
            "hf_prob": hf_prob,
            "trust": trust
        }
        leaderboard.append(leaderboard_entry)
    if report:
        report_str = "\n".join(report)
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as repf:
            repf.write(report_str)
        download_path = repf.name
    else:
        download_path = None
    return "\n".join(report), result_imgs, leaderboard, download_path, heatmap

def leaderboard_table():
    if not leaderboard:
        return [["-", "-", "-", "-", "-"]]
    return [[
        e["time"], e["label"],
        f"Keras:{e['keras_label']}({e['keras_prob']:.2f})",
        f"HF:{e['hf_label']}({e['hf_prob']:.2f})",
        f"{e['trust']:.2f}"
    ] for e in leaderboard]

def download_report(path):
    with open(path, "rb") as f:
        return f.read()

def main():
    api_curl = f"""curl -X POST \\
  -F "image=@your_face.jpg" \\
  http://localhost:7860/api/predict"""

    with gr.Blocks(title="Deepfake Detector MVP") as demo:
        gr.Markdown("""
        # 🚀 Deepfake Detector MVP
        - Upload an image **or** a video (1 frame sampled)
        - See prediction, explainability, and a *Trust Score*
        - Download a result report
        - Add feedback to improve our AI!
        """)
        gr.Markdown("**Try the demo API:**")
        gr.Code(api_curl, language="bash")
        with gr.Row():
            with gr.Column():
                img_input = gr.Image(type="pil", label="Upload Image (optional)")
                vid_input = gr.Video(label="Upload Video (optional)")
                feedback_input = gr.Textbox(label="Feedback for this result (optional)")
                submit = gr.Button("Predict!")
                download_btn = gr.Download(label="Download Report")
            with gr.Column():
                result_box = gr.Textbox(label="Result")
                cam_output = gr.Gallery(label="Explainability (Grad-CAM)").style(grid=1)
                leaderboard_output = gr.Dataframe(
                    headers=["Time", "Ensemble", "Keras", "HuggingFace", "Trust"],
                    label="Suspicious Uploads Leaderboard",
                    interactive=False
                )
                heatmap_output = gr.Image(type="numpy", label="Novel: Frame Consistency Heatmap")
        def ui_predict(img, vid, fb):
            result, cams, lb, repath, heatmap = process_upload(img, vid, fb)
            return result, cams, leaderboard_table(), repath, heatmap
        submit.click(
            ui_predict,
            inputs=[img_input, vid_input, feedback_input],
            outputs=[result_box, cam_output, leaderboard_output, download_btn, heatmap_output]
        )
        download_btn.click(download_report, inputs=[download_btn], outputs=gr.File())

    demo.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    main()

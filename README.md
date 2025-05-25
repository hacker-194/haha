# Deepfake Detection Pipeline

A robust, extensible, and production-ready pipeline for detecting deepfakes in images and videos. This project features frame extraction, face preprocessing, deep learning-based classification, user feedback, self-learning model retraining, and a user-friendly Gradio UI.

---

## Features

- **Frame Extraction**: Extracts frames from videos, supports batch processing, resumes partial extractions, and provides detailed logs.
- **Face Preprocessing**: Detects and crops faces using MTCNN, falls back to resizing when no face is found, batch processes frames for downstream ML tasks.
- **Deepfake Detection**: 
  - Ensemble of HuggingFace Transformers and Keras models.
  - Supports both EfficientNet and LSTM backends.
  - Trust score based on model prediction, face detection confidence, and image quality.
  - Robust error handling and logging.
- **User Feedback & Self-Learning**: 
  - Collects feedback on predictions with automatic deduplication.
  - Automatically retrains the model when enough new feedback is received.
  - Backs up previous models before overwriting.
- **Interactive UI**: 
  - Gradio interface for easy image/video uploads and predictions.
  - Downloadable reports, explainability via Grad-CAM, and a leaderboard for suspicious uploads.
- **Command Line Interface**: All core components support CLI execution for integration in scripts and pipelines.

---

## Setup

### Requirements

- Python 3.7+
- [See `requirements.txt`](#) *(not included here, but ensure to install dependencies such as `opencv-python`, `tensorflow`, `torch`, `transformers`, `mtcnn`, `gradio`, `tqdm`, `pyyaml`)*

Install dependencies:
```bash
pip install -r requirements.txt
```

### Environment Variables

- `DEEFAKE_DATA_PATH`: Root directory for data (default: `./deepfake_data`)
- `DATA_PATH`: Used in some scripts for specifying the data root.

---

## Usage

### 1. Frame Extraction

Extract frames from videos:

```bash
python frame_extraction.py \
  --manipulated_dir ./manipulated \
  --original_dir ./original \
  --output_manipulated ./frames_manip \
  --output_original ./frames_orig \
  --every_n_frames 10
```

### 2. Face Preprocessing

Crop faces and preprocess images for ML models:

```bash
python preprocessing_pipeline.py \
  --input_dir ./frames_manip \
  --output_dir ./processed/manipulated \
  --use_face_crop \
  --output_size 224 \
  --model_type efficientnet
```

### 3. Deepfake Prediction

Predict whether an image is a deepfake or real:

```bash
python predict.py image.jpg --model model.h5 --model_type efficientnet
```

### 4. Batch Prediction & Feedback (Model Ensemble)

Automatically predicts and collects feedback for batches of images:

```bash
python model.py
```

### 5. Self-Learning Retraining

Automatically retrain the model when enough new feedback is collected:

```bash
python selflearn.py
```

### 6. Gradio UI

Start an interactive web app for uploading images/videos and viewing results:

```bash
python ui.py
```
Navigate to `http://localhost:7860` in your browser.

---

## Project Structure

```
.
├── frame_extraction.py         # Frame extraction from videos
├── preprocessing_pipeline.py   # Robust image preprocessing with face detection/cropping
├── processing.py               # Batch face cropping using MTCNN
├── predict.py                  # Deepfake prediction script (EfficientNet/LSTM)
├── model.py                    # Batch prediction, feedback collection, ensemble logic
├── selflearn.py                # Automatic retrainer based on new feedback
├── ui.py                       # Gradio web interface
├── requirements.txt            # Python dependencies (to be created)
└── README.md                   # Project documentation
```

---

## Example Workflow

1. **Extract frames** from your dataset of videos.
2. **Preprocess** the frames (face detection, resizing) to prepare ML-ready images.
3. **Train or load models**, then run predictions (either via CLI or the Gradio UI).
4. **Collect user feedback** to improve model accuracy.
5. **Retrain** the model automatically with new labeled samples collected from users.

---

## Feedback & Contribution

- Issues and pull requests are welcome!
- For major changes, please open an issue first to discuss what you would like to change.
- Contributions for new model types, UI enhancements, and pipeline improvements are especially appreciated.

---

## License

This project is licensed under the MIT License.

---

## Acknowledgements

- [HuggingFace Transformers](https://huggingface.co/)
- [Gradio](https://gradio.app/)
- [TensorFlow](https://www.tensorflow.org/)
- [OpenCV](https://opencv.org/)
- [MTCNN](https://github.com/ipazc/mtcnn)

# haha: Smart Face Image Preprocessing Pipeline

![License](https://img.shields.io/github/license/hacker-194/haha)

## 🚩 About
A robust, flexible, and lightning-fast image preprocessing pipeline with face detection, cropping, and more — ready for your machine learning projects.

## ✨ Features
- Automatic face detection and cropping (MTCNN)
- Configurable via CLI, YAML, or environment
- Robust logging and progress bars
- Extensible: easily add new steps or detectors
- Parallel/batch processing for speed (NEW!)

## 🏁 Quickstart

```bash
# Install requirements
pip install -r requirements.txt

# Run preprocessing
python preprocessing_pipeline.py --input_dir ./input --output_dir ./output --use_face_crop --output_size 128
```

## ⚙️ Configuration

- Supports config.yaml, CLI flags, and env variables.
- Sample config:
```yaml
log_file: "preprocess.log"
image_exts: [".jpg", ".png"]
output_size: 128
```

## 🧪 Testing

```bash
pytest
# or
python -m unittest discover
```

## 📸 Screenshots

| Before      | After       |
|-------------|-------------|
| ![before](examples/before.jpg) | ![after](examples/after.jpg) |

## 🙋 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## 🛠️ License

MIT License

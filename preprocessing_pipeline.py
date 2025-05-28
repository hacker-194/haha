from utils import setup_logging, get_detector, detect_and_crop_face, preprocess_image, is_image_file
import os
import cv2
import logging
from tqdm import tqdm
import argparse
import sys
import yaml

DEFAULT_CONFIG = {
    "log_file": "preprocess.log",
    "image_exts": [".jpg", ".jpeg", ".png"],
}

def load_config(config_path=None):
    config = DEFAULT_CONFIG.copy()
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            user_config = yaml.safe_load(f)
        config.update(user_config)
    for key in config:
        envval = os.environ.get(key.upper())
        if envval is not None:
            config[key] = envval
    return config

def preprocess_directory(input_dir, output_dir, use_face_crop=True, output_size=128, model_type="lstm", image_exts=None):
    os.makedirs(output_dir, exist_ok=True)
    detector = get_detector() if use_face_crop else None
    files = [f for f in os.listdir(input_dir) if is_image_file(f, image_exts)]
    stats = {"processed": 0, "skipped": 0, "failed": 0, "no_face": 0}
    for fname in tqdm(files, desc=f"Preprocessing {os.path.basename(input_dir)}"):
        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, fname)
        try:
            img = cv2.imread(in_path)
            if img is None:
                logging.warning(f"Could not read image: {in_path}")
                stats["skipped"] += 1
                continue
            proc, face_found = preprocess_image(img, model_type=model_type, use_face_crop=use_face_crop, output_size=output_size, detector=detector)
            if use_face_crop and not face_found:
                logging.warning(f"No face detected in: {in_path}")
                stats["no_face"] += 1
            out_img = np.clip(proc * 255, 0, 255).astype(np.uint8) if proc.max() <= 1.0 else proc.astype(np.uint8)
            cv2.imwrite(out_path, cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
            stats["processed"] += 1
        except Exception as e:
            logging.error(f"Error processing {in_path}: {e}")
            stats["failed"] += 1
    return stats

def main():
    parser = argparse.ArgumentParser(description="Preprocess images with optional face cropping.")
    parser.add_argument("--input_dir", required=True, help="Input image directory")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--use_face_crop", action="store_true", help="Crop faces using MTCNN")
    parser.add_argument("--output_size", type=int, default=128, help="Output image size")
    parser.add_argument("--model_type", type=str, default="lstm", choices=["lstm", "efficientnet"], help="Model type (lstm or efficientnet)")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--log_file", type=str, default=None, help="Log file path")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(args.log_file or config.get("log_file"))

    logging.info(f"Starting preprocessing pipeline with args: {args}")

    if not os.path.isdir(args.input_dir):
        logging.error(f"Input directory does not exist: {args.input_dir}")
        sys.exit(1)
    if os.path.abspath(args.input_dir) == os.path.abspath(args.output_dir):
        logging.error(f"Input and output directories must be different.")
        sys.exit(1)

    stats = preprocess_directory(
        args.input_dir,
        args.output_dir,
        use_face_crop=args.use_face_crop,
        output_size=args.output_size,
        model_type=args.model_type,
        image_exts=config.get("image_exts", [".jpg", ".jpeg", ".png"]),
    )

    logging.info("--- Preprocessing Summary ---")
    logging.info(f"Processed: {stats['processed']}")
    logging.info(f"Skipped (unreadable): {stats['skipped']}")
    logging.info(f"Failed (error): {stats['failed']}")
    if args.use_face_crop:
        logging.info(f"No Face Detected: {stats['no_face']}")
    logging.info("Done.")

if __name__ == "__main__":
    main()

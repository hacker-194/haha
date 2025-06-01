import os
import sys
import logging

try:
    import cv2
except ImportError:
    print("Error: OpenCV (cv2) is not installed. Please install it to proceed.")
    sys.exit(1)

try:
    import yaml
except ImportError:
    print("Error: PyYAML is not installed. Please install it to proceed.")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("Error: tqdm is not installed. Please install it to proceed.")
    sys.exit(1)

try:
    from utils import setup_logging, get_detector, detect_and_crop_face, preprocess_image, is_image_file
except ImportError as e:
    print(f"Error importing utility functions: {e}")
    sys.exit(1)

import argparse

DEFAULT_CONFIG = {
    "log_file": "preprocess.log",
    "image_exts": [".jpg", ".jpeg", ".png"],
}

def load_config(config_path=None):
    config = DEFAULT_CONFIG.copy()
    if config_path:
        if not os.path.exists(config_path):
            print(f"Warning: Config file '{config_path}' does not exist. Using defaults.")
        else:
            try:
                with open(config_path, "r") as f:
                    user_config = yaml.safe_load(f)
                if user_config:
                    config.update(user_config)
            except Exception as e:
                print(f"Warning: Could not load config file: {e}")
    for key in config:
        envval = os.environ.get(key.upper())
        if envval is not None:
            config[key] = envval
    return config

def preprocess_directory(input_dir, output_dir, use_face_crop=True, output_size=128, model_type="lstm", image_exts=None):
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        logging.error(f"Failed to create output directory '{output_dir}': {e}")
        sys.exit(1)
    if not os.access(output_dir, os.W_OK):
        logging.error(f"Output directory '{output_dir}' is not writable.")
        sys.exit(1)
    detector = get_detector() if use_face_crop else None
    files = [f for f in os.listdir(input_dir) if is_image_file(f, image_exts)]
    if not files:
        logging.warning(f"No image files found in directory: {input_dir}")
    stats = {"processed": 0, "skipped": 0, "failed": 0, "no_face": 0}
    for fname in tqdm(files, desc=f"Preprocessing {os.path.basename(input_dir)}"):
        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, fname)
        if os.path.exists(out_path):
            logging.info(f"Output file already exists and will be skipped: {out_path}")
            stats["skipped"] += 1
            continue
        try:
            img = cv2.imread(in_path)
            if img is None or not hasattr(img, 'shape'):
                logging.warning(f"Could not read or invalid image: {in_path}")
                stats["skipped"] += 1
                continue
            proc, face_found = preprocess_image(img, model_type=model_type, use_face_crop=use_face_crop, output_size=output_size, detector=detector)
            if proc is None:
                logging.warning(f"Preprocessing failed for: {in_path}")
                stats["failed"] += 1
                continue
            if use_face_crop and not face_found:
                logging.warning(f"No face detected in: {in_path}")
                stats["no_face"] += 1
            try:
                import numpy as np
            except ImportError:
                logging.error("numpy is required but not installed.")
                sys.exit(1)
            out_img = np.clip(proc * 255, 0, 255).astype(np.uint8) if proc.max() <= 1.0 else proc.astype(np.uint8)
            try:
                cv2.imwrite(out_path, cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
            except Exception as e:
                logging.error(f"Failed to write output image: {out_path}, error: {e}")
                stats["failed"] += 1
                continue
            stats["processed"] += 1
        except PermissionError:
            logging.error(f"Permission denied while processing {in_path}")
            stats["failed"] += 1
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

    if not os.path.isdir(args.input_dir):
        print(f"Input directory does not exist: {args.input_dir}")
        sys.exit(1)
    if os.path.abspath(args.input_dir) == os.path.abspath(args.output_dir):
        print("Input and output directories must be different.")
        sys.exit(1)
    if args.output_size <= 0:
        print("Output size must be a positive integer.")
        sys.exit(1)

    config = load_config(args.config)
    setup_logging(args.log_file or config.get("log_file"))

    logging.info(f"Starting preprocessing pipeline with args: {vars(args)}")

    try:
        stats = preprocess_directory(
            args.input_dir,
            args.output_dir,
            use_face_crop=args.use_face_crop,
            output_size=args.output_size,
            model_type=args.model_type,
            image_exts=config.get("image_exts", [".jpg", ".jpeg", ".png"]),
        )
    except KeyboardInterrupt:
        logging.warning("Interrupted by user! Printing summary so far...")

    logging.info("--- Preprocessing Summary ---")
    logging.info(f"Processed: {stats.get('processed', 0)}")
    logging.info(f"Skipped (unreadable or exists): {stats.get('skipped', 0)}")
    logging.info(f"Failed (error): {stats.get('failed', 0)}")
    if args.use_face_crop:
        logging.info(f"No Face Detected: {stats.get('no_face', 0)}")
    logging.info("Done.")

if __name__ == "__main__":
    main()

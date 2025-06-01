import os, cv2, logging, sys, argparse
from tqdm import tqdm

def setup_logging(level=logging.INFO, log_file=None):
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file: handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s", handlers=handlers)

def list_video_files(d, exts=None): 
    exts = exts or ['.mp4', '.avi', '.mov', '.mkv']
    return [f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f)) and os.path.splitext(f)[1].lower() in exts]

def check_partial_extraction(outdir, prefix, label, expect, ext):
    imgs = [f for f in os.listdir(outdir) if f.startswith(f"{prefix}_{label}_frame") and f.endswith(ext)]
    return len(imgs) >= expect

def extract_frames_from_videos(vdir, odir, label, every_n_frames=1, overwrite=False, extensions=None, img_ext=".jpg"):
    logger = logging.getLogger()
    if not os.path.exists(vdir):
        logger.error(f"Video directory does not exist: {vdir}")
        return 0,0
    os.makedirs(odir, exist_ok=True)
    vfiles = list_video_files(vdir, extensions)
    if not vfiles:
        logger.warning(f"No video files in {vdir}.")
        return 0,0
    total_videos, total_saved = len(vfiles), 0
    for idx, vname in enumerate(vfiles,1):
        vpath = os.path.join(vdir, vname)
        prefix = os.path.splitext(vname)[0]
        cap = cv2.VideoCapture(vpath)
        try:
            if not cap.isOpened():
                logger.warning(f"Failed to open {vpath}")
                continue
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if not total_frames:
                logger.warning(f"Zero frames in {vpath}")
                continue
            expect = (total_frames + every_n_frames - 1) // every_n_frames
            first = os.path.join(odir, f"{prefix}_{label}_frame0{img_ext}")
            already = os.path.exists(first)
            partial = False
            if already and not overwrite:
                partial = not check_partial_extraction(odir, prefix, label, expect, img_ext)
                if not partial:
                    logger.info(f"[{idx}/{len(vfiles)}] Frames for '{vname}' already exist. Skipping.")
                    continue
                logger.warning(f"[{idx}/{len(vfiles)}] Partial extraction for '{vname}'. Will resume.")
            count, saved = 0, 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            pbar = tqdm(total=total_frames, desc=f"Extracting {vname}", unit="frame", leave=False)
            while True:
                ret, frame = cap.read()
                if not ret: break
                if count % every_n_frames == 0:
                    fpath = os.path.join(odir, f"{prefix}_{label}_frame{count}{img_ext}")
                    if not overwrite and os.path.exists(fpath):
                        saved += 1; count += 1; pbar.update(1); continue
                    try:
                        if cv2.imwrite(fpath, frame): saved += 1
                        else: logger.error(f"Failed to write frame {count} of '{vname}' to {fpath}")
                    except Exception as e:
                        logger.error(f"Exception saving frame {count} of '{vname}': {e}")
                count += 1; pbar.update(1)
            pbar.close()
            logger.info(f"[{idx}/{len(vfiles)}] '{vname}': {saved} frames saved (Total: {total_frames})")
            total_saved += saved
        except Exception as e:
            logger.error(f"Exception processing video {vpath}: {e}")
        finally:
            cap.release()
    return total_videos, total_saved

def print_summary(summary):
    logger = logging.getLogger()
    logger.info("----- Extraction Summary -----")
    for k, v in summary.items():
        logger.info(f"{k}: {v[0]} files / {v[1]} frames extracted")

def main():
    parser = argparse.ArgumentParser(description="Extract frames from video files in a directory.")
    parser.add_argument("--manipulated_dir", required=True)
    parser.add_argument("--original_dir", required=True)
    parser.add_argument("--output_manipulated", required=True)
    parser.add_argument("--output_original", required=True)
    parser.add_argument("--every_n_frames", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--img_ext", default=".jpg")
    parser.add_argument("--log_file", default=None)
    args = parser.parse_args()
    setup_logging(log_file=args.log_file)
    summary = {}
    v1, f1 = extract_frames_from_videos(args.manipulated_dir, args.output_manipulated, "manipulated",
        every_n_frames=args.every_n_frames, overwrite=args.overwrite, img_ext=args.img_ext)
    summary["manipulated"] = (v1, f1)
    v2, f2 = extract_frames_from_videos(args.original_dir, args.output_original, "original",
        every_n_frames=args.every_n_frames, overwrite=args.overwrite, img_ext=args.img_ext)
    summary["original"] = (v2, f2)
    print_summary(summary)

if __name__ == "__main__":
    main()

import os, argparse, json, numpy as np, pandas as pd
from tqdm import tqdm
from skimage import io as skio

from .config import PipelineConfig
from .io_utils import list_images_recursive, load_split_channels, ensure_dir, save_json, normalize_to_uint8
from .segmentation import segment_nuclei
from .crops import extract_nucleus_crop
from .features import extract_all_features
from .models import crossval_random_forest

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=str, required=True)
    ap.add_argument("--work-dir", type=str, required=True)
    ap.add_argument("--channel-mode", type=str, default="split", choices=["split","single"])
    ap.add_argument("--r-suffix", type=str, default="_R")
    ap.add_argument("--g-suffix", type=str, default="_G")
    ap.add_argument("--segmenter", type=str, default="otsu")
    ap.add_argument("--min-area", type=int, default=150)
    ap.add_argument("--max-area", type=int, default=999999)
    ap.add_argument("--gaussian-sigma", type=float, default=1.0)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--enable-cnn", action="store_true")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--random-state", type=int, default=42)
    return ap.parse_args()

def main():
    args = parse_args()
    cfg = PipelineConfig(
        input_dir=args.input_dir, work_dir=args.work_dir, channel_mode=args.channel_mode,
        segmenter=args.segmenter, min_area=args.min_area, max_area=args.max_area,
        gaussian_sigma=args.gaussian_sigma, n_jobs=args.n_jobs, enable_cnn=args.enable_cnn,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, random_state=args.random_state
    )
    cfg.r_suffix = args.r_suffix; cfg.g_suffix = args.g_suffix

    ensure_dir(cfg.work_dir); ensure_dir(os.path.join(cfg.work_dir,"crops")); ensure_dir(os.path.join(cfg.work_dir,"models"))
    paths = list_images_recursive(cfg.input_dir, cfg.image_exts)
    if cfg.channel_mode == "split":
        paths = [p for p in paths if any(p.endswith(cfg.r_suffix + ext) for ext in cfg.image_exts)]
    if not paths:
        raise SystemExit(f"No images found in {cfg.input_dir}")

    all_crops, all_masks, meta, labels = [], [], [], []
    for p in tqdm(paths, desc="Segmenting"):
        if cfg.channel_mode == "split":
            R, G = load_split_channels(p, cfg.r_suffix, cfg.g_suffix)
        else:
            R = skio.imread(p); G = None
        R8 = normalize_to_uint8(R)
        labels_map, _ = segment_nuclei(R8, method=cfg.segmenter, gaussian_sigma=cfg.gaussian_sigma,
                                       min_area=cfg.min_area, max_area=cfg.max_area)
        for lab in range(1, labels_map.max()+1):
            crop, msk = extract_nucleus_crop(R8, labels_map, lab, crop_size=cfg.crop_size)
            if crop is None: continue
            out_name = f"{os.path.splitext(os.path.basename(p))[0]}_lab{lab:03d}.png"
            out_path = os.path.join(cfg.work_dir, "crops", out_name)
            skio.imsave(out_path, crop, check_contrast=False)
            all_crops.append(crop); all_masks.append(msk); meta.append({"image": p, "label_id": int(lab), "crop_path": out_path})
            # weak labels off by default in baseline (set to 0); replace with your labels later
            labels.append(0)

    all_crops = np.stack(all_crops) if all_crops else np.zeros((0, cfg.crop_size, cfg.crop_size), dtype=np.uint8)
    all_masks = np.stack(all_masks) if all_masks else np.zeros_like(all_crops)
    labels = np.array(labels, dtype=int)
    save_json(meta, os.path.join(cfg.work_dir, "crops_meta.json"))
    np.save(os.path.join(cfg.work_dir, "labels.npy"), labels)

    rows = []
    for i in tqdm(range(len(all_crops)), desc="Features"):
        feats = extract_all_features(
            all_crops[i], all_masks[i],
            cfg.distances, cfg.angles,
            cfg.lbp_radius, cfg.lbp_n_points,
            cfg.hog_pixels_per_cell, cfg.hog_cells_per_block,
            cfg.gabor_frequencies, cfg.gabor_thetas
        )
        feats["y"] = int(labels[i]); feats["idx"] = i; rows.append(feats)
    df = pd.DataFrame(rows).set_index("idx")
    df.to_csv(os.path.join(cfg.work_dir, "features.csv"))

    # Train RF only if labels provided (baseline leaves y=0s).
    if len(set(labels.tolist())) >= 2:
        y = df["y"].values.astype(int); X = df.drop(columns=["y"]).values.astype(np.float32)
        rf_cv = crossval_random_forest(X, y, cfg, cfg.work_dir)
        with open(os.path.join(cfg.work_dir,"cv_metrics.json"),"w") as f: json.dump(rf_cv, f, indent=2)
        print("CV metrics written to cv_metrics.json")
    else:
        print("No class variation in labels; extracted features and crops are saved. Add labels and re-run.")

if __name__ == "__main__":
    main()

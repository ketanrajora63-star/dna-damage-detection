import gradio as gr, numpy as np, joblib
from pathlib import Path
from skimage import io as skio
from pipeline.io_utils import normalize_to_uint8
from pipeline.segmentation import segment_nuclei
from pipeline.crops import extract_nucleus_crop
from pipeline.features import extract_all_features
from pipeline.config import PipelineConfig

def run_inference(r_img, model_dir="outputs/run1"):
    cfg = PipelineConfig()
    r8 = normalize_to_uint8(r_img)
    labels, _ = segment_nuclei(r8, method=cfg.segmenter, gaussian_sigma=cfg.gaussian_sigma,
                               min_area=cfg.min_area, max_area=cfg.max_area)
    if labels.max() == 0:
        return [], "No nuclei found."
    rf_path = Path(model_dir) / "models" / "rf.joblib"
    rf = joblib.load(rf_path) if rf_path.exists() else None
    crops, out, feats_list = [], [], []
    for lab in range(1, labels.max()+1):
        crop, msk = extract_nucleus_crop(r8, labels, lab, crop_size=cfg.crop_size)
        if crop is None: continue
        feats = extract_all_features(crop, msk, cfg.distances, cfg.angles, cfg.lbp_radius, cfg.lbp_n_points,
                                     cfg.hog_pixels_per_cell, cfg.hog_cells_per_block, cfg.gabor_frequencies, cfg.gabor_thetas)
        feats_list.append(feats); crops.append(crop)
    if rf is not None and feats_list:
        import pandas as pd
        X = pd.DataFrame(feats_list).fillna(0.0).values.astype(np.float32)
        prob = rf.predict_proba(X)[:,1]; pred = (prob>=0.5).astype(int)
        for i, c in enumerate(crops):
            out.append((c, f"nucleus {i:02d}  probDamaged={prob[i]:.3f}  pred={pred[i]}"))
    else:
        for i, c in enumerate(crops):
            out.append((c, f"nucleus {i:02d}"))
    return out, f"Detected {len(out)} nuclei."

with gr.Blocks() as demo:
    gr.Markdown("# DNA Damage Detector — Baseline Inference")
    img = gr.Image(type="numpy", label="Upload R-channel image", image_mode="L")
    model_dir = gr.Textbox(value="outputs/run1", label="Model directory")
    btn = gr.Button("Detect")
    gallery = gr.Gallery(columns=4, height=450)
    status = gr.Markdown("")
    btn.click(run_inference, inputs=[img, model_dir], outputs=[gallery, status])

if __name__ == "__main__":
    demo.launch()

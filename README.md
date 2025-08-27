# DNA Damage Detection (Baseline Repo)

Clean, from-scratch baseline pipeline inspired by the MQP approach.
Includes:
- `pipeline/` (segmentation, crops, features, models, training)
- `notebooks/` (end-to-end demo)
- `app_gradio.py` (local inference UI)
- `.gitignore` (keeps data/outputs out of repo)
- `requirements.txt`

## Quickstart
```bash
pip install -r requirements.txt

# Train baseline (recursive over data/raw/)
python -m pipeline.train --input-dir data/raw --work-dir outputs/run1 --segmenter otsu --n-jobs -1

# (Optional) Enable CNN if you have PyTorch/GPU
python -m pipeline.train --input-dir data/raw --work-dir outputs/run_cnn --segmenter otsu   --enable-cnn --epochs 15 --batch-size 32 --lr 3e-4

# Launch local UI (after training)
python app_gradio.py
```

# DNA Damage Detection (Baseline Repo)

A clean, from-scratch baseline pipeline inspired by the [MQP DNA Damage Detection project](https://github.com/Mcdonoughd/MQP).  
The goal is to detect **DNA damage in fluorescent microscopy images of nuclei** and improve upon the baseline accuracy of ~74% reported in prior work.

---

## 📂 Repository Structure

dna-damage-detection/
├─ pipeline/ # Core code (segmentation, crops, features, models, training)
│ ├─ config.py
│ ├─ io_utils.py
│ ├─ segmentation.py
│ ├─ crops.py
│ ├─ features.py
│ ├─ models.py
│ └─ train.py
├─ notebooks/
│ └─ Pipeline_Demo.ipynb # Example notebook (end-to-end demo)
├─ app_gradio.py # Gradio app for local inference
├─ requirements.txt # Dependencies
├─ README.md # Project overview & instructions
└─ .gitignore # Ignore large data/outputs

yaml
Copy code

---

## ⚡ Quickstart

### 1. Setup environment
```bash
git clone https://github.com/ketanrajora63-star/dna-damage-detection.git
cd dna-damage-detection
pip install -r requirements.txt
2. Prepare your dataset
Organize input images into data/raw/.
Supported formats:

Split channels: e.g., sample1_R.tif (DNA/RFP) + sample1_G.tif (Damage/GFP)

Single channel: just sample1_R.tif (if no GFP available)

3. Train baseline (Random Forest on handcrafted features)
bash
Copy code
python -m pipeline.train \
  --input-dir data/raw \
  --work-dir outputs/run1 \
  --segmenter otsu \
  --n-jobs -1
This will:

Segment nuclei (Otsu + Watershed baseline)

Extract 150×150 crops (normalized to major axis)

Compute shape + texture features (Haralick, LBP, HOG, Gabor, Hu, LoG)

Train a Random Forest with 10-fold CV (if labels available)

Save crops, features, and metrics to outputs/run1/

🎨 Interactive App (Gradio)
After training:

bash
Copy code
python app_gradio.py
Upload an R-channel image

Nuclei are segmented and displayed with predicted probabilities of DNA damage

🔬 Roadmap (Planned Improvements)
 Better segmentation (Cellpose, StarDist, or U-Net)

 CNN fine-tuning (ResNet18/EfficientNet with focal loss)

 Ensemble models (handcrafted + deep features)

 Class balancing (data augmentation, SMOTE, or focal loss)

 Evaluation metrics optimized for Recall/F1 (damage detection is priority)

📊 Results
Baseline features are in place. Accuracy will improve as we:

Add better segmentation

Enable CNN fine-tuning

Combine models into a stacked ensemble

📸 Example Output (to add later)
Once training is run on your B2/C2 datasets, add example images here to show:

Nucleus segmentation

Crop gallery with predicted labels

Example (placeholder):

Input Image (R channel)	Segmented Nuclei	Crops with Predictions

📝 Citation
If you use this baseline repo, please reference the original MQP project and this repo:

Mcdonoughd/MQP

This repo: https://github.com/ketanrajora63-star/dna-damage-detection

yaml
Copy code

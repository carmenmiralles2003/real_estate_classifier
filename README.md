# Real-Estate Image Classifier

Automatic classification of real-estate scene images into 15 categories using transfer learning with ConvNeXt and Swin Transformer. Achieves **97.37% accuracy** and **97.62% macro-F1** on the test set using a ConvNeXt ensemble with TTA.

**W&B Project:** [real-estate-classifier](https://wandb.ai/202525416-universidad-pontificia-comillas/real-estate-classifier)

---

## Business Problem

Real-estate portals handle massive amounts of property photos uploaded by individuals and agencies. Manually labeling each image by room type (bedroom, kitchen, living room...) is expensive, inconsistent, and doesn't scale. This system automatically classifies property images into 15 scene categories, delivering:

- **Better user experience:** buyers can filter by room type and galleries are ordered coherently.
- **Quality control:** automatically detects and flags mislabeled images.
- **Cost savings:** eliminates manual review (~0.5s per image × millions of listings).

The system responds in under 500ms per image, runs on a mid-range 8GB GPU, and is accessible via a standard REST API so any real-estate portal can integrate it without changing its architecture.

---

## Scene Classes (15)

`Bedroom` · `Coast` · `Forest` · `Highway` · `Industrial` · `Inside city` · `Kitchen` · `Living room` · `Mountain` · `Office` · `Open country` · `Store` · `Street` · `Suburb` · `Tall building`

---

## Results

| Model | Parameters | Val Acc | Test+TTA | Macro-F1 |
|---|---|---|---|---|
| ConvNeXt-Small (288px) | 49.9M | 97.75% | 96.93% | 96.66% |
| ConvNeXt-Tiny (288px) | 28.4M | 97.31% | 96.05% | 96.50% |
| Swin-Tiny (224px) | 28.0M | 96.41% | 95.61% | 95.96% |
| **Ensemble (Tiny + Small)** | 78.3M | — | **97.37%** | **97.62%** |

For production, we recommend deploying **ConvNeXt-Small with TTA** as the main model (~50ms/image on GPU), and reserving the ensemble for batch pipelines where latency is not critical.

---

## Demo

https://github.com/user-attachments/assets/Test_videocam_classifier.mp4

---

## Project Structure

```
practica_ml2_sucio/
├── README.md
├── requirements.txt
├── .gitignore
├── .gitattributes              ← Git LFS config for .pth files
├── Test_videocam_classifier.mp4
├── data/
│   ├── raw/                    ← merged images per class
│   └── processed/              ← auto-generated train/val/test (70/15/15)
├── models/
│   ├── best_model.pth              ← best single model (ConvNeXt-Small)
│   ├── convnext_small_288.pth
│   ├── convnext_tiny_288.pth
│   ├── swin_tiny_224.pth
│   ├── effb0_full_finetune.pth
│   ├── best_model_meta.json
│   └── phase2_results.json
├── src/
│   ├── config.py               ← central configuration
│   ├── dataset.py              ← data loading & augmentation
│   ├── model.py                ← model factory (early phases)
│   ├── train.py                ← basic training loop (early phases)
│   ├── evaluate.py             ← metrics & confusion matrix
│   ├── ensemble.py             ← ensemble inference (early phases)
│   └── sweep_config.yaml       ← W&B sweep config (early phases)
├── api/
│   ├── main.py                 ← FastAPI inference server
│   └── schemas.py              ← request/response schemas
├── app/
│   └── streamlit_app.py        ← Streamlit front-end
├── scripts/
│   ├── download_data.py        ← dataset merge & 70/15/15 split
│   ├── run_experiments.py      ← systematic experimentation (phases 1–2)
│   ├── train_nuclear.py        ← final training pipeline
│   └── train_phase2.py         ← phase 2 training pipeline
├── wandb_plots/
│   ├── Test_3_best_models.png
│   ├── Train_3_best_models.png
│   └── Train_all_models.png
└── dataset/
    └── dataset/
        ├── training/           ← professor's original split
        └── validation/
```

---

## Setup & Execution (Windows + PowerShell)

### First time only

```powershell
# 1. Install Git LFS from https://git-lfs.com, then:
git lfs install
git clone https://github.com/nachomgf/real-estate-classifier
cd real-estate-classifier
git lfs pull   # downloads .pth model files

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1
# If blocked by PowerShell:
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned

# 3. Install dependencies
pip install -r requirements.txt
pip install "numpy<2" "opencv-python-headless<4.11" av streamlit-webrtc
```

### Every time you want to run the app

```powershell
# Terminal 1 — Backend (FastAPI)
cd real-estate-classifier
.venv\Scripts\Activate.ps1
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 — Frontend (Streamlit)
cd real-estate-classifier
.venv\Scripts\Activate.ps1
streamlit run app/streamlit_app.py --server.port 8501
```

Open **http://localhost:8501** in your browser.  
Swagger API docs available at **http://localhost:8000/docs**

---

## Reproducing Experiments

### Prepare dataset (requires professor's dataset in `dataset/dataset/`)

```powershell
python scripts/download_data.py
```

### Run final training (requires W&B login)

```powershell
wandb login
python scripts/train_nuclear.py
```

### Run early-phase experiments (EfficientNet/ResNet baselines)

```powershell
python scripts/run_experiments.py
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/predict` | Classify an image. Input: multipart image (JPEG/PNG/WebP, max 10MB). Output: `predicted_class`, `confidence`, `probabilities`. |
| `GET` | `/health` | Service status: model loaded, device (CPU/CUDA), active model. |
| `GET` | `/models` | List available models with metrics. |
| `POST` | `/models/select` | Hot-swap model without restarting the server. |

Full interactive documentation at **http://localhost:8000/docs**

---

## W&B Experimentation

All runs, hyperparameter sweeps, and model comparisons are tracked at:

[https://wandb.ai/202525416-universidad-pontificia-comillas/real-estate-classifier](https://wandb.ai/202525416-universidad-pontificia-comillas/real-estate-classifier)

**Experimentation phases:**
- **Phase 1 — CPU baselines:** EfficientNet-B0 at 224px → 84–87% val accuracy.
- **Phase 2 — Full fine-tuning on GPU:** EfficientNet-B0 with AdamW + AMP → 94.46% val accuracy.
- **Phase 3 — Advanced backbones:** ConvNeXt and Swin pretrained on ImageNet-22k → ConvNeXt-Tiny first to exceed 97% val accuracy.
- **Phase 4 — Ensemble + TTA:** best combination ConvNeXt-Tiny + ConvNeXt-Small → 97.37% test accuracy.

Training curves are also saved locally in `wandb_plots/`.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| "API connected but no model loaded" | Check `.pth` files exist in `models/`. If they weigh only a few bytes, run `git lfs pull`. |
| "Cannot reach API port=8001" | Change API URL in the Streamlit sidebar to `http://localhost:8000`. |
| `streamlit` or `uvicorn` not recognized | Make sure the virtual environment is active. |
| Port 8501 already in use | Use `--server.port 8502`. |
| numpy/opencv conflicts | Run `pip install "numpy<2" "opencv-python-headless<4.11"` |
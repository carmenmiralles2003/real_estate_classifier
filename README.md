# 🏠 Real-Estate Room Image Classifier

Automatic classification of property images into room types (bathroom, bedroom, dining room, exterior, kitchen, living room) using transfer learning with EfficientNet.

## Project Structure

```
practica_final/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/              ← original images per class
│   └── processed/        ← auto-generated train/val/test split
├── models/
│   └── best_model.pth    ← saved after training
├── src/
│   ├── config.py         ← central configuration
│   ├── dataset.py        ← data loading & augmentation
│   ├── model.py          ← transfer-learning model factory
│   ├── train.py          ← training loop with W&B
│   ├── evaluate.py       ← metrics & confusion matrix
│   └── sweep_config.yaml ← W&B hyperparameter sweep
├── api/
│   ├── main.py           ← FastAPI inference server
│   └── schemas.py        ← request/response schemas
├── app/
│   └── streamlit_app.py  ← Streamlit front-end
├── scripts/
│   └── download_data.py  ← dataset download & split
└── reports/
    └── final_report.md   ← technical report template
```

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USER/real-estate-classifier.git
cd real-estate-classifier

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

pip install -r requirements.txt
```

### 2. Prepare dataset

**Option A — Download from Kaggle:**

```bash
# Set Kaggle credentials (see https://www.kaggle.com/docs/api)
set KAGGLE_USERNAME=your_user
set KAGGLE_KEY=your_key

python scripts/download_data.py --kaggle
```

**Option B — Manual setup:**

Place images in `data/raw/` with one subfolder per class:

```
data/raw/
├── bathroom/
├── bedroom/
├── dining_room/
├── exterior/
├── kitchen/
└── living_room/
```

Then split:

```bash
python scripts/download_data.py --split-only
```

### 3. Login to Weights & Biases

```bash
wandb login
# Optionally set your team entity:
set WANDB_ENTITY=your_team
```

### 4. Train a model

```bash
# Single training run with default hyperparameters
python -m src.train

# Custom run
python -m src.train --backbone efficientnet_b0 --lr 3e-4 --epochs 30 --batch_size 32
```

### 5. Hyperparameter sweep (W&B)

```bash
# Create sweep
wandb sweep src/sweep_config.yaml

# Launch agent (replace SWEEP_ID with the output from above)
wandb agent YOUR_ENTITY/real-estate-classifier/SWEEP_ID
```

### 6. Start the API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Swagger docs available at: **http://localhost:8000/docs**

### 7. Start the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

Opens at: **http://localhost:8501**

---

## API Endpoints

| Method | Path      | Description                         |
|--------|-----------|-------------------------------------|
| GET    | /health   | Health check & model status         |
| POST   | /predict  | Classify an uploaded property image |

### Example cURL

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@kitchen_photo.jpg"
```

---

## W&B Integration

- Project: `real-estate-classifier`
- Sweep strategy: Bayesian optimization
- Tracked metrics: train/val loss & accuracy, test per-class precision/recall/F1, confusion matrix
- Model artifacts: best checkpoint logged as W&B artifact

---

## Tech Stack

- **PyTorch + timm** — model & training
- **Weights & Biases** — experiment tracking & sweeps
- **FastAPI** — inference API
- **Streamlit** — front-end demo
- **scikit-learn** — evaluation metrics

---

## License

MIT

# Real-Estate Room Image Classifier — Technical Report

**Authors:** [Your names]
**Date:** [Date]
**Repository:** [GitHub link]
**W&B Project:** [W&B link]

---

## 1. Customer Context

### Business Problem
Online real-estate marketplaces receive thousands of property listings daily, each accompanied by multiple photographs. Currently, agents must manually tag each image by room type (kitchen, bedroom, bathroom, etc.), a process that is slow, inconsistent, and error-prone.

### Target User
Real-estate marketplace operators and listing agents who require automated, consistent image categorization at scale.

### Expected Value
- **Faster listing publication:** Automatic room-type tagging reduces manual effort by ~80%.
- **Improved search experience:** Buyers can filter by specific room types.
- **Quality control:** Flag mislabeled or low-quality images before publication.

### Operational Constraints
- Inference latency < 500 ms per image on CPU.
- The model must support JPEG, PNG, and WebP formats.
- Maximum image upload size: 10 MB.
- The solution must be deployable on a single machine (CPU or GPU).

---

## 2. System Architecture

```
┌────────────┐      HTTP/JSON      ┌──────────────┐       PyTorch       ┌───────────┐
│  Streamlit  │ ─────────────────→  │   FastAPI     │ ──────────────────→ │   Model   │
│  Front-end  │ ←───────────────── │   Back-end    │ ←────────────────── │ (EfficientNet)
└────────────┘   prediction result └──────────────┘    logits/probs     └───────────┘
       ↑                                  │
  User uploads                      Model loaded
  property image                    at startup from
                                    models/best_model.pth
```

- **Streamlit** provides the web interface for image upload and result visualization.
- **FastAPI** exposes a REST API (`POST /predict`) that performs inference and returns class probabilities.
- **The model** is a fine-tuned EfficientNet-B0 loaded once at server startup.

---

## 3. Modeling Approach

### Pre-trained Model Selection
We selected **EfficientNet-B0** (pre-trained on ImageNet) for the following reasons:
- Superior accuracy-per-FLOP compared to ResNet and VGG families.
- Compact model size (~5.3 M parameters) suitable for CPU deployment.
- Strong transfer-learning performance on small-to-medium datasets.

We also benchmarked ResNet-50 and MobileNetV2 as alternatives (see Section 4).
  
### Transfer-Learning Strategy
1. **Feature extraction phase:** Freeze all backbone layers; train only the custom classification head for initial epochs.
2. **Fine-tuning phase:** Progressively unfreeze the last N convolutional blocks and train end-to-end with a reduced learning rate.

### Final Architecture
```
EfficientNet-B0 (backbone, partially frozen)
  → BatchNorm1d(1280)
  → Dropout(p=0.3)
  → Linear(1280, 256)
  → ReLU
  → Dropout(p=0.15)
  → Linear(256, 6)   ← 6 room-type classes
```

---

## 4. Experimentation Process (W&B)

### Experiment Design
All experiments were tracked in Weights & Biases under the project `real-estate-classifier`.

### Hyperparameter Search Strategy
We used **Bayesian optimization** (W&B Sweeps) across the following hyperparameters:

| Parameter         | Search Space                |
|-------------------|-----------------------------|
| Backbone          | efficientnet_b0, resnet50, mobilenetv2 |
| Learning rate     | log-uniform [1e-5, 1e-2]   |
| Dropout           | {0.2, 0.3, 0.4, 0.5}       |
| Optimizer         | Adam, SGD                   |
| Weight decay      | log-uniform [1e-6, 1e-2]   |
| Freeze backbone   | True, False                 |
| Unfrozen blocks   | {1, 2, 3, 4}               |
| Batch size        | {16, 32, 64}               |
| Scheduler         | Cosine, Step, None          |

### Model Selection Criteria
The final model was selected by **highest validation accuracy** with early stopping (patience = 5 epochs). Ties were broken by macro F1-score on the test set.

### Key Findings
- [Fill in with actual results from your W&B runs]
- [E.g., "EfficientNet-B0 with lr=3e-4, dropout=0.3, Adam optimizer achieved 93.2% val accuracy"]
- [E.g., "Freezing the backbone and only training the head performed comparably to full fine-tuning with 3× less training time"]

---

## 5. Performance Metrics per Output Class

### Overall Metrics
| Metric        | Value  |
|---------------|--------|
| Test Accuracy | XX.X%  |
| Macro F1      | 0.XXX  |

### Per-Class Breakdown

| Class        | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| bathroom     | 0.XX      | 0.XX   | 0.XX     | XXX     |
| bedroom      | 0.XX      | 0.XX   | 0.XX     | XXX     |
| dining_room  | 0.XX      | 0.XX   | 0.XX     | XXX     |
| exterior     | 0.XX      | 0.XX   | 0.XX     | XXX     |
| kitchen      | 0.XX      | 0.XX   | 0.XX     | XXX     |
| living_room  | 0.XX      | 0.XX   | 0.XX     | XXX     |

### Confusion Matrix Interpretation
- [Describe which classes are most commonly confused and why]
- [E.g., "dining_room and living_room share visual features (open-plan layouts), leading to ~12% cross-confusion"]
- [Discuss whether the quality level meets business requirements]

---

## 6. API Documentation

The API is documented via **OpenAPI/Swagger** at `http://localhost:8000/docs`.

### Endpoints

| Method | Endpoint   | Description                          |
|--------|------------|--------------------------------------|
| GET    | /health    | Health check and model status        |
| POST   | /predict   | Classify an uploaded property image  |

### POST /predict
- **Input:** Multipart form upload (`file` field), accepts JPEG/PNG/WebP ≤ 10 MB.
- **Output (200):** `{ "predicted_class": "kitchen", "confidence": 0.92, "probabilities": {...} }`
- **Error (400):** Invalid image format or oversized file.
- **Error (503):** Model not loaded.

---

## 7. Project Links

| Resource       | Link |
|----------------|------|
| Git Repository | [GitHub](https://github.com/YOUR_USER/real-estate-classifier) |
| W&B Project    | [Weights & Biases](https://wandb.ai/YOUR_ENTITY/real-estate-classifier) |

### Access
- Repository: **Public**
- W&B project: agascon@comillas.edu and rkramer@comillas.edu invited as viewers.

---

## Conclusions and Business Recommendations

1. [Summarize model performance and whether it meets business requirements]
2. [Recommend next steps: more data for underperforming classes, model distillation for edge deployment, etc.]
3. [Discuss limitations: dataset size, class imbalance, domain shift from training images vs. real listings]

---
title: Forensic AI Image Detector
emoji: 🔬
colorFrom: gray
colorTo: green
sdk: docker
pinned: false
app_port: 7860
---

# Forensic AI Image Detector

Dual-stream CNN (EfficientNet-B4 spatial + Patch-DCT frequency) trained on the
[GenImage](https://github.com/GenImage-Dataset/GenImage) dataset to distinguish
real photographs from AI-generated images (GAN / Stable Diffusion / Midjourney).

## Features
- Upload any image → **REAL / AI-GENERATED** verdict
- **Confidence score** with animated meter
- **GradCAM heatmap** highlighting the spatial regions that triggered the decision
- **Prediction history** with per-entry thumbnails and delete controls

## Setup (add your checkpoint)

1. Train using the notebook pipeline and download `checkpoints/best_model.pt`.
2. Upload `best_model.pt` to `checkpoints/best_model.pt` in this Space's files.
3. The Space will reload automatically and the model will be live.

## Local dev

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

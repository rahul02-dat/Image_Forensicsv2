import io
import base64
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

import database as db
from model import InferenceEngine
from gradcam import GradCAM, heatmap_overlay

CHECKPOINT = Path("./checkpoints/best_model.pt")
engine: InferenceEngine | None = None
gcam:   GradCAM | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, gcam
    db.init_db()
    if CHECKPOINT.exists():
        engine = InferenceEngine(str(CHECKPOINT))
        gcam   = GradCAM(engine.model)
        print("Inference engine ready.")
    else:
        print(f"WARNING: checkpoint not found at {CHECKPOINT}. /predict will return 503.")
    yield
    if gcam:
        gcam.remove_hooks()


app = FastAPI(title="Forensic AI Image Detector", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


def _pil_to_thumbnail_b64(pil_img: Image.Image, size: int = 160) -> str:
    thumb = pil_img.copy().convert("RGB")
    thumb.thumbnail((size, size))
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


@app.get("/", response_class=HTMLResponse)
async def index():
    return Path("static/index.html").read_text()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if engine is None:
        raise HTTPException(503, "Model checkpoint not loaded.")

    allowed = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
    if file.content_type not in allowed:
        raise HTTPException(415, f"Unsupported file type: {file.content_type}")

    raw = await file.read()
    pil_img = Image.open(io.BytesIO(raw)).convert("RGB")

    tensor = engine.preprocess(pil_img)
    prob   = engine.predict(tensor)                  # P(AI-generated)

    label      = "AI-GENERATED" if prob > 0.5 else "REAL"
    confidence = prob * 100 if prob > 0.5 else (1 - prob) * 100

    # GradCAM (re-run with grad enabled)
    cam_tensor  = engine.preprocess(pil_img)
    cam         = gcam.generate(cam_tensor)
    gradcam_b64 = heatmap_overlay(pil_img, cam)

    thumb_b64 = _pil_to_thumbnail_b64(pil_img)

    pred_id = db.save_prediction(
        filename=file.filename or "upload",
        prediction=label,
        confidence=round(confidence, 2),
        gradcam_b64=gradcam_b64,
        thumbnail_b64=thumb_b64,
    )

    return {
        "id":           pred_id,
        "label":        label,
        "confidence":   round(confidence, 2),
        "raw_prob":     round(prob, 4),
        "gradcam_b64":  gradcam_b64,
        "thumbnail_b64": thumb_b64,
    }


@app.get("/history")
async def history(limit: int = 50):
    return db.get_history(limit)


@app.get("/history/{pred_id}")
async def history_detail(pred_id: int):
    row = db.get_prediction(pred_id)
    if not row:
        raise HTTPException(404, "Prediction not found.")
    return row


@app.delete("/history/{pred_id}")
async def delete_entry(pred_id: int):
    if not db.delete_prediction(pred_id):
        raise HTTPException(404, "Prediction not found.")
    return {"deleted": pred_id}


@app.delete("/history")
async def clear_all():
    db.clear_history()
    return {"cleared": True}


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": engine is not None}

import sys
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from contextlib import asynccontextmanager

# ==================================================
# PATHS (Đã sửa thành đường dẫn tương đối động)
# ==================================================

# 1. Lấy đường dẫn của thư mục chứa file hiện tại (thư mục src/)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Dựa vào cấu trúc /src/saved_models/ của bạn:
SAVED_MODELS_DIR = os.path.join(CURRENT_DIR, "src", "saved_models")

# Override MODEL_DIR trước khi ModelStore import config
import config
config.MODEL_DIR = SAVED_MODELS_DIR
config.SEGMENT_MODEL_DIR = os.path.join(SAVED_MODELS_DIR, "VnCoreNLP-1.2.jar")

from src.inference.final_absa_pipeline_v2 import analyze_sentence
from models.loader import ModelStore
from models.predictor import predict_emotion


# ==================================================
# LIFESPAN
# ==================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.store = ModelStore()
    print("✓ Tất cả model đã sẵn sàng")
    yield
    if hasattr(app.state.store, 'rdrsegmenter') and app.state.store.rdrsegmenter:
        app.state.store.rdrsegmenter.close()


# ==================================================
# APP
# ==================================================

app = FastAPI(title="NLP API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================================================
# SCHEMAS
# ==================================================

class TextRequest(BaseModel):
    sentence: str

class AspectResult(BaseModel):
    aspect: str
    sentiment: str
    aspect_score: float
    sentiment_confidence: float

class ABSAResponse(BaseModel):
    sentence: str
    results: List[AspectResult]
    total_aspects: int

class EmotionResponse(BaseModel):
    sentence: str
    label: str
    probs: dict

class PredictRequest(BaseModel):
    text: str

# ==================================================
# ROUTES
# ==================================================

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/aspect-predict", response_model=ABSAResponse)
def aspect_predict(req: TextRequest):
    sentence = req.sentence.strip()
    if not sentence:
        raise HTTPException(status_code=400, detail="Câu không được để trống.")
    if len(sentence) > 512:
        raise HTTPException(status_code=400, detail="Câu quá dài (tối đa 512 ký tự).")
    results = analyze_sentence(sentence)
    return ABSAResponse(sentence=sentence, results=results, total_aspects=len(results))


@app.post("/emotion-predict")
def predict_emotion_endpoint(req: PredictRequest):
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="text không được để trống")
    return predict_emotion(req.text, app.state.store)
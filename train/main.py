from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from models.loader import ModelStore
from models.predictor import predict_emotion


# ── lifespan: load model 1 lần khi startup, giải phóng khi shutdown ──
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.store = ModelStore()
    print("✓ Tất cả model đã sẵn sàng")
    yield
    # Giải phóng JVM của VnCoreNLP khi server tắt
    app.state.store.rdrsegmenter.close()

app = FastAPI(lifespan=lifespan)


# ── schema ────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str


# ── endpoints ─────────────────────────────────────────────────────────
@app.post("/predict-emotion")
def predict_emotion_endpoint(req: PredictRequest):
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="text không được để trống")
    return predict_emotion(req.text, app.state.store)

@app.get("/health")
def health():
    return {"status": "ok"}

import os
import torch
import vncorenlp
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import (
    SEGMENT_MODEL_DIR,
    MODEL_DIR,
    MODEL_SAVE,
    MODEL_NAME,
    NUM_LABELS,
)
from common import get_device


class ModelStore:
    """
    Nơi lưu trữ tất cả model và các thành phần dùng chung (segmenter, tokenizer).
    Load một lần duy nhất khi server khởi động, tái sử dụng cho mọi request.

    Attributes:
        device          : thiết bị tính toán (cuda / mps / cpu)
        rdrsegmenter    : VnCoreNLP word segmenter dùng chung cho cả 2 model
        emotion_tokenizer: tokenizer của model cảm xúc
        emotion_model   : model phân loại cảm xúc đã load weights
        aspect_tokenizer : tokenizer của model aspect
        aspect_model    : model phân loại aspect đã load weights
    """

    def __init__(self):
        self.device = get_device()

        # ── segmenter dùng chung ──────────────────────────────────
        # VnCoreNLP chỉ cần khởi tạo một lần, tốn ~2GB heap
        self.rdrsegmenter = vncorenlp.VnCoreNLP(
            SEGMENT_MODEL_DIR,
            annotators="wseg",
            max_heap_size="-Xmx2g"
        )

        # ── emotion model ─────────────────────────────────────────
        self.emotion_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.emotion_model     = self._load_model(
            model_name=MODEL_NAME,
            num_labels=NUM_LABELS,
            weights_path=os.path.join(MODEL_DIR, MODEL_SAVE),
        )

    def _load_model(
        self,
        model_name: str,
        num_labels: int,
        weights_path: str,
    ) -> AutoModelForSequenceClassification:
        """
        Load kiến trúc model từ HuggingFace rồi nạp weights đã train.

        Args:
            model_name   : tên model trên HuggingFace Hub (hoặc đường dẫn local)
            num_labels   : số nhãn đầu ra của classification head
            weights_path : đường dẫn tới file .pt chứa state_dict

        Returns:
            Model ở chế độ eval, đã chuyển sang đúng device
        """
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
        )
        model.load_state_dict(
            torch.load(weights_path, map_location=self.device)
        )
        model.to(self.device)
        model.eval()
        return model


# Singleton — import ở bất kỳ đâu cũng dùng chung một instance
store = ModelStore()
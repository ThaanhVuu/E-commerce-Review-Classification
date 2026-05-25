import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import *
from models.loader import ModelStore


def _infer(
    text: str,
    store: ModelStore,
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    label_names: list[str],
) -> dict:
    """
    Hàm inference nội bộ dùng chung cho mọi loại model classification.

    Quy trình:
      1. Tách từ bằng VnCoreNLP (dùng segmenter chung từ store)
      2. Encode chuỗi đã tách từ bằng tokenizer tương ứng
      3. Forward pass (no_grad) → argmax logits → nhãn
      4. Softmax → xác suất từng nhãn

    Args:
        text        : câu văn bản tiếng Việt chưa tách từ
        store       : ModelStore chứa device và segmenter
        tokenizer   : tokenizer của model cần dùng
        model       : model classification cần dùng
        label_names : danh sách tên nhãn tương ứng với index

    Returns:
        dict gồm:
          - "label" : nhãn dự đoán (str)
          - "probs" : {tên_nhãn: xác_suất} cho tất cả nhãn
    """
    segmented = " ".join(store.rdrsegmenter.tokenize(text)[0])

    enc = tokenizer(
        text=segmented,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids      = enc["input_ids"].to(store.device)
    attention_mask = enc["attention_mask"].to(store.device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    pred_idx = logits.argmax(dim=-1).item()
    probs    = torch.softmax(logits, dim=-1).squeeze().tolist()

    return {
        "label": label_names[pred_idx],
        "probs": {name: round(p, 4) for name, p in zip(label_names, probs)},
    }


def predict_emotion(text: str, store: ModelStore) -> dict:
    """
    Dự đoán cảm xúc (emotion) cho một câu tiếng Việt.

    Args:
        text  : câu văn bản đầu vào
        store : ModelStore đã load sẵn emotion_model và emotion_tokenizer

    Returns:
        dict {"label": str, "probs": {str: float}}
    """
    return _infer(
        text=text,
        store=store,
        tokenizer=store.emotion_tokenizer,
        model=store.emotion_model,
        label_names=LABEL_NAMES,
    )


# def predict_aspect(text: str, store: ModelStore) -> dict:
#     """
#     Dự đoán aspect cho một câu tiếng Việt.
#
#     Args:
#         text  : câu văn bản đầu vào
#         store : ModelStore đã load sẵn aspect_model và aspect_tokenizer
#
#     Returns:
#         dict {"label": str, "probs": {str: float}}
#     """
#     return _infer(
#         text=text,
#         store=store,
#         tokenizer=store.aspect_tokenizer,
#         model=store.aspect_model,
#         label_names=ASPECT_LABEL_NAMES,
#     )
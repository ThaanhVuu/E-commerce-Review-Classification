import torch
import pandas as pd
import vncorenlp
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import *

from common import get_device, tokenize_df

device = get_device()
print(f"Thiết bị đang sử dụng: {device}")

df = pd.read_csv(DATA_DIR)
df = df[[TEXT_COL, LABEL_COL]].dropna()
df[TEXT_COL] = df[TEXT_COL].str.strip()
df = df[df[TEXT_COL] != ""]

train_val_df, test_df = train_test_split(
    df, test_size=TEST_SIZE, random_state=42, stratify=df[LABEL_COL]
)
print(f"Tổng mẫu test: {len(test_df)}")

rdrsegmenter = vncorenlp.VnCoreNLP(SEGMENT_MODEL_DIR, annotators="wseg", max_heap_size="-Xmx2g")
tokenizer    = AutoTokenizer.from_pretrained(MODEL_NAME)
model        = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
model.load_state_dict(torch.load('./saved_models/emotion-model.pt', map_location=device))
model.to(device)
model.eval()

print("Đang tokenize test...")
test_loader = tokenize_df(test_df, rdrsegmenter, tokenizer, shuffle=False)

all_preds, all_labels = [], []

with torch.no_grad():
    for input_ids, attention_mask, label in test_loader:
        input_ids      = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        label          = label.to(device)

        output = model(input_ids=input_ids, attention_mask=attention_mask)
        preds  = output.logits.argmax(dim=-1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(label.cpu().numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

print("\n========== KẾT QUẢ TEST ==========")
print(f"Tổng mẫu test: {len(all_labels)}")
print(f"Số đúng:       {(all_preds == all_labels).sum()}")
print(f"Accuracy:      {(all_preds == all_labels).mean():.4f}")

print("\n---------- Classification Report ----------")
print(classification_report(
    all_labels, all_preds, target_names=LABEL_NAMES, digits=4
))

print("---------- Confusion Matrix ----------")
cm = confusion_matrix(all_labels, all_preds)
print(f"{'':15}", end="")
for name in LABEL_NAMES:
    print(f"{name:12}", end="")
print()
for i, row in enumerate(cm):
    print(f"{LABEL_NAMES[i]:15}", end="")
    for val in row:
        print(f"{val:<12}", end="")
    print()

from models.loader import store
from models.predictor import predict_emotion

# Dự đoán cảm xúc
# result = predict_emotion("Áo đẹp nhưng giao hàng chậm", store)
# print(result)
# {"label": "negative", "probs": {"positive": 0.12, "neutral": 0.23, "negative": 0.65}}

# Dự đoán aspect
# result = predict_aspect("Áo đẹp nhưng giao hàng chậm", store)
# print(result)
# # {"label": "giao_hang", "probs": {...}}
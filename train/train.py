"""
check gpu/cpu
"""
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import torch
import pandas as pd
import vncorenlp
import os

from sympy.parsing.sympy_parser import null

from config import *
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, PhobertTokenizer

from common import get_device, tokenize_df
from utils.download_wseg_vncorenlp import download_vncorenlp

device = get_device()
print(f"Thiết bị đang sử dụng: {device}")

"""
load data
"""
df = pd.read_csv(DATA_DIR)
print("=======HEAD=======")
print(df.head())
print(f"\nTổng: {len(df)} mẫu")
print(f"Phân phối nhãn:\n{df[LABEL_COL].value_counts()}")

df = df[[TEXT_COL, LABEL_COL]].dropna()
df[TEXT_COL] = df[TEXT_COL].str.strip()
df = df[df[TEXT_COL] != ""]

train_val_df, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=42,
    stratify=df[LABEL_COL]
)

train_df, val_df = train_test_split(
    train_val_df,
    test_size=VAL_SIZE/ (1 - TEST_SIZE),
    random_state=42,
    stratify=train_val_df[LABEL_COL]
)
print(f"\nSL Train: {len(train_df)} | SL Val: {len(val_df)} | SL Test: {len(test_df)}")

"""
load model
"""
if not os.path.exists(SEGMENT_MODEL_DIR):
    print("\n Thiếu wseg của vncorenlp")
    download_vncorenlp(save_dir=MODEL_DIR)

rdrsegmenter = vncorenlp.VnCoreNLP(SEGMENT_MODEL_DIR ,annotators="wseg", max_heap_size="-Xmx2g")
# rdrsegmenter = null

tokenizer: PhobertTokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)

model.to(device)
model.eval()

"""
data loader
"""
train_loader = tokenize_df(train_df, rdrsegmenter, tokenizer, shuffle=True)
val_loader   = tokenize_df(val_df,   rdrsegmenter, tokenizer, shuffle=False)
test_loader  = tokenize_df(test_df,  rdrsegmenter, tokenizer, shuffle=False)

"""
training
"""
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.array([0, 1, 2]),
    y=train_df[LABEL_COL].values
)
print(f"Class weights: {class_weights}")
criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float).to(device))


optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01 )

total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=WARMUP_STEPS,
    num_training_steps=total_steps
)

PATIENCE = 2
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(EPOCHS):
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0

    for i, (input_ids, attention_mask, label) in enumerate(train_loader):
        input_ids      = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        label          = label.to(device)

        optimizer.zero_grad()
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        loss   = criterion(output.logits, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        train_loss    += loss.item()
        train_correct += (output.logits.argmax(dim=-1) == label).sum().item()
        train_total   += label.size(0)

        if (i + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS} | Batch {i+1}/{len(train_loader)} "
                  f"| Loss: {loss.item():.4f} "
                  f"| Acc: {train_correct/train_total:.4f}")

    train_acc  = train_correct / train_total
    train_loss = train_loss / len(train_loader)

    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0

    with torch.no_grad():
        for input_ids, attention_mask, label in val_loader:
            input_ids      = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            label          = label.to(device)

            output = model(input_ids=input_ids, attention_mask=attention_mask)
            loss   = criterion(output.logits, label)

            val_loss    += loss.item()
            val_correct += (output.logits.argmax(dim=-1) == label).sum().item()
            val_total   += label.size(0)

    val_acc  = val_correct / val_total
    val_loss = val_loss / len(val_loader)

    print(f"Epoch {epoch+1}/{EPOCHS} "
          f"| Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} "
          f"| Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, MODEL_SAVE))
        print(f"  ✓ Lưu model tốt nhất (val_loss={val_loss:.4f})")
    else:
        patience_counter += 1
        print(f"  ✗ Không cải thiện ({patience_counter}/{PATIENCE})")
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping tại epoch {epoch + 1}")
            break
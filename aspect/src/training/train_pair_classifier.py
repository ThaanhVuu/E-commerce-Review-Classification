import os
import torch
import pandas as pd
import numpy as np

from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.optim import AdamW

from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from src.models.aspect_pair_dataset import (
    AspectPairDataset
)

from src.models.aspect_pair_classifier import (
    AspectPairClassifier
)

# =========================
# CONFIG
# =========================

MODEL_NAME = "vinai/phobert-base"

MAX_LENGTH = 128

TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 16

EPOCHS = 5

LEARNING_RATE = 2e-5

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

SAVE_PATH = (
    "saved_models/pair_classifier.pt"
)

os.makedirs(
    "saved_models",
    exist_ok=True
)

print(f"\nDEVICE: {DEVICE}")

# =========================
# LOAD DATA
# =========================

train_df = pd.read_csv(
    "data/processed/pair_train.csv"
)

valid_df = pd.read_csv(
    "data/processed/pair_valid.csv"
)

print("\nTrain size:", len(train_df))
print("Valid size:", len(valid_df))

# =========================
# TOKENIZER
# =========================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

# =========================
# DATASET
# =========================

train_dataset = (
    AspectPairDataset(
        dataframe=train_df,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH
    )
)

valid_dataset = (
    AspectPairDataset(
        dataframe=valid_df,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH
    )
)

# =========================
# DATALOADER
# =========================

train_loader = DataLoader(
    train_dataset,
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=True
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=VALID_BATCH_SIZE,
    shuffle=False
)

# =========================
# MODEL
# =========================

model = AspectPairClassifier()

model.to(DEVICE)

print("\nModel loaded.")

# =========================
# LOSS
# =========================

criterion = (
    torch.nn.BCEWithLogitsLoss()
)

# =========================
# OPTIMIZER
# =========================

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)

# =========================
# SCHEDULER
# =========================

total_steps = (
    len(train_loader) * EPOCHS
)

scheduler = (
    get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
)

# =========================
# MIXED PRECISION
# =========================

scaler = torch.cuda.amp.GradScaler()

# =========================
# TRAIN FUNCTION
# =========================

def train_epoch():

    model.train()

    total_loss = 0

    all_preds = []
    all_labels = []

    progress_bar = tqdm(
        train_loader,
        desc="Training"
    )

    for batch in progress_bar:

        input_ids = batch[
            "input_ids"
        ].to(DEVICE)

        attention_mask = batch[
            "attention_mask"
        ].to(DEVICE)

        labels = batch[
            "label"
        ].to(DEVICE)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            loss = criterion(
                logits,
                labels
            )

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        scheduler.step()

        total_loss += loss.item()

        probs = torch.sigmoid(
            logits
        )

        preds = (
            probs > 0.5
        ).float()

        all_preds.extend(
            preds.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

        progress_bar.set_postfix(
            loss=loss.item()
        )

    acc = accuracy_score(
        all_labels,
        all_preds
    )

    precision = precision_score(
        all_labels,
        all_preds
    )

    recall = recall_score(
        all_labels,
        all_preds
    )

    f1 = f1_score(
        all_labels,
        all_preds
    )

    return (
        total_loss / len(train_loader),
        acc,
        precision,
        recall,
        f1
    )

# =========================
# VALID FUNCTION
# =========================

def eval_model():

    model.eval()

    total_loss = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for batch in tqdm(
            valid_loader,
            desc="Validation"
        ):

            input_ids = batch[
                "input_ids"
            ].to(DEVICE)

            attention_mask = batch[
                "attention_mask"
            ].to(DEVICE)

            labels = batch[
                "label"
            ].to(DEVICE)

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            loss = criterion(
                logits,
                labels
            )

            total_loss += loss.item()

            probs = torch.sigmoid(
                logits
            )

            preds = (
                probs > 0.5
            ).float()

            all_preds.extend(
                preds.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    acc = accuracy_score(
        all_labels,
        all_preds
    )

    precision = precision_score(
        all_labels,
        all_preds
    )

    recall = recall_score(
        all_labels,
        all_preds
    )

    f1 = f1_score(
        all_labels,
        all_preds
    )

    return (
        total_loss / len(valid_loader),
        acc,
        precision,
        recall,
        f1
    )

# =========================
# TRAIN LOOP
# =========================

best_f1 = 0

print("\nSTART TRAINING...\n")

for epoch in range(EPOCHS):

    print("=" * 50)

    print(f"EPOCH {epoch+1}/{EPOCHS}")

    print("=" * 50)

    train_loss, train_acc, train_p, train_r, train_f1 = (
        train_epoch()
    )

    valid_loss, valid_acc, valid_p, valid_r, valid_f1 = (
        eval_model()
    )

    print(f"\nTrain Loss : {train_loss:.4f}")
    print(f"Train Acc  : {train_acc:.4f}")
    print(f"Train Prec : {train_p:.4f}")
    print(f"Train Rec  : {train_r:.4f}")
    print(f"Train F1   : {train_f1:.4f}")

    print(f"\nValid Loss : {valid_loss:.4f}")
    print(f"Valid Acc  : {valid_acc:.4f}")
    print(f"Valid Prec : {valid_p:.4f}")
    print(f"Valid Rec  : {valid_r:.4f}")
    print(f"Valid F1   : {valid_f1:.4f}")

    # =====================
    # SAVE BEST MODEL
    # =====================

    if valid_f1 > best_f1:

        best_f1 = valid_f1

        torch.save(
            model.state_dict(),
            SAVE_PATH
        )

        print("\nBEST MODEL SAVED!")

print("\nTRAINING FINISHED!")
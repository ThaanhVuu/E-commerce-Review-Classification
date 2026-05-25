import os
import json
import random
import pandas as pd

from sklearn.model_selection import (
    train_test_split
)

from src.configs.aspect_descriptions import (
    ASPECT_DESCRIPTIONS
)

random.seed(42)

# =========================
# LOAD CLEAN DATA
# =========================

df = pd.read_csv(
    "data/processed/cleaned_data.csv"
)

print("Original shape:")
print(df.shape)

# =========================
# GET ALL ASPECTS
# =========================

all_aspects = list(
    ASPECT_DESCRIPTIONS.keys()
)

print("\nAll aspects:")
print(all_aspects)

# =========================
# GROUP BY SENTENCE
# =========================

grouped = df.groupby(
    "sentence"
)["aspect"].apply(list).reset_index()

print("\nGrouped shape:")
print(grouped.shape)

print("\nSample grouped:")
print(grouped.head())

# =========================
# BUILD PAIRS
# =========================

pairs = []

for _, row in grouped.iterrows():

    sentence = row["sentence"]

    positive_aspects = row["aspect"]

    # =====================
    # POSITIVE PAIRS
    # =====================

    for aspect in positive_aspects:

        description = (
            ASPECT_DESCRIPTIONS[
                aspect
            ]
        )

        input_text = (
            sentence
            + " [SEP] "
            + description
        )

        pairs.append({
            "sentence": sentence,
            "aspect": aspect,
            "description": description,
            "input_text": input_text,
            "label": 1
        })

    # =====================
    # NEGATIVE PAIRS
    # =====================

    negative_aspects = [
        a for a in all_aspects
        if a not in positive_aspects
    ]

    # sample negatives
    sampled_negatives = random.sample(
        negative_aspects,
        min(
            len(positive_aspects),
            len(negative_aspects)
        )
    )

    for aspect in sampled_negatives:

        description = (
            ASPECT_DESCRIPTIONS[
                aspect
            ]
        )

        input_text = (
            sentence
            + " [SEP] "
            + description
        )

        pairs.append({
            "sentence": sentence,
            "aspect": aspect,
            "description": description,
            "input_text": input_text,
            "label": 0
        })

# =========================
# FINAL DATAFRAME
# =========================

pair_df = pd.DataFrame(
    pairs
)

print("\nPair dataset shape:")
print(pair_df.shape)

print("\nLabel distribution:")
print(
    pair_df["label"]
    .value_counts()
)

print("\nSample pairs:")
print(
    pair_df.head(10)
)

# =========================
# TRAIN VALID TEST SPLIT
# =========================

train_df, temp_df = train_test_split(
    pair_df,
    test_size=0.2,
    random_state=42,
    stratify=pair_df["label"]
)

valid_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=42,
    stratify=temp_df["label"]
)

# =========================
# SAVE
# =========================

os.makedirs(
    "data/processed",
    exist_ok=True
)

train_df.to_csv(
    "data/processed/pair_train.csv",
    index=False
)

valid_df.to_csv(
    "data/processed/pair_valid.csv",
    index=False
)

test_df.to_csv(
    "data/processed/pair_test.csv",
    index=False
)

print("\nSaved pair datasets.")

print("\nTrain:", len(train_df))
print("Valid:", len(valid_df))
print("Test :", len(test_df))
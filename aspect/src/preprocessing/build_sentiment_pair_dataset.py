import os
import pandas as pd

from sklearn.model_selection import (
    train_test_split
)

from src.configs.aspect_descriptions import (
    ASPECT_DESCRIPTIONS
)

# =========================
# LOAD DATA
# =========================

df = pd.read_csv(
    "data/processed/cleaned_data.csv"
)

print("Original shape:")
print(df.shape)

# =========================
# BUILD INPUT TEXT
# =========================

input_texts = []

for _, row in df.iterrows():

    sentence = row["sentence"]

    aspect = row["aspect"]

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

    input_texts.append(
        input_text
    )

df["input_text"] = input_texts

# =========================
# LABEL MAP
# =========================

label2id = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}

df["label_id"] = (
    df["label_text"]
    .map(label2id)
)

# =========================
# SHOW SAMPLE
# =========================

print("\nSample:")
print(
    df[
        [
            "input_text",
            "label_text",
            "label_id"
        ]
    ].head()
)

# =========================
# SPLIT
# =========================

train_df, temp_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label_id"]
)

valid_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=42,
    stratify=temp_df["label_id"]
)

# =========================
# SAVE
# =========================

os.makedirs(
    "data/processed",
    exist_ok=True
)

train_df.to_csv(
    "data/processed/sentiment_pair_train.csv",
    index=False
)

valid_df.to_csv(
    "data/processed/sentiment_pair_valid.csv",
    index=False
)

test_df.to_csv(
    "data/processed/sentiment_pair_test.csv",
    index=False
)

print("\nSaved sentiment pair datasets.")

print("\nTrain:", len(train_df))
print("Valid:", len(valid_df))
print("Test :", len(test_df))
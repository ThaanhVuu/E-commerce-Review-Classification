import pandas as pd

# =========================
# LOAD DATA
# =========================

df = pd.read_excel("data/raw/dataAspect.xlsx")

print("Original shape:", df.shape)

# =========================
# DROP NULL LABEL
# =========================

df = df.dropna(subset=["label"])

# =========================
# CONVERT LABEL
# =========================

df["label"] = df["label"].astype(int)

# =========================
# NORMALIZE ASPECT
# =========================

ASPECT_MAPPING = {
    "gia-ca": "gia_ca",
    "gia-cong": "gia_cong",
    "chat_liieu": "chat_lieu"
}

df["aspect"] = df["aspect"].replace(ASPECT_MAPPING)

# =========================
# LOWERCASE TEXT
# =========================

df["sentence"] = df["sentence"].str.lower()

df["aspect_token"] = df["aspect_token"].str.lower()

# =========================
# REMOVE DUPLICATES
# =========================

df = df.drop_duplicates()

# =========================
# LABEL MAP
# =========================

LABEL_MAP = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

df["label_text"] = df["label"].map(LABEL_MAP)

# =========================
# SHOW RESULT
# =========================

print("\nCleaned shape:", df.shape)

print("\nUnique aspects:")
print(df["aspect"].unique())

print("\nLabel distribution:")
print(df["label_text"].value_counts())

# =========================
# SAVE CLEAN DATA
# =========================

output_path = "data/processed/cleaned_data.csv"

df.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"\nSaved cleaned data to: {output_path}")
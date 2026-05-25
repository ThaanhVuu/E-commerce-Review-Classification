import torch

from torch.utils.data import Dataset


class AspectPairDataset(Dataset):

    def __init__(
        self,
        dataframe,
        tokenizer,
        max_length=128
    ):

        self.dataframe = dataframe

        self.tokenizer = tokenizer

        self.max_length = max_length

    def __len__(self):

        return len(self.dataframe)

    def __getitem__(self, idx):

        row = self.dataframe.iloc[idx]

        input_text = row["input_text"]

        label = row["label"]

        encoding = self.tokenizer(
            input_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {

            "input_ids":
                encoding[
                    "input_ids"
                ].flatten(),

            "attention_mask":
                encoding[
                    "attention_mask"
                ].flatten(),

            "label":
                torch.tensor(
                    label,
                    dtype=torch.float
                )
        }


# =========================
# TEST
# =========================

if __name__ == "__main__":

    import pandas as pd

    from transformers import (
        AutoTokenizer
    )

    print("=" * 50)
    print("TESTING ASPECT PAIR DATASET")
    print("=" * 50)

    df = pd.read_csv(
        "data/processed/pair_train.csv"
    )

    print("\nDataset shape:")
    print(df.shape)

    tokenizer = (
        AutoTokenizer.from_pretrained(
            "vinai/phobert-base"
        )
    )

    dataset = AspectPairDataset(
        dataframe=df,
        tokenizer=tokenizer
    )

    print("\nDataset size:")
    print(len(dataset))

    sample = dataset[0]

    print("\nSample keys:")
    print(sample.keys())

    print("\nInput shape:")
    print(
        sample["input_ids"].shape
    )

    print("\nAttention shape:")
    print(
        sample[
            "attention_mask"
        ].shape
    )

    print("\nLabel:")
    print(sample["label"])

    print("\nDecoded:")
    print(
        tokenizer.decode(
            sample["input_ids"]
        )
    )

    print("\nSUCCESS!")
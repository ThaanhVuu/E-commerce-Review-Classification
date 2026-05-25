import torch
import torch.nn as nn

from transformers import AutoModel


class PhoBERTABSA(nn.Module):

    def __init__(
        self,
        model_name="vinai/phobert-base",
        num_classes=3,
        dropout_rate=0.3
    ):

        super().__init__()

        # =========================
        # LOAD PHOBERT
        # =========================

        self.phobert = AutoModel.from_pretrained(
            model_name
        )

        hidden_size = self.phobert.config.hidden_size

        # =========================
        # DROPOUT
        # =========================

        self.dropout = nn.Dropout(
            dropout_rate
        )

        # =========================
        # CLASSIFIER
        # =========================

        self.classifier = nn.Linear(
            hidden_size,
            num_classes
        )

    def forward(
        self,
        input_ids,
        attention_mask
    ):

        outputs = self.phobert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # CLS token embedding
        cls_output = outputs.last_hidden_state[:, 0]

        # Dropout
        cls_output = self.dropout(
            cls_output
        )

        # Classification
        logits = self.classifier(
            cls_output
        )

        return logits


if __name__ == "__main__":

    from transformers import AutoTokenizer

    print("=" * 50)
    print("TESTING PHOBERT ABSA MODEL")
    print("=" * 50)

    # =========================
    # DEVICE
    # =========================

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print(f"\nDevice: {device}")

    # =========================
    # LOAD TOKENIZER
    # =========================

    tokenizer = AutoTokenizer.from_pretrained(
        "vinai/phobert-base"
    )

    # =========================
    # LOAD MODEL
    # =========================

    model = PhoBERTABSA()

    model.to(device)

    print("\nModel loaded successfully.")

    # =========================
    # SAMPLE TEXT
    # =========================

    text = (
        "quần rộng nhưng chăm sóc khách hàng rất tệ "
        "[SEP] cham_soc_khach_hang"
    )

    # =========================
    # TOKENIZE
    # =========================

    encoding = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)

    attention_mask = encoding[
        "attention_mask"
    ].to(device)

    # =========================
    # FORWARD PASS
    # =========================

    with torch.no_grad():

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

    print("\nLogits shape:")
    print(logits.shape)

    print("\nLogits:")
    print(logits)

    # =========================
    # PREDICTION
    # =========================

    prediction = torch.argmax(
        logits,
        dim=1
    )

    print("\nPrediction:")
    print(prediction)

    print("\nSUCCESS!")
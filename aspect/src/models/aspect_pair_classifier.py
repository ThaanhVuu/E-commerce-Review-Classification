import torch
import torch.nn as nn

from transformers import (
    AutoModel
)


class AspectPairClassifier(
    nn.Module
):

    def __init__(
        self,
        model_name="vinai/phobert-base",
        dropout=0.3
    ):

        super().__init__()

        # =====================
        # PHOBERT
        # =====================

        self.phobert = (
            AutoModel.from_pretrained(
                model_name
            )
        )

        hidden_size = (
            self.phobert.config.hidden_size
        )

        # =====================
        # CLASSIFIER
        # =====================

        self.dropout = (
            nn.Dropout(dropout)
        )

        self.classifier = (
            nn.Linear(
                hidden_size,
                1
            )
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

        cls_output = (
            outputs.last_hidden_state[
                :, 0
            ]
        )

        cls_output = (
            self.dropout(
                cls_output
            )
        )

        logits = (
            self.classifier(
                cls_output
            )
        )

        return logits.squeeze(-1)


# =========================
# TEST
# =========================

if __name__ == "__main__":

    from transformers import (
        AutoTokenizer
    )

    print("=" * 50)
    print("TESTING PAIR CLASSIFIER")
    print("=" * 50)

    DEVICE = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            "vinai/phobert-base"
        )
    )

    model = (
        AspectPairClassifier()
    )

    model.to(DEVICE)

    text = (
        "quần rộng nhưng shop rep chậm "
        "[SEP] "
        "size kích cỡ rộng chật"
    )

    encoding = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    input_ids = encoding[
        "input_ids"
    ].to(DEVICE)

    attention_mask = encoding[
        "attention_mask"
    ].to(DEVICE)

    with torch.no_grad():

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        probs = torch.sigmoid(
            logits
        )

    print("\nLogits:")
    print(logits)

    print("\nProbabilities:")
    print(probs)

    print("\nSUCCESS!")
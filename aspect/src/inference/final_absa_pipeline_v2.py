import torch

from transformers import AutoTokenizer

from src.models.aspect_pair_classifier import (
    AspectPairClassifier
)

from src.models.phobert_absa import (
    PhoBERTABSA
)

from src.configs.aspect_descriptions import (
    ASPECT_DESCRIPTIONS
)

# ==================================================
# CONFIG
# ==================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

MODEL_NAME = "vinai/phobert-base"

MAX_LENGTH = 128

ASPECT_THRESHOLD_DEFAULT = 0.3

SENTIMENT_THRESHOLD = 0.5

# ==================================================
# ASPECT THRESHOLDS
# ==================================================

ASPECT_THRESHOLDS = {

    "cham_soc_khach_hang": 0.35,

    "chat_lieu": 0.50,

    "chat_luong": 0.75,

    "do_ben": 0.55,

    "doi_tra": 0.50,

    "dong_goi": 0.50,

    "gia_ca": 0.50,

    "gia_cong": 0.60,

    "giao_hang": 0.50,

    "kich_co": 0.30,

    "kieu_dang": 0.55,

    "mau_sac": 0.65,

    "uy_tin_cua_hang": 0.60
}

# ==================================================
# LABEL MAP
# ==================================================

id2label = {

    0: "negative",

    1: "neutral",

    2: "positive"
}

# ==================================================
# LOAD TOKENIZER
# ==================================================

print("=" * 60)
print("LOADING TOKENIZER")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print("Tokenizer loaded successfully")

# ==================================================
# LOAD ASPECT MODEL
# ==================================================

print("\n" + "=" * 60)
print("LOADING ASPECT MODEL")
print("=" * 60)

aspect_model = AspectPairClassifier()

aspect_model.load_state_dict(
    torch.load(
        "/home/thanhvu/Projects/E-commerce-Review-Classification/aspect/src/saved_models/pair_classifier.pt",
        map_location=DEVICE,
        weights_only=True
    )
)

aspect_model.to(DEVICE)

aspect_model.eval()

print("Aspect Pair Classifier loaded")

# ==================================================
# LOAD SENTIMENT MODEL
# ==================================================

print("\n" + "=" * 60)
print("LOADING SENTIMENT MODEL")
print("=" * 60)

sentiment_model = PhoBERTABSA()

sentiment_model.load_state_dict(
    torch.load(
        "/home/thanhvu/Projects/E-commerce-Review-Classification/aspect/src/saved_models/sentiment_pair_model.pt",
        map_location=DEVICE,
        weights_only=True
    )
)

sentiment_model.to(DEVICE)

sentiment_model.eval()

print("Sentiment model loaded")

# ==================================================
# DETECT ASPECTS
# ==================================================

def detect_aspects(sentence):

    detected_aspects = []

    print("\nALL ASPECT SCORES:")
    print("=" * 60)

    for aspect, description in (
        ASPECT_DESCRIPTIONS.items()
    ):

        threshold = ASPECT_THRESHOLDS.get(
            aspect,
            ASPECT_THRESHOLD_DEFAULT
        )

        input_text = (
            sentence
            + " [SEP] "
            + description
        )

        encoding = tokenizer(
            input_text,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        input_ids = encoding[
            "input_ids"
        ].to(DEVICE)

        attention_mask = encoding[
            "attention_mask"
        ].to(DEVICE)

        with torch.no_grad():

            logits = aspect_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            probability = torch.sigmoid(
                logits
            ).item()

        print(
            f"{aspect:<25}"
            f"{probability:.4f}"
            f"  (threshold={threshold})"
        )

        if probability >= threshold:

            detected_aspects.append({

                "aspect": aspect,

                "score": round(
                    probability,
                    4
                )
            })

    return detected_aspects

# ==================================================
# PREDICT SENTIMENT
# ==================================================

def predict_sentiment(
    sentence,
    aspect
):

    description = ASPECT_DESCRIPTIONS.get(
        aspect,
        aspect
    )

    input_text = (
        sentence
        + " [SEP] "
        + description
    )

    encoding = tokenizer(
        input_text,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    input_ids = encoding[
        "input_ids"
    ].to(DEVICE)

    attention_mask = encoding[
        "attention_mask"
    ].to(DEVICE)

    with torch.no_grad():

        logits = sentiment_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        probs = torch.softmax(
            logits,
            dim=1
        )

        confidence, pred = torch.max(
            probs,
            dim=1
        )

    sentiment = id2label[
        pred.item()
    ]

    low_confidence = (
        confidence.item() < SENTIMENT_THRESHOLD
    )

    if low_confidence:
        sentiment = "neutral"

    return {

        "sentiment": sentiment,

        "confidence": round(
            confidence.item(),
            4
        ),

        "low_confidence": low_confidence
    }

# ==================================================
# ANALYZE SENTENCE
# ==================================================

def analyze_sentence(sentence):

    final_results = []

    detected_aspects = detect_aspects(
        sentence
    )

    print("\nDETECTED ASPECTS:")
    print("=" * 60)

    print(detected_aspects)

    # Không detect được aspect
    if len(detected_aspects) == 0:

        return []

    print("\nSENTIMENT RESULTS:")
    print("=" * 60)

    for item in detected_aspects:

        aspect = item["aspect"]

        sentiment_result = (
            predict_sentiment(
                sentence,
                aspect
            )
        )

        print(
            f"{aspect:<25}"
            f"{sentiment_result['sentiment']:<10}"
            f"(confidence="
            f"{sentiment_result['confidence']})"
        )

        if sentiment_result["low_confidence"]:
            print(
                f"NOTE: {aspect} has low sentiment confidence "
                f"({sentiment_result['confidence']}). "
                f"Falling back to neutral."
            )

        final_results.append({

            "aspect": aspect,

            "sentiment":
                sentiment_result[
                    "sentiment"
                ],

            "aspect_score":
                item["score"],

            "sentiment_confidence":
                sentiment_result[
                    "confidence"
                ]
        })

    return final_results

# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("ABSA SYSTEM READY")
    print("=" * 60)

    print(f"\nDevice: {DEVICE}")

    while True:

        print("\nNhap cau danh gia:")
        print("(go 'exit' de thoat)\n")

        sentence = input(">>> ")

        if sentence.lower() == "exit":

            print("\nBye!")

            break

        if len(sentence.strip()) == 0:

            print("Cau rong.")

            continue

        results = analyze_sentence(
            sentence
        )

        print("\nFINAL RESULTS:")
        print("=" * 60)

        if len(results) == 0:

            print(
                "Khong phat hien ket qua phu hop."
            )

        else:

            for item in results:

                print(
                    f"Aspect: {item['aspect']}"
                )

                print(
                    f"Sentiment: {item['sentiment']}"
                )

                print(
                    f"Aspect Score: "
                    f"{item['aspect_score']}"
                )

                print(
                    f"Sentiment Confidence: "
                    f"{item['sentiment_confidence']}"
                )

                print("-" * 60)
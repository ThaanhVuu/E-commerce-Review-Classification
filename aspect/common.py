import torch
import vncorenlp
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer

from config import MAX_LEN, BATCH_SIZE, TEXT_COL, LABEL_COL


def get_device() -> torch.device:
    """
    Tự động chọn thiết bị tính toán theo thứ tự ưu tiên:
      1. CUDA  — GPU NVIDIA
      2. MPS   — GPU Apple Silicon (M1/M2/M3)
      3. CPU   — fallback

    Returns:
        torch.device: thiết bị được chọn
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def tokenize_df(
    df,
    rdrsegmenter: vncorenlp.VnCoreNLP,
    tokenizer: AutoTokenizer,
    shuffle: bool
) -> DataLoader:
    """
    Tokenize một DataFrame văn bản tiếng Việt thành DataLoader sẵn sàng đưa vào model.

    Quy trình cho mỗi dòng:
      1. Tách từ (word segmentation) bằng VnCoreNLP
      2. Encode câu đã tách từ bằng tokenizer (PhoBERT)
      3. Gom input_ids, attention_mask và label thành TensorDataset

    Args:
        df          : DataFrame chứa ít nhất cột TEXT_COL và LABEL_COL
        rdrsegmenter: VnCoreNLP đã khởi tạo với annotator "wseg"
        tokenizer   : AutoTokenizer (thường là PhobertTokenizer)
        shuffle     : True khi dùng cho tập train, False cho val/test

    Returns:
        DataLoader với batch_size=BATCH_SIZE
    """
    input_ids, attention_masks, labels = [], [], []

    for _, row in df.iterrows():
        segmented_sentence = " ".join(rdrsegmenter.tokenize(row[TEXT_COL])[0])

        enc = tokenizer(
            text=segmented_sentence,
            # text=row[TEXT_COL],
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids.append(enc["input_ids"].squeeze(0))
        attention_masks.append(enc["attention_mask"].squeeze(0))
        labels.append(torch.tensor(row[LABEL_COL], dtype=torch.long))

    dataset = TensorDataset(
        torch.stack(input_ids),
        torch.stack(attention_masks),
        torch.stack(labels),
    )

    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)
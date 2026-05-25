MODEL_NAME = "vinai/phobert-base-v2"
MODEL_SAVE = "emotion-model.pt"
NUM_LABELS = 3
LABEL_NAMES = ["Tiêu cực", "Trung tính", "Tích cực"]

# Tokenization
MAX_LEN = 128
# Training
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 2e-5
WARMUP_STEPS = 0
GRAD_CLIP = 1.0
# Class weights — phạt nặng nhãn ít mẫu (Trung tính ~18% data)
CLASS_WEIGHTS = [1.0, 2.5, 1.0]
# Data split
TEST_SIZE = 0.15
VAL_SIZE = 0.15
# Path
MODEL_DIR = './saved_models'
DATA_DIR = './data/data.csv'
SEGMENT_MODEL_DIR = './saved_models/VnCoreNLP-1.2.jar'
# data
TEXT_COL = "sentence"
LABEL_COL = "label"

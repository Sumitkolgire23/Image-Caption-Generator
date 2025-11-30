from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"

DEVICE = "cuda" if __import__('torch').cuda.is_available() else "cpu"

# model / preprocessing
EMBED_SIZE = 256
HIDDEN_SIZE = 512
NUM_LAYERS = 1
DROP_OUT = 0.5
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
NUM_EPOCHS = 20
MAX_LEN = 30
MIN_WORD_FREQ = 5

# paths
CAPTIONS_FILE = RAW_DIR / "Flickr8k.token.txt"
IMAGES_DIR = RAW_DIR / "Flickr8k_images"

VOCAB_PATH = PROCESSED_DIR / "vocab.pkl"
FEATURES_DIR = PROCESSED_DIR / "features"

MODEL_ENCODER = MODELS_DIR / "encoder.pth"
MODEL_DECODER = MODELS_DIR / "decoder.pth"
TOKENIZER = MODELS_DIR / "tokenizer.pkl"

SEED = 42
def set_seed(seed=SEED):
    import torch # type: ignore
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  
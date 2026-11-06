from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FLAG = "pathmnist"

IMAGE_SIZE = 28
PATCH_SIZE = 7
HIDDEN_DIM = 218
NUM_LAYERS = 8

BATCH_SIZE = 64

PRETRAIN_EPOCHS = 5
FINETUNE_EPOCHS = 10

PRETRAIN_BATCH = 64
FINETUNE_BATCH = 64

MASK_RATIO = 0.20

LR_PRETRAIN = 1e-3
LR_FINETUNE = 1e-2

NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"

CHECKPOINT_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

SEED = 42
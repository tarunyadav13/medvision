import torch
import sys
from pathlib import Path

# Add project root to Python import path
ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
    
from medvision.config import (
    DEVICE,
    IMAGE_SIZE,
    PATCH_SIZE,
    HIDDEN_DIM,
    NUM_LAYERS
)

from medvision.data.dataset import get_dataloaders

from medvision.models.cmixer import CMixerFull

from medvision.training.self_supervised import SSLFramework

from medvision.training.pretrain import pretrain_model

from medvision.training.finetune import train_model


def main():

    print("========== MEDVISION ==========")

    print("Device:", DEVICE)

    train_loader, val_loader, test_loader, info = (
        get_dataloaders()
    )

    channels = info["n_channels"]

    classes = len(info["label"])

    model = CMixerFull(
        in_channels=channels,
        n_classes=classes,
        img_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS
    )

    parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Model parameters: {parameters / 1e6:.2f}M"
    )

    ssl_model = SSLFramework(
        model,
        hidden_dim=HIDDEN_DIM
    )

    print("\nStarting self-supervised learning...")

    ssl_model = pretrain_model(
        ssl_model,
        train_loader
    )

    print("\nStarting supervised fine-tuning...")

    trained_model, metrics = train_model(
        ssl_model.online,
        train_loader,
        val_loader,
        test_loader,
        classes
    )

    return trained_model, metrics


if __name__ == "__main__":

    main()
import torch

from medvision.config import (
    DEVICE,
    CHECKPOINT_DIR,
    IMAGE_SIZE,
    PATCH_SIZE,
    HIDDEN_DIM,
    NUM_LAYERS
)

from medvision.data.dataset import get_dataloaders

from medvision.models.cmixer import CMixerFull

from medvision.training.finetune import evaluate_model


def main():

    _, _, test_loader, info = get_dataloaders()

    model = CMixerFull(
        in_channels=info["n_channels"],
        n_classes=len(info["label"]),
        img_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS
    ).to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT_DIR / "cmixer_pathmnist.pt",
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(
        checkpoint["model_state"]
    )

    results = evaluate_model(
        model,
        test_loader,
        len(info["label"])
    )

    print("\n========== MEDVISION EVALUATION ==========")

    print("Accuracy:", results["accuracy"])

    print("ROC-AUC:", results["roc_auc"])


if __name__ == "__main__":

    main()
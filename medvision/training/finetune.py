import copy

import torch
import torch.nn as nn
import torch.optim as optim

from medvision.config import (
    DEVICE,
    CHECKPOINT_DIR,
    FINETUNE_EPOCHS,
    LR_FINETUNE
)

from medvision.evaluation.metrics import calculate_metrics


def evaluate_model(model, loader, n_classes):

    model.eval()

    all_logits = []

    all_labels = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE).float()

            logits = model(images)

            all_logits.append(
                logits.cpu()
            )

            all_labels.append(
                labels.cpu()
            )

    logits = torch.cat(all_logits)

    labels = torch.cat(all_labels)

    return calculate_metrics(
        logits,
        labels,
        n_classes
    )


def train_model(
    model,
    train_loader,
    val_loader,
    test_loader,
    n_classes
):

    model = model.to(DEVICE)

    optimizer = optim.SGD(
        model.parameters(),
        lr=LR_FINETUNE,
        momentum=0.9,
        weight_decay=1e-4
    )

    criterion = nn.CrossEntropyLoss()

    best_auc = -float("inf")

    best_state = copy.deepcopy(
        model.state_dict()
    )

    for epoch in range(FINETUNE_EPOCHS):

        model.train()

        total_loss = 0.0

        for images, labels in train_loader:

            images = images.to(DEVICE).float()

            labels = labels.to(
                DEVICE
            ).long().view(-1)

            optimizer.zero_grad()

            logits = model(images)

            loss = criterion(
                logits,
                labels
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            total_loss += loss.item()

        avg_loss = (
            total_loss / len(train_loader)
        )

        val_metrics = evaluate_model(
            model,
            val_loader,
            n_classes
        )

        auc = val_metrics["roc_auc"]

        print(
            f"Epoch {epoch + 1}/{FINETUNE_EPOCHS} "
            f"| Loss: {avg_loss:.4f} "
            f"| Val Accuracy: {val_metrics['accuracy']:.4f} "
            f"| Val AUC: {auc:.4f}"
        )

        if not torch.isnan(torch.tensor(auc)) and auc > best_auc:

            best_auc = auc

            best_state = copy.deepcopy(
                model.state_dict()
            )

    model.load_state_dict(best_state)

    checkpoint = CHECKPOINT_DIR / "cmixer_pathmnist.pt"

    torch.save(
        {
            "model_state": model.state_dict(),
            "best_val_auc": best_auc,
            "n_classes": n_classes
        },
        checkpoint
    )

    print(f"\nCheckpoint saved: {checkpoint}")

    test_metrics = evaluate_model(
        model,
        test_loader,
        n_classes
    )

    print("\n========== TEST RESULTS ==========")

    print(
        f"Accuracy: {test_metrics['accuracy']:.4f}"
    )

    print(
        f"ROC-AUC: {test_metrics['roc_auc']:.4f}"
    )

    return model, test_metrics
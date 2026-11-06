import torch
import torch.optim as optim

from medvision.config import (
    DEVICE,
    PRETRAIN_EPOCHS,
    LR_PRETRAIN,
    MASK_RATIO
)


def pretrain_model(
    ssl_model,
    train_loader
):

    ssl_model = ssl_model.to(DEVICE)

    optimizer = optim.AdamW(
        ssl_model.parameters(),
        lr=LR_PRETRAIN,
        weight_decay=0.05
    )

    for epoch in range(PRETRAIN_EPOCHS):

        ssl_model.train()

        total_loss = 0.0

        for images, _ in train_loader:

            images = images.to(
                DEVICE
            ).float()

            target_images = torch.flip(
                images,
                dims=[3]
            )

            optimizer.zero_grad()

            loss = ssl_model(
                images,
                target_images,
                MASK_RATIO
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        print(
            f"[Pretraining] "
            f"Epoch {epoch + 1}/{PRETRAIN_EPOCHS} "
            f"| Loss: {avg_loss:.4f}"
        )

    return ssl_model
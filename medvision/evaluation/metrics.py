import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score
)


def calculate_metrics(
    logits,
    labels,
    n_classes
):

    probabilities = torch.softmax(
        logits,
        dim=1
    ).numpy()

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    labels = np.asarray(labels).reshape(-1)

    accuracy = accuracy_score(
        labels,
        predictions
    )

    try:

        if n_classes == 2:

            auc = roc_auc_score(
                labels,
                probabilities[:, 1]
            )

        else:

            auc = roc_auc_score(
                labels,
                probabilities,
                multi_class="ovr",
                average="macro"
            )

    except ValueError:

        auc = float("nan")

    return {
        "accuracy": float(accuracy),
        "roc_auc": float(auc)
    }
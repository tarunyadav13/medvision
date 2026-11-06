import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


def random_mask(x, mask_ratio=0.20):

    batch, channels, height, width = x.shape

    mask = torch.bernoulli(
        torch.ones(
            batch,
            1,
            height,
            width,
            device=x.device
        ) * (1 - mask_ratio)
    )

    return x * mask


class SSLFramework(nn.Module):

    def __init__(
        self,
        backbone,
        hidden_dim=218,
        proj_dim=128,
        ema_decay=0.99
    ):

        super().__init__()

        self.online = backbone

        self.target = copy.deepcopy(backbone)

        for parameter in self.target.parameters():

            parameter.requires_grad_(False)

        self.proj_online = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.Tanh()
        )

        self.proj_target = copy.deepcopy(
            self.proj_online
        )

        for parameter in self.proj_target.parameters():

            parameter.requires_grad_(False)

        self.ema_decay = ema_decay

    @torch.no_grad()
    def update_target(self):

        for online, target in zip(
            self.online.parameters(),
            self.target.parameters()
        ):

            target.mul_(self.ema_decay)

            target.add_(
                online,
                alpha=1 - self.ema_decay
            )

        for online, target in zip(
            self.proj_online.parameters(),
            self.proj_target.parameters()
        ):

            target.mul_(self.ema_decay)

            target.add_(
                online,
                alpha=1 - self.ema_decay
            )

    def forward(
        self,
        anchor,
        target_image,
        mask_ratio=0.20
    ):

        anchor = random_mask(
            anchor,
            mask_ratio
        )

        target_image = random_mask(
            target_image,
            mask_ratio
        )

        online_features = self.online.get_features(
            anchor
        )

        online_projection = self.proj_online(
            online_features
        )

        with torch.no_grad():

            target_features = self.target.get_features(
                target_image
            )

            target_projection = self.proj_target(
                target_features
            )

        target_distribution = F.softmax(
            target_projection,
            dim=-1
        )

        loss = -(
            target_distribution *
            F.log_softmax(
                online_projection,
                dim=-1
            )
        ).sum(dim=-1).mean()

        self.update_target()

        return loss
import torch

from medvision.models.cmixer import CMixerFull


def test_cmixer_output():

    model = CMixerFull(
        in_channels=3,
        n_classes=9,
        img_size=28,
        patch_size=7,
        hidden_dim=64,
        num_layers=2
    )

    images = torch.randn(
        2, 3, 28, 28
    )

    output = model(images)

    assert output.shape == (2, 9)


def test_cmixer_parameters():

    model = CMixerFull(
        in_channels=3,
        n_classes=9,
        hidden_dim=64,
        num_layers=2,
        patch_size=7
    )

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    assert parameters > 0
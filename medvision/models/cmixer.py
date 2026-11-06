import torch
import torch.nn as nn

from medvision.models.complex_layers import (
    ComplexLinear,
    ComplexMLP,
    PearsonCurve
)


class CMixerLayer(nn.Module):

    def __init__(
        self,
        seq_len,
        hidden_dim
    ):

        super().__init__()

        self.norm1 = nn.LayerNorm(seq_len)

        self.norm2 = nn.LayerNorm(hidden_dim)

        self.token_mix = ComplexMLP(
            seq_len,
            expansion=2
        )

        self.channel_mix = ComplexMLP(
            hidden_dim,
            expansion=4
        )

    def forward(self, real, imag):

        residual_real = real
        residual_imag = imag

        real_t = self.norm1(
            real.transpose(1, 2)
        )

        imag_t = imag.transpose(1, 2)

        real_t, imag_t = self.token_mix(
            real_t,
            imag_t
        )

        real = residual_real + real_t.transpose(1, 2)

        imag = residual_imag + imag_t.transpose(1, 2)

        residual_real = real
        residual_imag = imag

        real_n = self.norm2(real)

        real_n, imag_n = self.channel_mix(
            real_n,
            imag
        )

        real = residual_real + real_n

        imag = residual_imag + imag_n

        return real, imag


class CMixer(nn.Module):

    def __init__(
        self,
        in_channels,
        n_classes,
        img_size=28,
        patch_size=8,
        hidden_dim=218,
        num_layers=8
    ):

        super().__init__()

        if img_size % patch_size != 0:
            raise ValueError(
                "Image size must be divisible by patch size."
            )

        self.patch_size = patch_size

        grid = img_size // patch_size

        self.n_patches = grid * grid

        patch_dim = (
            in_channels * patch_size * patch_size
        )

        self.proj_real = nn.Linear(
            patch_dim,
            hidden_dim
        )

        self.proj_imag = nn.Linear(
            patch_dim,
            hidden_dim
        )

        self.layers = nn.ModuleList([

            CMixerLayer(
                self.n_patches,
                hidden_dim
            )

            for _ in range(num_layers)

        ])

        self.norm = nn.LayerNorm(hidden_dim)

        self.pearson = PearsonCurve()

        self.head = nn.Linear(
            hidden_dim,
            n_classes
        )

    def patchify(self, x):

        batch, channels, height, width = x.shape

        p = self.patch_size

        x = x.unfold(2, p, p)

        x = x.unfold(3, p, p)

        x = x.contiguous().view(
            batch,
            channels,
            -1,
            p * p
        )

        x = x.permute(0, 2, 1, 3)

        x = x.reshape(
            batch,
            self.n_patches,
            -1
        )

        return x

    def forward(self, x_real, x_imag):

        real_patches = self.patchify(x_real)

        imag_patches = self.patchify(x_imag)

        real = self.proj_real(real_patches)

        imag = self.proj_imag(imag_patches)

        for layer in self.layers:

            real, imag = layer(real, imag)

        real = real.mean(dim=1)

        imag = imag.mean(dim=1)

        features = self.pearson(
            self.norm(real),
            imag
        )

        return self.head(features)

    def get_features(self, x_real, x_imag):

        real = self.proj_real(
            self.patchify(x_real)
        )

        imag = self.proj_imag(
            self.patchify(x_imag)
        )

        for layer in self.layers:

            real, imag = layer(real, imag)

        real = real.mean(dim=1)

        imag = imag.mean(dim=1)

        return self.pearson(
            self.norm(real),
            imag
        )


class CMixerFull(nn.Module):

    def __init__(
        self,
        in_channels,
        n_classes,
        img_size=28,
        patch_size=7,
        hidden_dim=218,
        num_layers=8
    ):

        super().__init__()

        self.imag_gen = ImaginaryMatrixGeneratorPlaceholder(
            in_channels,
            img_size
        )

        self.mixer = CMixer(
            in_channels,
            n_classes,
            img_size,
            patch_size,
            hidden_dim,
            num_layers
        )

    def forward(self, x):

        imaginary = self.imag_gen(x)

        return self.mixer(x, imaginary)

    def get_features(self, x):

        imaginary = self.imag_gen(x)

        return self.mixer.get_features(
            x,
            imaginary
        )


from medvision.models.imaginary_generator import ImaginaryMatrixGenerator as ImaginaryMatrixGeneratorPlaceholder
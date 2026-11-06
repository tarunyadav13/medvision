import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexLinear(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True
    ):
        super().__init__()

        self.A = nn.Linear(
            in_features,
            out_features,
            bias=bias
        )

        self.B = nn.Linear(
            in_features,
            out_features,
            bias=False
        )

    def forward(self, real, imag):

        real_out = self.A(real) - self.B(imag)

        imag_out = self.B(real) + self.A(imag)

        return real_out, imag_out


def crelu(real, imag):

    return F.relu(real), F.relu(imag)


class PearsonCurve(nn.Module):

    def forward(self, real, imag):

        return torch.tanh(real + imag)


class ComplexMLP(nn.Module):

    def __init__(self, dim, expansion=4):

        super().__init__()

        hidden = dim * expansion

        self.fc1 = ComplexLinear(dim, hidden)

        self.fc2 = ComplexLinear(hidden, dim)

    def forward(self, real, imag):

        real, imag = self.fc1(real, imag)

        real, imag = crelu(real, imag)

        real, imag = self.fc2(real, imag)

        return real, imag
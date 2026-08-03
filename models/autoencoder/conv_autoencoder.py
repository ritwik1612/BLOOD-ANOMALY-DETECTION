from __future__ import annotations

import torch
from torch import nn


class ConvAutoencoder(nn.Module):
    """64x64 RGB convolutional autoencoder with a 128-dimensional bottleneck."""

    def __init__(self, latent_dim: int = 128) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            self._encoder_block(3, 32),
            self._encoder_block(32, 64),
            self._encoder_block(64, 128),
            self._encoder_block(128, 256),
        )
        self.to_latent = nn.Sequential(nn.Flatten(), nn.Linear(256 * 4 * 4, latent_dim))
        self.from_latent = nn.Sequential(nn.Linear(latent_dim, 256 * 4 * 4), nn.ReLU(inplace=True))
        self.decoder = nn.Sequential(
            self._decoder_block(256, 256),
            self._decoder_block(256, 128),
            self._decoder_block(128, 64),
            self._decoder_block(64, 32),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _encoder_block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    @staticmethod
    def _decoder_block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.to_latent(self.encoder(images))

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        features = self.from_latent(latent).view(-1, 256, 4, 4)
        return self.decoder(features)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(images)
        return self.decode(latent), latent

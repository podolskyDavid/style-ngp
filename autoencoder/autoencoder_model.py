import torch
from torch import nn


# TODO: make deeper?
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Create encoder that transforms 256x256 image to 8x8 feature map; use at least 3 layers
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=4, stride=4),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            # Reverse encoder with conv2d transpose
            nn.ConvTranspose2d(8, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 16, kernel_size=4, stride=4),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=4),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        print(f"shape of encoder output: {x.shape}")
        x = self.decoder(x)
        print(f"shape of decoder output: {x.shape}")
        return x

    def forward_encoder(self, x):
        return self.encoder(x)

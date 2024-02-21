#  Write a simple python script that initializes the autoencoder_model that uses a hardcoded array of pngs to train

# Path: autoencoder/autoencoder_training.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.optim import Adam
from torch.nn import functional as F
from tqdm import tqdm
from PIL import Image
from typing import List
from pathlib import Path

import argparse
from autoencoder_model import Autoencoder
from torch.utils.data import DataLoader
from torch import nn

from datasets import StylesAndMasks

import imageio


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=10000)
    return parser.parse_args()


if __name__ == "__main__":
    # Get arguments
    args = parse_args()

    # Create autoencoders
    autoencoder_parenchyma = Autoencoder()
    autoencoder_vessels = Autoencoder()

    # Put on device
    device = 'cuda:0'
    autoencoder_parenchyma.to(device)
    autoencoder_vessels.to(device)

    # Create dataset
    dataset = StylesAndMasks(args.data_dir)

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Create loss
    loss = nn.MSELoss(reduction='none')

    # Set up for training
    autoencoder_parenchyma.train()
    autoencoder_vessels.train()

    # Create adam optimizers
    optimizer_parenchyma = Adam(autoencoder_parenchyma.parameters(), lr=args.learning_rate)
    optimizer_vessels = Adam(autoencoder_vessels.parameters(), lr=args.learning_rate)

    for epoch in range(args.num_epochs):
        for data in dataloader:
            # Extract image and mask
            img, mask = data

            # Put on device
            img = img.to(device)
            mask = mask.to(device)
            print(f"shape of img: {img.shape}")

            # TODO: probably need to scale mask between 0 and 1

            # Prepare backward pass
            optimizer_parenchyma.zero_grad()
            optimizer_vessels.zero_grad()

            # Forward pass
            output_parenchyma = autoencoder_parenchyma(img)
            output_vessels = autoencoder_vessels(img)

            # Save the first output for debugging, reshuffle the channels to get an RGB
            imageio.imwrite(
                "output_parenchyma.png",
                (output_parenchyma[0].cpu().detach().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            )
            imageio.imwrite(
                "output_vessels.png",
                (output_vessels[0].cpu().detach().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            )

            # Compute losses
            loss_parenchyma = loss(output_parenchyma, img)
            loss_vessels = loss(output_vessels, img)

            # Create a mask for parenchyma and vessels; mask is a 3 channel image with parenchyma = red and
            #  vessels = green
            mask_red = mask[:, 0, :, :].unsqueeze(1).expand(-1, 3, -1, -1)
            mask_green = mask[:, 1, :, :].unsqueeze(1).expand(-1, 3, -1, -1)

            # Save those as images as well to debug
            imageio.imwrite(
                "mask_parenchyma.png",
                (mask_red[0].cpu().detach().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            )
            imageio.imwrite(
                "mask_vessel.png",
                (mask_green[0].cpu().detach().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            )

            # Multiply the losses by the masks
            loss_parenchyma = (loss_parenchyma * mask_red).mean()
            loss_vessels = (loss_vessels * mask_green).mean()

            # Backward pass
            loss_parenchyma.backward()
            loss_vessels.backward()

            # Update weights
            optimizer_parenchyma.step()
            optimizer_vessels.step()
            print(f"Epoch [{epoch + 1}/{args.num_epochs}], Loss parenchyma: {loss_parenchyma:.4f}, Loss vessels: {loss_vessels:.4f}")
    print("Training finished")

    # Save the 2 models
    torch.save(autoencoder_parenchyma.state_dict(), "autoencoder_parenchyma.pth")
    torch.save(autoencoder_vessels.state_dict(), "autoencoder_vessels.pth")
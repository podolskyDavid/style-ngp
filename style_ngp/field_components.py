import torch
from torchvision import models
from torch import nn

from style_ngp.util import compute_gram_matrix


class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        # Extracting the first 5 blocks of VGG16
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        return

    def forward(self, X):
        h = self.slice1(X)
        h = self.slice2(h)
        h = self.slice3(h)
        h = self.slice4(h)
        h = self.slice5(h)

        # Don't flatten the last-layer feature maps but instead compute Gram matrix, then flatten
        gram_matrix = compute_gram_matrix(h)

        # Create view of (batch_size, -1) to flatten the gram matrix
        return gram_matrix.view(X.size(0), -1)


class SimpleFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv1x1 = nn.Conv2d(16, 1, kernel_size=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=16, stride=16)
        self.flatten = nn.Flatten()

    def forward(self, X):
        h = self.conv(X)
        h = self.conv1x1(h)
        h = self.relu(h)
        h = self.pool(h)
        h = self.flatten(h)
        return h


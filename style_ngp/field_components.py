import torch
from torchvision import models
from torch import nn

import torch
import torchvision.io


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
        self.normalization = torch.nn.LayerNorm(normalized_shape=256)
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
        h1 = self.slice1(X)  # 256x256
        h2 = self.slice2(h1)  # 128x128
        h3 = self.slice3(h2)  # 64x64
        h4 = self.slice4(h3)  # 32x32
        h5 = self.slice5(h4)  # 16x16

        # # Use pooling to make h3 16x16
        # h3 = nn.AvgPool2d(kernel_size=4, stride=4)(h3).mean(dim=1, keepdim=True).view(h3.size(0), -1)
        # h5 = h5.mean(dim=1, keepdim=True).view(h5.size(0), -1)
        # # Normalize both
        # # Apply Layer Normalization
        # # TODO: hardcoded for now, fix later
        # h3_normalized = self.normalization(h3)
        # h5_normalized = self.normalization(h5)
        # concat_features = torch.cat((h3_normalized, h5_normalized), dim=1)
        # return concat_features

        # Only return the middle layer features
        h3 = nn.AvgPool2d(kernel_size=4, stride=4)(h3).mean(dim=1, keepdim=True).view(h3.size(0), -1)
        return self.normalization(h3)


class HistogramExtractor:
    '''
    Extracts the histogram of an image
    '''
    def get_hist(self, img_path):
        # Load the image directly into a tensor
        image_tensor = torchvision.io.read_image(img_path).float() / 255.0

        # Define the number of bins and range
        num_bins = 32
        # Image tensors are normalized, so range is [0, 1]
        min_range = 0
        max_range = 1

        # Initialize the histogram tensor
        hist_tensor = torch.zeros((3, num_bins))

        # Calculate the histogram for each channel
        for i in range(3):
            hist = torch.histc(image_tensor[i], bins=num_bins, min=min_range, max=max_range)
            hist = hist / hist.sum()  # Normalize the histogram
            hist_tensor[i] = hist

        # Flatten the histogram to use as input to neural network
        hist_flattened = hist_tensor.view(1, -1)

        return hist_flattened


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


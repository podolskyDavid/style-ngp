import torch
from imageio import imread
from PIL import Image
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from torchvision.io import read_image
from torchvision.transforms import v2


def load_img(img_path):
    if img_path is None:
        raise ValueError("Style image path is None.")

    # Load image
    image = read_image(img_path)

    # Remove alpha channel
    if image.shape[0] == 4:
        image = image[:3, :, :]

    # Crop and resize the image
    size_image = image.size()
    min_dim = min(size_image[1], size_image[2])
    image = transforms.CenterCrop(min_dim)(image)
    image = transforms.Resize((256, 256))(image)

    # Bring to [0, 1] and float
    image = image / 255.0

    # Normalize it according to imagenet
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image = normalize(image)

    return image


import torch
from imageio import imread
from PIL import Image
import numpy as np
import torch.nn.functional as F
from torchvision import transforms

def load_img(img_path):
    if img_path is None:
        raise ValueError("Style image path is None.")

    # Load image
    image = imread(img_path)
    # Remove alpha channel
    image = image[:, :, :3]
    # Convert the image to a PIL Image
    image = Image.fromarray(image)
    # Get the dimensions of the image
    width, height = image.size
    # Determine the size of the square and the top left coordinates of the square
    size = min(width, height)
    left = (width - size) / 2
    top = (height - size) / 2
    right = (width + size) / 2
    bottom = (height + size) / 2

    # Crop the image to a square
    image = image.crop((left, top, right, bottom))
    # Resize the image to 256x256
    image = image.resize((256, 256))

    # Convert the image back to a numpy array
    image = np.array(image)
    # Convert the image to PyTorch tensor and normalize it to [0, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # transforms.Lambda(lambda x: x.half())
    ])
    image = transform(image)

    # Add an extra dimension for the batch size
    image = image.unsqueeze(0)
    # Move image to GPU
    image = image.to("cuda:0")
    return image

def compute_gram_matrix(features):
    a, b, c, d = features.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = features.view(a * b, c * d)  # resize F_XL into \hat F_XL

    gram = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    gram_normalized = gram.div(a * b * c * d)

    # Apply average pooling to the gram matrix to reduce its size to 16 x 16
    gram_pooled = F.avg_pool2d(gram_normalized.unsqueeze(0), kernel_size=(32, 32)).squeeze(0)

    return gram_pooled


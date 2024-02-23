import torch
from imageio import imread
from PIL import Image
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import v2


# TODO: rewrite everything to use torch only


def crop_and_resize(image, target_size=256, mask=False):
    """ Crop and resize the image to a square image taken from center. Be aware that the conversion to PIL and back
    to numpy array might change the values of the pixels if they are not in the range [0, 255]."""
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


    # Resize the image
    if mask:
        image = image.resize((target_size, target_size), Image.NEAREST)
    else:
        image = image.resize((target_size, target_size))

    # Convert the image back to a numpy array
    image = np.array(image)

    return image

def load_img(img_path, augment=False):
    if img_path is None:
        raise ValueError("Style image path is None.")

    # Load image
    image = imread(img_path)
    # Remove alpha channel
    image = image[:, :, :3]

    # Crop and resize the image
    image = crop_and_resize(image)

    # Convert the image to PyTorch tensor and normalize it according to imagenet
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # transforms.Lambda(lambda x: x.half())
    ])
    # Apply transform
    image = transform(image)

    # if augment:
    #     techniques = [
    #         # Takes random crop, then resizes to desired size
    #         v2.CenterCrop(size=(256, 256)),
    #         # Randomly rotates the image
    #         v2.RandomRotation(degrees=(0, 180))
    #     ]
    #     # Apply random augmentation technique
    #     technique = np.random.randint(len(techniques))
    #     additional_transform = techniques[technique]
    #     image = additional_transform(image)
    #
    #     # Save image in local folder for debugging
    #     debug_image = image.numpy()
    #     debug_image = np.moveaxis(debug_image, 0, -1)
    #     debug_image = (debug_image * 255).astype(np.uint8)
    #     debug_image = Image.fromarray(debug_image)
    #     debug_image.save(f'./augmented_image.png')

    return image


def load_mask(mask_path, scale_factor=None):
    if mask_path is None:
        raise ValueError("Mask path is None.")

    # Load mask
    mask = imread(mask_path)
    # Remove alpha channel
    mask = mask[:, :, :3]

    # Values need to be in [0, 255] for cropping and resizing; scale_factor is used to scale the mask
    if scale_factor is not None:
        mask = mask * scale_factor
    # Convert to int array
    mask = mask.astype(np.uint8)

    # Crop and resize the mask
    mask = crop_and_resize(mask, mask=True)

    # Convert the mask to PyTorch tensor
    transform = transforms.ToTensor()
    mask = transform(mask)

    return mask


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


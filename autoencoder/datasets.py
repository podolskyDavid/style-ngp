import os
from torch.utils.data import Dataset
from torchvision.io import read_image
from style_ngp.util import load_img, load_mask


class StylesAndMasks(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir

        # Get all files in that dir
        self.files = os.listdir(data_dir)

        # Track style images and their masks
        self.style_imgs = []
        self.masks = []

        for file in self.files:
            # style images all pngs that do not contain 'mask'
            if 'mask' not in file:
                # Exclude 2 styles
                if 'case207' not in file and 'style03' not in file:
                    self.style_imgs.append(os.path.join(data_dir, file))
                    self.masks.append(os.path.join(data_dir, file.replace('.png', '_mask.png')))

    def __len__(self):
        assert len(self.style_imgs) == len(self.masks)
        return len(self.style_imgs)

    def __getitem__(self, idx):
        # Load image
        image = load_img(self.style_imgs[idx])

        # Load mask; need a scale factor here since our masks are in [0, 128]
        mask = load_mask(self.masks[idx], scale_factor=255/128)

        return image, mask

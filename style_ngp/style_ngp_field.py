"""
Template Nerfstudio Field

Currently this subclasses the NerfactoField. Consider subclassing the base Field.
"""


from nerfstudio.fields.nerfacto_field import NerfactoField  # for subclassing NerfactoField
from nerfstudio.fields.base_field import Field  # for custom Field

from typing import Dict, Literal, Optional, Tuple

import torch
from torch import Tensor, nn
# TODO: remove later, just for now to test
from torchvision import transforms

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.encodings import NeRFEncoding, SHEncoding
from nerfstudio.field_components.field_heads import (
    FieldHeadNames,
    PredNormalsFieldHead,
    SemanticFieldHead,
    TransientDensityFieldHead,
    TransientRGBFieldHead,
    UncertaintyFieldHead,
)
from nerfstudio.field_components.mlp import MLP, MLPWithHashEncoding
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field, get_normalized_directions

from style_ngp.field_components import VGGFeatureExtractor
from style_ngp.util import flip_parameters_to_tensors, set_all_parameters

from imageio import imread
from PIL import Image
import numpy as np

import hyperlight as hl

import math

import os

class StyleNGPField(Field):
    """Compound Field

    Args:
        aabb: parameters of scene aabb bounds
        num_images: number of images in the dataset
        num_layers: number of hidden layers
        hidden_dim: dimension of hidden layers
        geo_feat_dim: output geo feat dimensions
        num_levels: number of levels of the hashmap for the base mlp
        base_res: base resolution of the hashmap for the base mlp
        max_res: maximum resolution of the hashmap for the base mlp
        log2_hashmap_size: size of the hashmap for the base mlp
        num_layers_color: number of hidden layers for color network
        features_per_level: number of features per level for the hashgrid
        hidden_dim_color: dimension of hidden layers for color network
    """

    aabb: Tensor

    def __init__(
        self,
        aabb: Tensor,
        num_images: int,
        num_layers: int = 2,
        hidden_dim: int = 64,
        geo_feat_dim: int = 15,
        num_levels: int = 16,
        base_res: int = 16,
        max_res: int = 2048,
        log2_hashmap_size: int = 19,
        num_layers_color: int = 3,
        features_per_level: int = 2,
        hidden_dim_color: int = 16,
        average_init_density: float = 1.0,
        implementation: Literal["tcnn", "torch"] = "tcnn",
    ) -> None:
        # Call the superclass with the same arguments
        super().__init__()

        self.register_buffer("aabb", aabb)
        self.geo_feat_dim = geo_feat_dim

        self.register_buffer("max_res", torch.tensor(max_res))
        self.register_buffer("num_levels", torch.tensor(num_levels))
        self.register_buffer("log2_hashmap_size", torch.tensor(log2_hashmap_size))

        self.num_images = num_images
        self.base_res = base_res
        self.average_init_density = average_init_density
        self.step = 0

        self.direction_encoding = SHEncoding(
            levels=4,
            implementation=implementation,
        )

        self.position_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=2, min_freq_exp=0, max_freq_exp=2 - 1, implementation=implementation
        )

        self.mlp_base = MLPWithHashEncoding(
            num_levels=num_levels,
            min_res=base_res,
            max_res=max_res,
            log2_hashmap_size=log2_hashmap_size,
            features_per_level=features_per_level,
            num_layers=num_layers,
            layer_width=hidden_dim,
            out_dim=1 + self.geo_feat_dim,
            activation=nn.ReLU(),
            out_activation=None,
            implementation=implementation,
        )

        self.start_mlp_head = MLP(
            in_dim=self.direction_encoding.get_out_dim() + self.geo_feat_dim,
            num_layers=num_layers_color,
            layer_width=hidden_dim_color,
            out_dim=3,
            activation=nn.ReLU(),
            out_activation=nn.Sigmoid(),
            implementation=implementation,
        )

        # self.test_mlp_head = MLP(
        #     in_dim=self.direction_encoding.get_out_dim() + self.geo_feat_dim,
        #     num_layers=num_layers_color,
        #     layer_width=hidden_dim_color,
        #     out_dim=3,
        #     activation=nn.ReLU(),
        #     out_activation=nn.Sigmoid(),
        #     implementation=implementation,
        # )

        num_layers_hyper_mlp = 2
        layer_width_hyper_mlp = 128
        self.hyper_mlp_head = MLP(
            in_dim=self.direction_encoding.get_out_dim() + 3,  # +3 for the rgb values
            num_layers=num_layers_hyper_mlp,
            layer_width=layer_width_hyper_mlp,
            out_dim=3,
            activation=nn.ReLU(),
            out_activation=nn.Sigmoid(),
            implementation=implementation,
        )

        # Add feature extractor
        self.feature_extractor = VGGFeatureExtractor()
        # Freeze the feature extractor, remains frozen throughout
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # Add rgb transform net that will be controlled by hypernets
        self.hyper_mlp_head = hl.hypernetize(self.hyper_mlp_head, [self.hyper_mlp_head.tcnn_encoding])
        weights_shape = self.hyper_mlp_head.external_shapes()['tcnn_encoding.params'][0]
        print(f"weights_shape: {weights_shape}")

        # TODO: hardcoded here, needs to be changed later
        features_dim = 256
        input_shapes = {'h': (features_dim,)}

        #         num_layers_color: int = 3,
        #         hidden_dim_color: int = 16,
        self.hypernet = MLP(
            in_dim=input_shapes['h'][0],
            num_layers=num_layers_color,
            layer_width=hidden_dim_color,
            out_dim=weights_shape,
            activation=nn.ReLU(),
            out_activation=None,
            implementation=implementation,
        ).to("cuda:0")

        # Add variable that indicates whether the hypernetwork is active
        self.hypernetwork_active = False

        # Add variables that keeps track of the style img to use and its features
        self.style_img_path = None
        self.style_features = None

    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, Tensor]:
        """Computes and returns the densities."""
        positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
        # Make sure the tcnn gets inputs between 0 and 1.
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        positions = positions * selector[..., None]
        self._sample_locations = positions
        if not self._sample_locations.requires_grad:
            self._sample_locations.requires_grad = True
        positions_flat = positions.view(-1, 3)
        h = self.mlp_base(positions_flat).view(*ray_samples.frustums.shape, -1)
        density_before_activation, base_mlp_out = torch.split(h, [1, self.geo_feat_dim], dim=-1)
        self._density_before_activation = density_before_activation

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = self.average_init_density * trunc_exp(density_before_activation.to(positions))
        density = density * selector[..., None]
        return density, base_mlp_out

    # def save_checkpoint(self, folder, style):
    #     # Assume model is your PyTorch model
    #     state_dict = self.state_dict()
    #
    #     # List of keys to exclude
    #     exclude_keys = []  # Adjust keys accordingly
    #
    #     # Create a new dict without excluded keys
    #     filtered_state_dict = {k: v for k, v in state_dict.items() if k not in exclude_keys}
    #
    #     print(f"filtered_state_dict keys: {filtered_state_dict.keys()}")
    #
    #     torch.save(filtered_state_dict, os.path.join(folder, f"{style}.pth"))
    #
    #     # TODO: remove, only debugging
    #     self.load_checkpoint(folder, style)
    #     return
    #
    # def load_checkpoint(self, folder, style):
    #     # Load the whole field
    #     self = torch.load(os.path.join(folder, f"{style}.pth"))
    #     return

    def activate_hypernetwork(self):
        self.hypernetwork_active = True

        # Freeze the positional encoding
        for param in self.position_encoding.parameters():
            param.requires_grad = False

        # Freeze the weights of the mlp_base, density network remains untouched
        for param in self.mlp_base.parameters():
            param.requires_grad = False

        # Freeze the direction encoding
        for param in self.direction_encoding.parameters():
            param.requires_grad = False

        # Freeze the weights of the start_mlp_head
        for param in self.start_mlp_head.parameters():
            param.requires_grad = False
        return

    def reset_rgb(self):
        self.start_mlp_head.load_state_dict(self.test_mlp_head.state_dict())
        print("resetting start_mlp_head to untrained test_mlp_head")

    def deactivate_hypernetwork(self):
        # TODO: Implement
        return

    def load_img(self, img_path):
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

    def update_mlp_head(self):
        new_params =  {"tcnn_encoding.params": self.hypernet(self.style_features).view(-1,)}
        return new_params
    
    def update_style_img(self, img_path):
        self.style_img_path = img_path
        print(f"new img_path is {img_path}")

        # Load style image
        style_img = self.load_img(self.style_img_path)
        # Extract features
        self.style_features = self.feature_extractor(style_img)
        return None

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None
    ) -> Dict[FieldHeadNames, Tensor]:
        assert density_embedding is not None
        outputs = {}
        if ray_samples.camera_indices is None:
            raise AttributeError("Camera indices are not provided.")
        camera_indices = ray_samples.camera_indices.squeeze()
        directions = get_normalized_directions(ray_samples.frustums.directions)
        directions_flat = directions.view(-1, 3)
        d = self.direction_encoding(directions_flat)

        outputs_shape = ray_samples.frustums.directions.shape[:-1]

        # TODO: what is adding the empty list for? is only empty when there is no appearance embedding
        h = torch.cat(
            [
                d,
                density_embedding.view(-1, self.geo_feat_dim),
            ]
            + (
                []
            ),
            dim=-1,
        )

        # Let hypernet update the mlp_head weights if in that training stage
        if self.hypernetwork_active:
            # Get new weights
            new_params = self.update_mlp_head()

            # Run normal RGB net
            rgb = self.start_mlp_head(h).view(*outputs_shape, -1).to(directions)

            # and then use the hyper network
            with self.hyper_mlp_head.using_externals(new_params):
                # Input hypernet
                concat_input = torch.cat([d, rgb], dim=-1)
                rgb = self.hyper_mlp_head(concat_input).view(*outputs_shape, -1).to(directions)
        else:
            rgb = self.start_mlp_head(h).view(*outputs_shape, -1).to(directions)

        outputs.update({FieldHeadNames.RGB: rgb})

        return outputs

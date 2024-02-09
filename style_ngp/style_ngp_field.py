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

        self.hyper_mlp_head = MLP(
            in_dim=self.direction_encoding.get_out_dim() + self.geo_feat_dim,
            num_layers=num_layers_color,
            layer_width=hidden_dim_color,
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

        # Add rgb net that will be controlled by hypernets
        self.hyper_mlp_head = hl.hypernetize(self.hyper_mlp_head, [self.hyper_mlp_head.tcnn_encoding])

        # TODO: hardcoded here, needs to be changed later
        features_dim = 256
        input_shapes = {'h': (features_dim,)}

        # Initialize the list to store the hypernets
        self.hypernets = []

        # TODO: adapt the following hypernet size calculation to pytorch at some point

        # input dim needs to be a multiple of 16 in tcnn model
        input_dim_16 = math.ceil((self.direction_encoding.get_out_dim() + self.geo_feat_dim) / 16) * 16
        input_layer_params = input_dim_16 * hidden_dim_color

        hypernet_hidden_sizes = [16, 32, 64, 64]

        # Create hypernet for the first layer
        hypernet = hl.HyperNet(
            input_shapes=input_shapes,
            output_shapes={'tcnn_encoding.params': (input_layer_params,)},
            hidden_sizes=hypernet_hidden_sizes,
        ).to("cuda:0")
        self.hypernets.append(hypernet)

        # Iterate through hidden layers of target net; num_layers_color - 1 because it counts the output layer as well
        for i in range(num_layers_color - 1):
            if i == num_layers_color - 2:
                # Layer that connects to the output layer - rgb with 3 values - but also multiple of 16 in tcnn model
                hidden_layer_params = hidden_dim_color * 16
            else:
                hidden_layer_params = hidden_dim_color * hidden_dim_color
            print(f"hidden_layer_params: {hidden_layer_params}")

            # Condition hypernet also on previous hypernets output
            if i == 0:
                input_shapes = {'h': (features_dim + input_layer_params,)}
            else:
                input_shapes = {'h': (features_dim + hidden_layer_params,)}
            print(f"input shapes: {input_shapes}")

            hypernet = hl.HyperNet(
                input_shapes=input_shapes,
                output_shapes={'tcnn_encoding.params': (hidden_layer_params,)},
                hidden_sizes=hypernet_hidden_sizes,
            ).to("cuda:0")
            self.hypernets.append(hypernet)

        # Add variable that indicates whether the hypernetwork is active
        self.hypernetwork_active = False

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

    def activate_hypernetwork(self):
        self.hypernetwork_active = True

        # # Freeze the weights of the mlp_base, density network remains untouched
        for param in self.mlp_base.parameters():
            param.requires_grad = False

        # # Freeze the positional encoding
        # for param in self.position_encoding.parameters():
        #     param.requires_grad = False

        return

    def deactivate_hypernetwork(self):
        # TODO: Implement
        return


    def update_mlp_head(self):
        # TODO: hardcoded dirty version, remove later
        # Load image
        image = imread(f"/home/maximilian_fehrentz/Documents/nerf_data/VR/MRA/images/0001.jpg")
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

        # Extract features
        features = self.feature_extractor(image)

        # # Run hypernetwork
        # new_params = self.hypernet(h=features)

        # # Run hypernetworks and concatenate the tensors of key tcnn_encoding.params, returned again as a dict with that
        # #  key
        # new_params = {}
        # outputs = torch.cat(
        #     [hypernet(h=features)["tcnn_encoding.params"] for hypernet in self.hypernets], dim=-1
        # )
        # print(f"shape of outputs: {outputs.shape}")
        # new_params["tcnn_encoding.params"] = outputs

        pred_params_list = []
        for i in range(len(self.hypernets)):
            if i == 0:
                pred_params = self.hypernets[i](h=features)["tcnn_encoding.params"]
            else:
                # Normalize the previous predicted parameters to be between 0 and 1
                normalized_prev_pred_params = (pred_params_list[i - 1] - pred_params_list[i - 1].min()) / (
                    pred_params_list[i - 1].max() - pred_params_list[i - 1].min()
                )
                pred_params = self.hypernets[i](
                    h=torch.cat([features, normalized_prev_pred_params.unsqueeze(0)], dim=-1)
                )["tcnn_encoding.params"]
            pred_params_list.append(pred_params)

        pred_params_tensor = torch.cat(pred_params_list, dim=-1)
        new_params = {"tcnn_encoding.params": pred_params_tensor}

        return new_params

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
            new_params = self.update_mlp_head()

            # and then use the main network
            with self.hyper_mlp_head.using_externals(new_params):
                # Within this with block, the weights are accessible
                rgb = self.hyper_mlp_head(h).view(*outputs_shape, -1).to(directions)

        else:
            rgb = self.start_mlp_head(h).view(*outputs_shape, -1).to(directions)

        outputs.update({FieldHeadNames.RGB: rgb})

        return outputs

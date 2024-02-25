"""
Template Nerfstudio Field

Currently this subclasses the NerfactoField. Consider subclassing the base Field.
"""


from nerfstudio.fields.nerfacto_field import NerfactoField  # for subclassing NerfactoField
from nerfstudio.fields.base_field import Field  # for custom Field

from typing import Dict, Literal, Optional, Tuple

import torch
from torch import Tensor, nn

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

from style_ngp.field_components import VGGFeatureExtractor, SimpleFeatureExtractor, HistogramExtractor
from style_ngp.util import load_img

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
        num_layers_color: int = 2,
        features_per_level: int = 2,
        hidden_dim_color: int = 8,
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

        num_layers_hyper_mlp = num_layers_color
        layer_width_hyper_mlp = hidden_dim_color
        input_dim_hyper_mlp = self.direction_encoding.get_out_dim() + self.geo_feat_dim
        self.hyper_mlp_head = MLP(
            in_dim=input_dim_hyper_mlp,
            num_layers=num_layers_hyper_mlp,
            layer_width=layer_width_hyper_mlp,
            out_dim=3,
            activation=nn.ReLU(),
            out_activation=nn.Sigmoid(),
            implementation=implementation,
        )

        # Add feature extractor
        self.feature_extractor = HistogramExtractor()

        # # Freeze the feature extractor, remains frozen throughout
        # for param in self.feature_extractor.parameters():
        #     param.requires_grad = False

        # Add rgb transform net that will be controlled by hypernets
        self.hyper_mlp_head = hl.hypernetize(self.hyper_mlp_head, [self.hyper_mlp_head.tcnn_encoding])
        weights_shape = self.hyper_mlp_head.external_shapes()['tcnn_encoding.params'][0]
        print(f"weights_shape: {weights_shape}")

        # TODO: hardcoded here, needs to be changed later
        features_dim = 96

        num_layers_hypernet = 4
        layer_width_hypernet = 32
        out_hypernet = 128
        self.hypernet = MLP(
            in_dim=features_dim,
            num_layers=num_layers_hypernet,
            layer_width=layer_width_hypernet,
            out_dim=out_hypernet,
            activation=nn.ReLU(),
            out_activation=nn.ReLU(),
            implementation=implementation,
        ).to("cuda:0")

        # Initialize the list to store the heads
        self.hypernet_heads = []

        # TODO: adapt the following hypernet size calculation to pytorch at some point
        # input dim needs to be a multiple of 16 in tcnn model
        input_dim_16 = math.ceil(input_dim_hyper_mlp / 16) * 16
        input_layer_params = input_dim_16 * layer_width_hyper_mlp
        print(f"input_layer_params: {input_layer_params}")

        layers_head = 2
        width_head = 64
        head = MLP(
            in_dim=out_hypernet,
            num_layers=layers_head,
            layer_width=width_head,
            out_dim=input_layer_params,
            activation=nn.ReLU(),
            out_activation=None,
            implementation=implementation,
        ).to("cuda:0")

        self.hypernet_heads.append(head)

        # Iterate through hidden layers of target net; num_layers_color - 1 because it counts the output layer as well
        for i in range(num_layers_hyper_mlp - 1):
            if i == num_layers_hyper_mlp - 2:
                # Layer that connects to the output layer - rgb with 3 values - but also multiple of 16 in tcnn model
                hidden_layer_params = layer_width_hyper_mlp * 16
            else:
                hidden_layer_params = layer_width_hyper_mlp * layer_width_hyper_mlp
            print(f"hidden_layer_params: {hidden_layer_params}")

            # # Condition hypernet also on previous hypernets output
            # if i == 0:
            #     input_shapes = {'h': (features_dim + input_layer_params,)}
            # else:
            #     input_shapes = {'h': (features_dim + hidden_layer_params,)}
            # print(f"input shapes: {input_shapes}")

            head = MLP(
                in_dim=out_hypernet,
                num_layers=layers_head,
                layer_width=width_head,
                out_dim=hidden_layer_params,
                activation=nn.ReLU(),
                out_activation=None,
                implementation=implementation,
            )
            self.hypernet_heads.append(head)

        # Add variable that indicates whether the hypernetwork is active
        self.hypernetwork_active = False

        # Add variables that keeps track of the style img to use
        self.style_img = None
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

        for param in self.hyper_mlp_head.parameters():
            param.requires_grad = False
        return

    def reset_rgb(self):
        self.start_mlp_head.load_state_dict(self.test_mlp_head.state_dict())
        print("resetting start_mlp_head to untrained test_mlp_head")

    def deactivate_hypernetwork(self):
        # TODO: Implement
        return

    def update_mlp_head(self):
        # Prepare list to store the head outputs
        pred_params_list = []

        # Run base self.hypernet that will be input to the heads
        base_output = self.hypernet(self.style_features)

        for i in range(len(self.hypernet_heads)):
            pred_params = self.hypernet_heads[i](base_output)
            pred_params_list.append(pred_params)

        pred_params_tensor = torch.cat(pred_params_list, dim=-1)
        new_params = {"tcnn_encoding.params": pred_params_tensor.view(-1,)}
        return new_params
    
    def update_style_img(self, img_path):
        # Compute style features and move to GPU
        self.style_features = self.feature_extractor.get_hist(img_path).to("cuda:0")
        return

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

            # and then use the target network
            with self.hyper_mlp_head.using_externals(new_params):
                rgb = self.hyper_mlp_head(h).view(*outputs_shape, -1).to(directions)
        else:
            rgb = self.start_mlp_head(h).view(*outputs_shape, -1).to(directions)

        outputs.update({FieldHeadNames.RGB: rgb})

        return outputs

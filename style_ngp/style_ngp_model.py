"""
Template Model File

Currently this subclasses the Nerfacto model. Consider subclassing from the base Model.
"""
from dataclasses import dataclass, field
from typing import Type

from nerfstudio.models.instant_ngp import InstantNGPModelConfig, NGPModel # for subclassing InstantNGP model
from style_ngp.style_ngp_field import StyleNGPField

# import all the other necessary imports
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.model_components.ray_samplers import VolumetricSampler
from nerfstudio.model_components.renderers import AccumulationRenderer, DepthRenderer, RGBRenderer
from nerfstudio.model_components.losses import MSELoss

from torch.nn import Parameter
import nerfacc

@dataclass
class StyleNGPModelConfig(InstantNGPModelConfig):
    """Template Model Configuration.

    Add your custom model config parameters here.
    """

    _target: Type = field(default_factory=lambda: StyleNGPModel)


class StyleNGPModel(NGPModel):
    """Template Model."""

    config: StyleNGPModelConfig
    field: StyleNGPField
    # TODO: overwrite populate_modules to put custom fields in the right place via self.field
    # TODO: figure out how to switch hyper network on and off -> simple attribute?
    # TODO: overwrite get_outputs
    # TODO: overwrite get_metrics_dict
    # TODO: overwrite get_loss_dict
    # TODO: overwrite get_image_metrics_and_images

    def populate_modules(self):
        # Calling superclass
        super().populate_modules()

        # Rest adapted from instant-ngp
        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        self.field = StyleNGPField(
            aabb=self.scene_box.aabb,
            # TODO: is this correct? Why 0 if use?
            num_images=self.num_train_data,
            log2_hashmap_size=self.config.log2_hashmap_size,
            max_res=self.config.max_res,
        )

        self.scene_aabb = Parameter(self.scene_box.aabb.flatten(), requires_grad=False)

        if self.config.render_step_size is None:
            # auto step size: ~1000 samples in the base level grid
            self.config.render_step_size = ((self.scene_aabb[3:] - self.scene_aabb[:3]) ** 2).sum().sqrt().item() / 1000
        # Occupancy Grid.
        self.occupancy_grid = nerfacc.OccGridEstimator(
            roi_aabb=self.scene_aabb,
            resolution=self.config.grid_resolution,
            levels=self.config.grid_levels,
        )

        # Sampler
        self.sampler = VolumetricSampler(
            occupancy_grid=self.occupancy_grid,
            density_fn=self.field.density_fn,
        )

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method="expected")

        # losses
        self.rgb_loss = MSELoss()

        # metrics
        from torchmetrics.functional import structural_similarity_index_measure
        from torchmetrics.image import PeakSignalNoiseRatio
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

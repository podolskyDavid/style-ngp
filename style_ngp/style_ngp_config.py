"""
Nerfstudio Template Config

Define your custom method here that registers with Nerfstudio CLI.
"""

from __future__ import annotations

# from style_ngp.template_datamanager import (
#     TemplateDataManagerConfig,
# )
# from style_ngp.style_ngp_model import TemplateModelConfig
# from style_ngp.template_pipeline import (
#     TemplatePipelineConfig,
# )

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    ExponentialDecaySchedulerConfig,
)
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification

from nerfstudio.pipelines.dynamic_batch import DynamicBatchPipelineConfig
# from nerfstudio.models.instant_ngp import InstantNGPModelConfig
from style_ngp.style_ngp_model import StyleNGPModelConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig

from style_ngp.style_ngp_pipeline import StyleNGPPipelineConfig

style_ngp = MethodSpecification(
    config=TrainerConfig(
        method_name="style-ngp",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=15000,
        mixed_precision=True,
        pipeline=StyleNGPPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
            ),
            model=StyleNGPModelConfig(eval_num_rays_per_chunk=8192),
        ),
        optimizers={
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            }
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 12),
        vis="viewer",
    ),
    description="Instant-NGP adaptation for instant stylization via a hyper network.",
)

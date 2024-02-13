"""
Nerfstudio Template Pipeline
"""

import typing
from dataclasses import dataclass, field
from typing import Literal, Optional, Type

from torch.cuda.amp.grad_scaler import GradScaler

from nerfstudio.pipelines.dynamic_batch import DynamicBatchPipeline, DynamicBatchPipelineConfig


@dataclass
class StyleNGPPipelineConfig(DynamicBatchPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: StyleNGPPipeline)
    """target class to instantiate"""
    # datamanager: DataManagerConfig = TemplateDataManagerConfig()
    # """specifies the datamanager config"""
    # model: ModelConfig = TemplateModelConfig()
    # """specifies the model config"""


class StyleNGPPipeline(DynamicBatchPipeline):
    """StyleNGP Pipeline

    Args:
        config: the pipeline config used to instantiate class
    """

    def __init__(
        self,
        config: StyleNGPPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super().__init__(config, device, test_mode, world_size, local_rank, grad_scaler)

    def get_train_loss_dict(self, step: int):
        # # TODO: dirty hack, change that later
        # #  just for now to have hardcoded switch between training stages
        # if step == 500:
        #     self.model.field.activate_hypernetwork()
        #     # TODO: switch/modify the data loader

        model_outputs, loss_dict, metrics_dict = super().get_train_loss_dict(step)
        return model_outputs, loss_dict, metrics_dict

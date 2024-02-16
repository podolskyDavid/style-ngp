"""
Nerfstudio Template Pipeline
"""

import typing
from dataclasses import dataclass, field
from typing import Literal, Optional, Type

from torch.cuda.amp.grad_scaler import GradScaler

from nerfstudio.pipelines.dynamic_batch import DynamicBatchPipeline, DynamicBatchPipelineConfig

from pathlib import Path
import os


@dataclass
class StyleNGPPipelineConfig(DynamicBatchPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: StyleNGPPipeline)
    """target class to instantiate"""

    data_base_folder = "/home/maximilian_fehrentz/Documents/nerf_data/craniotomy/MICCAI/mesh_videos"

    def get_datasets(self):
        folder_names = [
            "mr",
            "207_065_cat5_1.5",
            "207_089_cat5_1.0",
            "207_101_sum_1.0",
            "207_103_cat5_1.0",
            "207_105_sum_1.0",
            "207_109_sum_0.5",
            "207_110_sum_0.75",
            "207_111_cat5_1.0",
            "207_112_cat5_1.0",
            "207_114_sum_0.75",
            "207_117_cat5_1.0",
            "207_201_sum_1.0",
            "207_202_cat5_2",
            "207_205_cat5_2",
            "207_207_sum_1.0",
            "207_209_sum_1.0"
        ]
        return [os.path.join(self.data_base_folder, x) for x in folder_names]


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
        self.datasets = self.config.get_datasets()
        self.i = 0
        self.structure_train_steps = 1000
        self.rgb_train_steps = 500

    def get_train_loss_dict(self, step: int):
        # TODO: dirty hack, change that later
        #  just for now to have hardcoded switch between training stages
        if step == self.structure_train_steps:
            self.model.field.activate_hypernetwork()
        if step == self.structure_train_steps or (step > self.structure_train_steps and step % self.rgb_train_steps == 0):
            # Determine a style name/identifier to save the weights
            if self.i != 0:
                # Use name of the data folder as model identifier; dataset i - 1 because i has been increased to move on
                style_name = self.datasets[self.i - 1].split("/")[-1]
            else:
                # Initial data set
                style_name = "initial_style"

            # Save model
            self.model.field.save_checkpoint(folder=self.config.data_base_folder, style=style_name)

            # Reset RGB net
            self.model.field.reset_rgb()

            # Move to next data set
            self.config.datamanager.data = Path(self.datasets[self.i])
            self.datamanager = self.config.datamanager.setup(
                device='cuda:0', test_mode=self.test_mode, world_size=1, local_rank=0
            )

            # Keep going through all styles
            if self.i == len(self.datasets) - 1:
                raise ValueError("All datasets have been trained on")
            self.i += 1

        model_outputs, loss_dict, metrics_dict = super().get_train_loss_dict(step)
        return model_outputs, loss_dict, metrics_dict

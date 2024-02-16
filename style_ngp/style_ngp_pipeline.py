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

    base_data_dir = "/home/maximilian_fehrentz/Documents/nerf_data/craniotomy/MICCAI/mesh_videos"
    style_dir = "/home/maximilian_fehrentz/Documents/nerf_data/craniotomy/MICCAI/data/styles"

    def get_datasets(self):
        data_dirs = [
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
        style_names = [
            "case065.png",
            "case089.png",
            "case101.png",
            "case103.png",
            "case105.png",
            "case109.png",
            "case110.png",
            "case111.png",
            "case112.png",
            "case114.png",
            "case117.png",
            "case201.png",
            "case202.png",
            "case205.png",
            "case207.png",
            "case209.png"
        ]
        return [{
            "data_folder": os.path.join(self.base_data_dir, data_dir),
            "style_img": os.path.join(self.style_dir, style_name)
        } for data_dir, style_name in zip(data_dirs, style_names)
        ]



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
        self.rgb_train_steps = 200

    def get_train_loss_dict(self, step: int):
        # Activate hypernetwork after some training on initial data set
        if step == self.structure_train_steps:
            self.model.field.activate_hypernetwork()

        if step == self.structure_train_steps or (step > self.structure_train_steps and step % self.rgb_train_steps == 0):
            # Determine a style name/identifier to save the weights
            if self.i != 0:
                # Use name of the style image as model identifier; dataset i - 1 to account for prev increase
                style_name = os.path.splitext(self.datasets[self.i - 1]["style_img"])[0]
            else:
                # Initial data set
                style_name = "initial_style"

            # # Save model
            # self.model.field.save_checkpoint(folder=self.config.base_data_dir, style=style_name)

            # # Reset RGB net
            # self.model.field.reset_rgb()

            # Move to next data set
            self.config.datamanager.data = Path(self.datasets[self.i]["data_folder"])
            self.datamanager = self.config.datamanager.setup(
                device='cuda:0', test_mode=self.test_mode, world_size=1, local_rank=0
            )

            # Set the corresponding style image
            style_img_path = self.datasets[self.i]["style_img"]
            self.model.field.update_style_img(style_img_path)

            # Keep cycling through all styles
            if self.i == len(self.datasets) - 1:
                print("All datasets have been trained on. Next epoch")
                self.i = 0
            else:
                self.i += 1

        model_outputs, loss_dict, metrics_dict = super().get_train_loss_dict(step)
        return model_outputs, loss_dict, metrics_dict

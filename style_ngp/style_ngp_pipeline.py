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

from nerfstudio.viewer.viewer_elements import ViewerDropdown


import numpy as np


@dataclass
class StyleNGPPipelineConfig(DynamicBatchPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: StyleNGPPipeline)
    """target class to instantiate"""

    base_data_dir = "/home/maximilian_fehrentz/Documents/MICCAI/207/data/datasets"
    test_style_dir = "/home/maximilian_fehrentz/Documents/MICCAI/styles"
    style_dir = "/home/maximilian_fehrentz/Documents/MICCAI/styles"

    # TODO: manually switching between train and test; fix that and automatically set based on ns-train or ns-viewer

    train_data_dirs = [
        # "207_065_cat5_2",
        # "207_089_cat5_2",
        "207_101_cat5_2",
        "207_103_cat5_2",
        # "207_105_cat5_2",
        "207_109_cat5_2",
        "207_110_cat5_2",
        "207_111_cat5_2",
        "207_112_cat5_2",
        "207_114_cat5_2",
        "207_117_cat5_2",
        "207_201_cat5_2",
        "207_202_cat5_2",
        "207_205_cat5_2",
        # "207_207_cat5_2",
        # "207_209_cat5_2",
    ]

    train_style_names = [
        # "case065.png",
        # "case089_crop2.png",
        "case101_crop1.png",
        "case103_crop1.png",
        # "case105_crop1.png",
        "case109_crop1.png",
        "case110_crop1.png",
        "case111_crop1.png",
        "case112_crop1.png",
        "case114_crop1.png",
        "case117_crop1.png",
        "case201_crop1.png",
        "case202_crop1.png",
        "case205.png",
        # "case207_crop1.png",
        # "case209_crop1.png",
    ]

    test_style_names = [
        "case065.png",
    ]

    def get_train_datasets(self):
        return [{
            "data_folder": os.path.join(self.base_data_dir, data_dir),
            "style_img": os.path.join(self.style_dir, style_name)
        } for data_dir, style_name in zip(self.train_data_dirs, self.train_style_names)
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

        # TODO: remove, make this dependent somehow on whether ns-train or ns-viewer/ns-render is used
        self.train_styles = False

        if self.train_styles:
            self.datasets = self.config.get_train_datasets()
        else:
            self.datasets = None

        if self.train_styles:
            self.style_dropdown = ViewerDropdown(
                name="Style",
                default_value=self.config.train_style_names[0],
                options=self.config.train_style_names,
                cb_hook=self.on_style_dropdown_change,
            )
        else:
            self.style_dropdown = ViewerDropdown(
                name="Style",
                default_value=self.config.test_style_names[0],
                options=self.config.test_style_names,
                cb_hook=self.on_style_dropdown_change,
            )
        self.current_style_index = 0

        # Used for MICCAI
        self.structure_train_steps = 500

        # Used for MICCAI
        self.rgb_train_steps = 5

        # TODO: very hacky solution, fix this; using this so that ns-render can "set" the style image during inference
        if not self.train_styles:
            self.change_style_from_dropdown(self.style_dropdown.value)

    def on_style_dropdown_change(self, handle: ViewerDropdown) -> None:
        self.change_style_from_dropdown(handle.value)

    def change_style_from_dropdown(self, style_name):
        # Style dropdown will only be used during inference, therefore activate hypernetwork
        if not self.model.field.hypernetwork_active:
            self.model.field.activate_hypernetwork()

        # Set the style image
        print(f"new style_name: {style_name}")
        if self.train_styles:
            self.model.field.update_style_img(os.path.join(self.config.style_dir, style_name))
        else:
            self.model.field.update_style_img(os.path.join(self.config.test_style_dir, style_name))


    def get_train_loss_dict(self, step: int):
        # Activate hypernetwork after some training on initial data set
        if step == self.structure_train_steps:
            self.model.field.activate_hypernetwork()

        if step == self.structure_train_steps or \
                (step > self.structure_train_steps and step % self.rgb_train_steps == 0):
            # Generate random index for next dataset
            self.current_style_index = np.random.randint(len(self.datasets))

            # Set the corresponding data dir
            self.config.datamanager.data = Path(self.datasets[self.current_style_index]["data_folder"])
            self.datamanager = self.config.datamanager.setup(
                device='cuda:0', test_mode=self.test_mode, world_size=1, local_rank=0
            )

            # Set the corresponding style image
            style_img_path = self.datasets[self.current_style_index]["style_img"]
            self.model.field.update_style_img(style_img_path)

        model_outputs, loss_dict, metrics_dict = super().get_train_loss_dict(step)
        return model_outputs, loss_dict, metrics_dict

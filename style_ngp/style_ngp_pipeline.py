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

# from style_ngp.util import compute_gram_matrix
#
# import math


@dataclass
class StyleNGPPipelineConfig(DynamicBatchPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: StyleNGPPipeline)
    """target class to instantiate"""

    base_data_dir = "/home/maximilian_fehrentz/Documents/nerf_data/craniotomy/MICCAI/mesh_videos"
    style_dir = "/home/maximilian_fehrentz/Documents/nerf_data/craniotomy/MICCAI/data/styles"

    data_dirs = [
        "207_065_cat5_1.5",
        "207_089_cat5_1.0",
        "207_101_sum_1.0",
        # "207_103_cat5_1.0",
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
        # "207_207_sum_1.0",
        "207_209_sum_1.0"
    ]
    style_names = [
        "case065.png",
        "case089_crop2.png",
        "case101_crop1.png",
        # "case103_crop1.png",
        "case105_crop1.png",
        "case109_crop1.png",
        "case110_crop1.png",
        "case111.png",
        "case112_crop1.png",
        "case114_crop1.png",
        "case117_crop1.png",
        "case201.png",
        "case202_crop1.png",
        "case205.png",
        # "case207.png",
        "case209_crop1.png"
    ]

    def get_datasets(self):
        return [{
            "data_folder": os.path.join(self.base_data_dir, data_dir),
            "style_img": os.path.join(self.style_dir, style_name)
        } for data_dir, style_name in zip(self.data_dirs, self.style_names)
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
        self.style_dropdown = ViewerDropdown(
            name="Style",
            default_value="case103_crop1.png",
            options=self.config.style_names,
            cb_hook=self.on_style_dropdown_change,
        )
        self.i = 0
        self.structure_train_steps = 400
        self.rgb_train_steps = 50

        # grams = []
        # for style_img in self.config.style_names:
        #     self.model.field.update_style_img(os.path.join(self.config.style_dir, style_img))
        #     grams.append(self.model.field.style_features)
        # # Compute pairwise differences between all elements in gram and store in a 2d array
        # self.gram_diffs = []
        # for i in range(len(grams)):
        #     diffs = []
        #     for j in range(len(grams)):
        #         diff = math.fabs((grams[i] - grams[j]).mean().cpu().numpy().tolist())
        #         if diff < 0.05:
        #             print(f"Style {self.config.style_names[i]} and {self.config.style_names[j]} are very similar")
        #         diffs.append(diff)
        #     self.gram_diffs.append(diffs)
        # print(f"self.gram_diffs: {self.gram_diffs}")


    def on_style_dropdown_change(self, handle: ViewerDropdown) -> None:
        # Style dropdown will only be used during inference, therefore activate hypernetwork
        if not self.model.field.hypernetwork_active:
            self.model.field.activate_hypernetwork()
        self.model.field.update_style_img(os.path.join(self.config.style_dir, handle.value))

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

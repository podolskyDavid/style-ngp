# TODO: fix this script, hardcoded based on my textures

import json
import pyvista as pv
import numpy as np
import argparse
import cv2
from util import ctw_to_wtc, add_noise_to_transform_matrix
import os


class TargetPlotter:
    def __init__(self, args):
        self.args = args

        # Initialize the plotter
        self.plotter = pv.Plotter(off_screen=False)

        # Set size of the images
        self.plotter.window_size = [self.args.width, self.args.height]

        # Make background black
        self.plotter.set_background('black')

        # Load the mesh
        self.mesh = pv.read(args.mesh_path)

        # Load the texture
        self.original_texture = pv.read_texture(args.original_tex)

        # Apply texture to the mesh and add it to the plotter
        self.plotter.add_mesh(self.mesh, texture=self.original_texture, lighting=False)

        self.i = 0

        # Setup JSON file that tracks a list of targets
        self.targets = {'065': {'targets': []},
                        '089': {'targets': []},
                        '105': {'targets': []},
                        '207': {'targets': []},
                        '209': {'targets': []},
                        }

        # Setup camera capture
        self.plotter.add_key_event('a', self.capture_scene)  # Press 'a' to save screenshot and camera pose

    def capture_scene(self):
        if self.i < self.args.num_targets:
            # Get camera
            camera = self.plotter.camera

            # Get vtk transform
            target_matrix = camera.GetModelViewTransformMatrix()
            # Convert vtk 4x4 matrix to nested list
            target_matrix = np.array(
                [[target_matrix.GetElement(i, j) for j in range(4)] for i in range(4)]
            )
            # Bring to world-to-camera
            target_matrix = ctw_to_wtc(target_matrix)

            # Do perturbation on gt for init pose
            init_matrix = add_noise_to_transform_matrix(target_matrix, self.args.rot_range, self.args.trans_range)

            gt_matrix = target_matrix.tolist()
            init_matrix = init_matrix.tolist()

            # Save screenshot, using every texture provided
            # TODO: hardcoded for now, fix later
            styles = ['065',
                      '089',
                      '105',
                      '207',
                      '209',
                      ]
            for style in styles:
                # Load texture, apply to mesh, screenshot, then clear plotter
                texture = pv.read_texture(f'textures/0_{style}_cat5_2.0.png')
                self.plotter.clear()
                self.plotter.add_mesh(self.mesh, texture=texture, lighting=False)

                # Make sure that dirs exist
                if not os.path.isdir(self.args.output_folder):
                    os.mkdir(self.args.output_folder)
                if not os.path.isdir(f'{self.args.output_folder}/{style}'):
                    os.mkdir(f'{self.args.output_folder}/{style}')

                # Create screenshot
                screenshot_path = f'{self.args.output_folder}/{style}/{self.i}.png'
                self.plotter.screenshot(screenshot_path)

                pose_information = {'target_pose': gt_matrix,
                                    'init_pose': init_matrix,
                                    'target_img_path': screenshot_path,
                                    }

                self.targets[style]['targets'].append(pose_information)

            self.i += 1

        else:
            print("All targets generated!")
            print(f"targets: {self.targets}")
            for style in self.targets.keys():
                output_path = f'{self.args.output_folder}/{style}_targets.json'
                with open(output_path, 'w') as f:
                    json.dump(self.targets[style], f, indent=4)
                print(f"Targets saved at {output_path}")
            self.plotter.close()

    def show(self):
        self.plotter.show()


def argparser():
    parser = argparse.ArgumentParser(description="Generate targets and inits for evaluation.")
    parser.add_argument('--mesh_path', type=str, required=True, help='Path to the mesh file')
    parser.add_argument('--original_tex', type=str, required=True, help='Path to the some texture file used for visual help')
    parser.add_argument('--num_targets', type=int, default=50, help='Number of targets to generate')
    parser.add_argument('--rot_range', type=float, default=15, help='Range of rotation in degrees')
    parser.add_argument('--trans_range', type=float, default=5, help='Range of translation in mm')
    parser.add_argument('--width', type=int, default=512, help='Width of the image')
    parser.add_argument('--height', type=int, default=512, help='Height of the image')
    parser.add_argument('--output_folder', type=str, default='synth_targets', help='Folder to save the targets and inits')
    return parser.parse_args()


# Important: assuming that you are in the data folder!
if __name__ == '__main__':
    args = argparser()
    plotter = TargetPlotter(args)
    plotter.show()

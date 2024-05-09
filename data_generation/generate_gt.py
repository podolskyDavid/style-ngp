import json
import pyvista as pv
import numpy as np
import argparse
import cv2
from utils import apply_padding


class GtPlotter:
    def __init__(self, args):
        # Load the background image to get its dimensions
        background_img = cv2.imread(args.background_img_path)
        height, width, _ = background_img.shape

        # Initialize the plotter with the background image size
        self.plotter = pv.Plotter(off_screen=False)
        self.plotter.window_size = [width, height]

        # Load and potentially pad the original texture
        if args.tex_size:
            # TODO: right now for png only, fix later!
            padded_tex_path = args.original_texture_path.replace('.png', '_padded.png')
            apply_padding(args.original_texture_path, padded_tex_path, args.tex_size, args.roi_size, args.roi_pos)
            texture_path = padded_tex_path
        else:
            texture_path = args.original_texture_path

        self.original_texture = pv.read_texture(texture_path)

        # Load the mesh and make it semi-transparent
        self.mesh = pv.read(args.mesh_path)
        self.mesh_opacity = 0.75  # Adjust the transparency level

        # Load the background image as texture and add it to the plotter
        # self.background_texture = pv.read_texture(args.background_img_path)
        self.plotter.add_background_image(args.background_img_path, scale=True)

        # Apply texture to the mesh and add it to the plotter
        self.plotter.add_mesh(self.mesh, texture=self.original_texture, opacity=self.mesh_opacity)

        # Setup camera capture
        self.plotter.add_key_event('a', self.capture_scene)  # Press 'a' to save screenshot and camera pose

    def capture_scene(self):
        screenshot_path = 'registration.png'
        camera_pose_path = 'gt_camera_pose.json'

        # Save screenshot
        self.plotter.screenshot(screenshot_path)

        # Get camera
        camera = self.plotter.camera

        # Get vtk transform
        transform_matrix = camera.GetModelViewTransformMatrix()

        # Convert vtk 4x4 matrix to nested list
        transform_matrix = np.array(
            [[transform_matrix.GetElement(i, j) for j in range(4)] for i in range(4)]
        )

        # Compute transpose(R) and -transpose(R) * T
        # Syntax below: R.T is the transpose of R and @ is matrix multiplication
        R = transform_matrix[:3, :3]
        T = transform_matrix[:3, 3]
        transform_matrix[:3, 3] = -R.T @ T
        transform_matrix[:3, :3] = R.T
        transform_matrix = transform_matrix.tolist()

        pose_json = {'transform_matrix': transform_matrix}

        # Save pose
        with open(camera_pose_path, 'w') as f:
            json.dump(pose_json, f)

        print(f"Screenshot and camera pose saved at '{screenshot_path}' and '{camera_pose_path}'")

    def show(self):
        self.plotter.show()


# Note that tex_size, roi_size, and roi_pos only come in when the mesh is larger than the texture images
#  In those cases, the texture images are padded to the size of tex_size,
#  and the region of interest is specified by roi_size and roi_pos
def argparser():
    parser = argparse.ArgumentParser(description="Overlay a mesh on a background and capture scenes.")
    parser.add_argument('--mesh_path', type=str, required=True, help='Path to the mesh file')
    parser.add_argument('--background_img_path', type=str, required=True, help='Path to the background image')
    parser.add_argument('--original_texture_path', type=str, required=True, help='Path to the original texture')
    parser.add_argument('--tex_size', type=int, nargs=2, default=None, help='Size of the texture')
    parser.add_argument('--roi_size', type=int, nargs=2, default=None, help='Size of the region of interest')
    parser.add_argument('--roi_pos', type=int, nargs=2, default=None, help='Position of the region of interest')
    return parser.parse_args()


if __name__ == '__main__':
    args = argparser()
    plotter = GtPlotter(args)
    plotter.show()

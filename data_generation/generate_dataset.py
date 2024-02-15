import json
import pyvista as pv
import os
import shutil
import math
import numpy as np
import argparse


class HappyPlotter:
    def __init__(self, num_poses, original_tex_path, background='black', width=512, height=512):
        # Create a plotter
        self.plotter = pv.Plotter(off_screen=False)

        self.plotter.set_background(background)

        # Determines the resolution of the generated images (unless the window is resized manually!)
        self.image_width = width
        self.image_height = height
        self.plotter.window_size = [self.image_width, self.image_height]

        # Original texture to apply to the mesh
        self.original_tex_path = original_tex_path
        tex = pv.read_texture(self.original_tex_path)

        # Add the mesh to the plotter
        self.plotter.add_mesh(mesh, texture=tex, lighting=False)

        # Keeping track of cameras; storing full vtk cameras in there later
        self.cameras = []

        # Add key event that will trigger getting the camera pose and saving the image
        self.plotter.add_key_event("a", self.get_camera)

        # number of poses to generate
        self.num_poses = num_poses
        # tracking; property because used in callback
        self.i = 0


    # Triggered by key event
    def get_camera(self):
        output_path = f'original_tex/{self.i}.png'
        self.i += 1

        print(f"saving to output_path: {output_path}")
        self.plotter.screenshot(output_path)
        new_camera = pv.Camera()
        new_camera.DeepCopy(self.plotter.camera)
        self.cameras.append(new_camera)
        return

    def create_cameras(self):
        # Create folder to save images of original textures
        if os.path.isdir('original_tex'):
            shutil.rmtree('original_tex')
        os.mkdir('original_tex')

        # Show the plotter
        self.plotter.show(auto_close=False, interactive_update=True)

        # Keep rendering and listening to key events
        while self.i < self.num_poses:
            self.plotter.update()
        return

    def mesh_it(self, target_folder):
        # Show the plotter
        self.plotter.show(auto_close=False, interactive_update=True)

        all_styles = os.listdir(target_folder)
        # Add original texture as well
        # TODO: regenerating original texture is redundant
        all_styles.append(self.original_tex_path)

        for file in os.listdir(target_folder):
            if file.endswith(".png"):
                # Get texture
                tex = pv.read_texture(os.path.join(target_folder, file))

                # Apply texture to mesh
                self.plotter.add_mesh(mesh, texture=tex, lighting=False)

                # Split the filename from the file extension
                filename, _ = os.path.splitext(file)

                # Create a folder for the rendered images
                if os.path.isdir(f'{filename}'):
                    shutil.rmtree(f'{filename}')
                os.mkdir(f'{filename}')
                os.mkdir(f'{filename}/images')

                frames = []

                # Go through all the camera positions and regenerate the screenshots
                for i, camera in enumerate(self.cameras):
                    # # Set position of the camera
                    # self.plotter.camera.SetPosition(position)
                    # self.plotter.camera.SetViewUp(view_up)

                    self.plotter.camera = camera
                    self.plotter.update()

                    output_path = f'{filename}/images/{i}.png'
                    self.plotter.screenshot(output_path)

                    transform_matrix = camera.GetModelViewTransformMatrix()
                    # Convert vtk 4x4 matrix to nested list
                    transform_matrix = np.array(
                        [[transform_matrix.GetElement(i, j) for j in range(4)] for i in range(4)]
                    )

                    # It looks like something needs to be done additionally with the transform matrix
                    # See
                    #  https://github.com/NVlabs/instant-ngp/issues/1360
                    #  https://github.com/colmap/colmap/issues/434
                    #  https://github.com/NVlabs/instant-ngp/discussions/153 (most promising)

                    # Compute transpose(R) and -transpose(R) * T
                    # Syntax below: R.T is the transpose of R and @ is matrix multiplication
                    R = transform_matrix[:3, :3]
                    T = transform_matrix[:3, 3]
                    transform_matrix[:3, 3] = -R.T @ T
                    transform_matrix[:3, :3] = R.T

                    transform_matrix = transform_matrix.tolist()

                    frames.append(
                        {
                            'file_path': os.path.join('./images', f'{i}.png'),
                            'transform_matrix': transform_matrix
                        }
                    )

                # Some of below information is based on https://gist.github.com/decrispell/fc4b69f6bedf07a3425b
                # Get vertical (default) and horizontal field of view
                fov_y = math.radians(self.plotter.camera.GetViewAngle())
                self.plotter.camera.UseHorizontalViewAngleOn()
                fov_x = math.radians(self.plotter.camera.GetViewAngle())
                self.plotter.camera.UseHorizontalViewAngleOff()

                fl_x = (self.image_width * 0.5) / math.tan(fov_x * 0.5)
                fl_y = (self.image_height * 0.5) / math.tan(fov_y * 0.5)

                transforms = {
                    'camera_angle_x': fov_x,
                    'camera_angle_y': fov_y,
                    'fl_x': fl_x,
                    'fl_y': fl_y,
                    'k1': 0,
                    'k2': 0,
                    'k3': 0,
                    'k4': 0,
                    'p1': 0,
                    'p2': 0,
                    'is_fisheye': False,
                    'cx': self.image_width / 2,
                    'cy': self.image_height / 2,
                    'w': self.image_width,
                    'h': self.image_height,
                    'aabb_scale': 32,
                    'frames': frames
                }

                json.dump(transforms, open(f'{filename}/transforms.json', 'w'))

        self.plotter.close()
        return


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh_path', type=str, help='Path to mesh file')
    parser.add_argument('--target_folder_path', type=str, help='Path to folder with textures')
    parser.add_argument('--original_texture_path', type=str, help='Path to original texture')
    parser.add_argument('--num_poses', type=int, default=100, help='Number of poses to generate')
    parser.add_argument('--width', type=int, default=512, help='Width of the images to be generated')
    parser.add_argument('--height', type=int, default=512, help='Height of the images to be generated')
    return parser.parse_args()


if __name__ == '__main__':
    # Get args
    args = argparser()

    # Load mesh (tested with obj file)
    mesh = pv.read(args.mesh_path)

    my_plotter = HappyPlotter(
        num_poses=args.num_poses,
        original_tex_path=args.original_texture_path,
        width=args.width,
        height=args.height
    )
    my_plotter.create_cameras()
    my_plotter.mesh_it(args.target_folder_path)

import argparse
import pyvista as pv
import os
import shutil

from utils import reverse_transform

import numpy as np
import cv2
import json

def get_argparser():
    argparser = argparse.ArgumentParser()

    # Our case: brain surface
    argparser.add_argument('--tex_path', type=str, required=True, help='Path to the texture file')
    argparser.add_argument('--mesh_path', type=str, required=True, help='Path to the mesh file')

    # When we created the NeRF dataset, the mesh of the brain surface was centered; this is the centered mesh
    argparser.add_argument('--centered_mesh_path', type=str, required=True, help='Path to the centered mesh file')

    # Our case: full brain volume
    argparser.add_argument('--full_mesh_path', type=str, required=True, help='Path to the full mesh file')

    # Contains the results from Parallel Inversion
    argparser.add_argument('--record_file', type=str, required=True, help='Path to the record JSON file')

    # Contains information about the dataset-specific scaling ngp scaling and translation the was applied to the poses
    #  Usually found in the ngp ground truth files or in the synthetic targets file as top level keys
    argparser.add_argument('--transform_params_file', type=str, required=True, help='Path to the transformation parameters JSON file')

    # Usually Parallel Inversion results contain several targets, this is the index of the pose we want to visualize
    argparser.add_argument('--index', type=int, required=True, help='Index of the pose in the record file')
    return argparser


def load_and_transform_poses(record_file, transform_params_file, index, translation):
    # Load pose log
    with open(record_file, 'r') as f:
        record_data = json.load(f)
    poses = record_data['results'][index]['pose_log']
    start = record_data['results'][index]['start_pose']
    gt = record_data['results'][index]['gt_pose']

    # Since pose_log is a 4x4 coming from Parallel Inversion under NVIDIA ngp convention, we need to reverse the
    #  dataset-specific translation and scaling that ngp does and then go from w2c that NeRF (not NVIDIA ngp environment
    #  e.g. Nerfstudio) uses to c2w to comply with VTK convention

    # Load ngp dataset-specific transforms
    with open(transform_params_file, 'r') as f:
        transform_params = json.load(f)
    centroid = transform_params['centroid']
    avglen = transform_params['avglen']
    totp = transform_params['totp']

    c2w_poses = []
    for pose in poses:
        # Poses in Parallel Inversion are only logged as 3x4 to save space, add last row
        pose = np.vstack([pose, [0, 0, 0, 1]])

        # Go from NVIDIA ngp convention (c2w scaled) back to unscaled; staying in c2w of NeRF instead of converting to
        #  w2c of VTK because it is useful for visualizing the cameras in the world
        unscaled_pose = np.array((reverse_transform(pose, centroid, avglen, totp)))

        # Add translation due to centering the brain surface mesh at some point, need to reverse that
        unscaled_pose[:3, 3] += translation

        c2w_poses.append(unscaled_pose)

    c2w_start = np.array((reverse_transform(start, centroid, avglen, totp)))
    c2w_start[:3, 3] += translation
    c2w_gt = np.array((reverse_transform(gt, centroid, avglen, totp)))
    c2w_gt[:3, 3] += translation

    return c2w_poses, c2w_start, c2w_gt


# Shrinking the mesh slightly, otherwise patch on the surface is slightly occluded
def shrink2(m, s):
    c = np.array(m.center)
    mm = m.copy(deep=True)
    mm.translate(-c)
    mm.transform(np.array([
        [s,0,0,0],
        [0,s,0,0],
        [0,0,s,0],
        [0,0,0,1]
    ]))
    mm.translate(c)
    return mm

def main(args):
    cone_radius = 1
    cone_height = 3
    pv.global_theme.anti_aliasing = 'fxaa'
    plotter = pv.Plotter(off_screen=False, window_size=[1024, 1024])
    # # Set background of plotter
    plotter.set_background('black')

    mesh = pv.read(args.full_mesh_path)
    mesh = shrink2(mesh, .99)
    full_mesh_actor = plotter.add_mesh(mesh, smooth_shading=True, lighting=False, diffuse=1, opacity=1, style='wireframe')

    mesh = pv.read(args.mesh_path)
    tex = pv.read_texture(args.tex_path)
    mesh_actor = plotter.add_mesh(mesh, texture=tex, smooth_shading=True, lighting=False, diffuse=1, opacity=1)

    # Need to correct for shift we introduced by centering our mesh of the brain surface when creating the data
    centered_mesh = pv.read(args.centered_mesh_path)
    # Compute the translation between center of the centered mesh and the other mesh
    translation = np.array(mesh.center) - np.array(centered_mesh.center)

    # Load the pose log and start and ground truth from the record file
    pose_log, start, gt = load_and_transform_poses(args.record_file, args.transform_params_file, args.index, translation)

    # For the poses during optimization
    pts = [pose[:3, 3] for pose in pose_log]
    # Direction and up vector follow because each column in the rotation matrix represents where the unit vector of that
    #  axis would end up after the rotation; e.g, first column of R is where the x-axis would end up after the rotation.
    #  In VTK convention, the camera looks towards -Z and Y is up, that's why the second and (negative) third column
    #  represent the view-up and direction after rotation.
    dirs = [-pose[:3, 2] for pose in pose_log]
    ups = [pose[:3, 1] for pose in pose_log]

    # start_pt = start[:3, 3]
    # start_dir = -start[:3, 2]

    gt_pt = gt[:3, 3]
    gt_dir = -gt[:3, 2]

    start_cone = pv.Cone(pts[0], -dirs[0], radius=cone_radius, height=cone_height, resolution=4, capping=True)
    # start_cone = pv.Cone(start_pt, start_dir, radius=cone_radius, height=cone_height, resolution=4, capping=True)
    start_cone_actor = plotter.add_mesh(start_cone, color='red', style='wireframe', line_width=4)

    gt_cone = pv.Cone(gt_pt, -gt_dir, radius=cone_radius, height=cone_height, resolution=4, capping=True)
    gt_cone_actor = plotter.add_mesh(gt_cone, color='green', style='wireframe', line_width=4)

    def create_trajectory():
        print("Trajectory started!")

        for p in pts:
            s = pv.Sphere(radius=0.1)
            s = s.translate(p)
            plotter.add_mesh(s, color='red', line_width=4)

        for idx in range(len(pts) - 1):
            ln = pv.Line(pts[idx], pts[idx + 1])
            plotter.add_mesh(ln, color='cyan', line_width=2, render_lines_as_tubes=True)

        # Create folder
        trajectory_dir = '../vis/trajectory_snapshots'
        if os.path.exists(trajectory_dir):
            shutil.rmtree(trajectory_dir)
        os.makedirs(trajectory_dir)

        for i, (p, dir) in enumerate(zip(pts, dirs)):
            a = pv.Cone(p, -dir, radius=cone_radius, height=cone_height, resolution=4, capping=True)
            actor = plotter.add_mesh(a, color='white', show_edges=True, style='wireframe', line_width=4)
            img_path = os.path.join(trajectory_dir, f'pose_{i:03d}.png')
            plotter.screenshot(img_path)
            plotter.remove_actor(actor)

    def create_screenshots():
        print("Rendering started!")

        # Determines the resolution of the generated images (unless the window is resized manually!)
        plotter.window_size = [512, 512]

        # Remove the big mesh
        plotter.remove_actor(full_mesh_actor)
        # Remove the start and gt cone
        plotter.remove_actor(start_cone_actor)
        plotter.remove_actor(gt_cone_actor)

        snapshots_dir = '../vis/mr_snapshots'
        snapshots_dir_with_target = '../vis/mr_snapshots_with_target'
        # Create folders, remove if necessary
        if os.path.exists(snapshots_dir):
            shutil.rmtree(snapshots_dir)
        os.makedirs(snapshots_dir)
        if os.path.exists(snapshots_dir_with_target):
            shutil.rmtree(snapshots_dir_with_target)
        os.makedirs(snapshots_dir_with_target)

        for i, (p, dir) in enumerate(zip(pts, dirs)):
            # Create a snapshot
            camera = plotter.camera
            camera.SetPosition(pts[i])
            camera.SetFocalPoint(pts[i] + dirs[i])
            camera.SetViewUp(ups[i])
            plotter.camera = camera
            plotter.update()
            img_path = os.path.join(snapshots_dir, f'pose_{i:03d}.png')
            plotter.screenshot(img_path)

            # Load the screenshot again and put a target png with transparency in the background
            img = cv2.imread(img_path)
            target = cv2.imread('target.png', cv2.IMREAD_UNCHANGED)
            target = cv2.resize(target, (img.shape[1], img.shape[0]))
            img = cv2.addWeighted(img, 1, target, 0.25, 0)
            img_path_with_target = os.path.join(snapshots_dir_with_target, f'pose_{i:03d}.png')
            cv2.imwrite(img_path_with_target, img)

    plotter.add_key_event(key="t", callback=create_trajectory)
    plotter.add_key_event(key="s", callback=create_screenshots)

    plotter.show()
    plotter.close()


if __name__ == "__main__":
    args = get_argparser().parse_args()
    main(args)

import json
import numpy as np
import pyvista as pv
import argparse
from utils import reverse_transform, c2w_w2c


def compute_add_loss(mesh, start_pose, gt_pose, pred_pose):
    # start_transformed_vertices = mesh.points @ start_pose[:3, :3] + start_pose[:3, 3]
    # gt_transformed_vertices = mesh.points @ gt_pose[:3, :3] + gt_pose[:3, 3]
    # pred_transformed_vertices = mesh.points @ pred_pose[:3, :3] + pred_pose[:3, 3]

    print(f"start_pose: {start_pose}")

    start_transformed_vertices = []
    gt_transformed_vertices = []
    pred_transformed_vertices = []
    for point in mesh.points:
        start_point = start_pose[:3, :3] @ point + start_pose[:3, 3]
        gt_point = gt_pose[:3, :3] @ point + gt_pose[:3, 3]
        pred_point = pred_pose[:3, :3] @ point + pred_pose[:3, 3]
        start_transformed_vertices.append(start_point)
        gt_transformed_vertices.append(gt_point)
        pred_transformed_vertices.append(pred_point)
    start_transformed_vertices = np.array(start_transformed_vertices)
    gt_transformed_vertices = np.array(gt_transformed_vertices)
    pred_transformed_vertices = np.array(pred_transformed_vertices)

    # Compute ADD
    add_start = np.mean(np.linalg.norm(gt_transformed_vertices - start_transformed_vertices, axis=1))
    add_pred = np.mean(np.linalg.norm(gt_transformed_vertices - pred_transformed_vertices, axis=1))

    return add_start, add_pred


def visualize(mesh, start_pose, gt_pose, pred_pose):
    # Apply the transformations to the mesh
    gt_mesh = mesh.copy()
    gt_mesh.transform(gt_pose)

    pred_mesh = mesh.copy()
    pred_mesh.transform(pred_pose)

    # Create a PyVista plotter
    plotter = pv.Plotter()

    # Add the meshes to the plotter
    plotter.add_mesh(mesh, color='red', opacity=0.5)
    plotter.add_mesh(gt_mesh, color='green', opacity=0.5)
    plotter.add_mesh(pred_mesh, color='blue', opacity=0.5)

    # Add the cameras
    points = np.array(
        [-start_pose[:3, :3].T @ start_pose[:3, 3], -gt_pose[:3, :3].T @ gt_pose[:3, 3], -pred_pose[:3, :3].T @ pred_pose[:3, 3]]
    )
    labels = ['Start', 'Gt', 'Pred']
    actor = plotter.add_point_labels(
        points,
        labels,
        italic=True,
        font_size=20,
        point_color='red',
        point_size=20,
        render_points_as_spheres=True,
        always_visible=True,
        shadow=True,
    )

    # Add coordinate system
    plotter.add_axes_at_origin()

    # Display the plotter
    plotter.show()


def main(results_file, transform_params_file, mesh_file):
    # Read the JSON file with results
    with open(results_file, 'r') as file:
        results = json.load(file)

    # Read the JSON file with transformation parameters
    with open(transform_params_file, 'r') as file:
        transform_params = json.load(file)

    # Reverse the transformation process
    results = results["trial"][0]
    # start_pose = reverse_transform(results['start_pose'], transform_params['centroid'], transform_params['avglen'], transform_params['totp'])
    # gt_pose = reverse_transform(results['gt_pose'], transform_params['centroid'], transform_params['avglen'], transform_params['totp'])
    # pred_pose = reverse_transform(results['pred_pose'], transform_params['centroid'], transform_params['avglen'], transform_params['totp'])

    start_pose = c2w_w2c(np.array(reverse_transform(results['start_pose'], transform_params['centroid'], transform_params['avglen'], transform_params['totp'])))
    gt_pose = c2w_w2c(np.array(reverse_transform(results['gt_pose'], transform_params['centroid'], transform_params['avglen'], transform_params['totp'])))
    pred_pose = c2w_w2c(np.array(reverse_transform(results['pred_pose'], transform_params['centroid'], transform_params['avglen'], transform_params['totp'])))

    print(f"translation error: {np.linalg.norm(np.array(gt_pose)[:3, 3] - np.array(pred_pose)[:3, 3])}")

    # Read the obj file
    mesh = pv.read(mesh_file)

    print(f"diameter: {mesh.length}")

    # Compute the ADD loss
    add_start, add_pred = compute_add_loss(mesh, np.array(start_pose), np.array(gt_pose), np.array(pred_pose))

    print(f"ADD loss: {add_pred}")
    print(f"ADD of start pose: {add_start}")
    print(f"ADD to diameter: {add_pred / mesh.length}")

    # # Visualize the mesh and the cameras
    # visualize(mesh, np.array(start_pose), np.array(gt_pose), np.array(pred_pose))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute ADD loss")
    # Comes from Parallel Inversion
    parser.add_argument('--results_file', type=str, required=True, help='Path to the results JSON file')
    # Should be the VTK -> NGP transformed ground truth JSON, contains transform params between the 2 conventions
    parser.add_argument('--transform_params_file', type=str, required=True, help='Path to the transformation parameters JSON file')
    parser.add_argument('--mesh_file', type=str, required=True, help='Path to the mesh file')
    args = parser.parse_args()

    main(args.results_file, args.transform_params_file, args.mesh_file)
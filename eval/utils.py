import numpy as np


def reverse_transform(transform, centroid, avglen, totp):
    transform = np.array(transform)
    transform[:3, 3] += totp
    transform = switch_coord_system(transform)
    transform[:3, 3] /= 3. / avglen
    transform[:3, 3] += centroid
    return transform.tolist()


# Code adapted from https://github.com/NVlabs/instant-ngp/issues/658
def switch_coord_system(current_transform):
    # Assuming that I am in OpenGL with NerfStudio

    # Transformation matrix for coordinate system conversion to ngp
    transformation_matrix = np.array([[0, 1,  0, 0],
                                      [1, 0,  0, 0],
                                      [0, 0, -1, 0],
                                      [0, 0,  0, 1]])
    new_pose_matrix = transformation_matrix @ current_transform
    return new_pose_matrix


# VTK gives w2c and NeRF takes c2w in transforms.json, see https://github.com/NVlabs/instant-ngp/issues/61
def c2w_w2c(transform_matrix):
    # Explanation for modification of rotation and translation below:
    #  - transform coming from pyvista (underlying VTK) describes world-to-camera (w2c)
    #  - example: transform would map camera's world coordinates to 0,0,0 (local camera coord system)
    #  - for NeRF (and its transforms.json), we need camera-to-world
    #  - therefore: rotation needs to be inverted (since orthonormal matrix -> transpose) and camera needs to be
    #   translated back to world coordinates

    # Note that the method can go from c2w to w2c and vice versa, it's the same transformation

    # Compute transpose(R) and -transpose(R) * T
    # Syntax below: R.T is the transpose of R and @ is matrix multiplication
    R = transform_matrix[:3, :3]
    T = transform_matrix[:3, 3]
    transform_matrix[:3, 3] = -R.T @ T
    transform_matrix[:3, :3] = R.T

    return transform_matrix


# Taken from Parallel Inversion
rot_psi = lambda phi: np.array([
		[1, 0, 0, 0],
		[0, np.cos(phi), -np.sin(phi), 0],
		[0, np.sin(phi), np.cos(phi), 0],
		[0, 0, 0, 1]])


# Taken from Parallel Inversion
rot_theta = lambda th: np.array([
		[np.cos(th), 0, -np.sin(th), 0],
		[0, 1, 0, 0],
		[np.sin(th), 0, np.cos(th), 0],
		[0, 0, 0, 1]])


# Taken from Parallel Inversion
rot_phi = lambda psi: np.array([
		[np.cos(psi), -np.sin(psi), 0, 0],
		[np.sin(psi), np.cos(psi), 0, 0],
		[0, 0, 1, 0],
		[0, 0, 0, 1]])


# Taken from Parallel Inversion
trans_t = lambda t1, t2, t3: np.array([
		[1, 0, 0, t1],
		[0, 1, 0, t2],
		[0, 0, 1, t3],
		[0, 0, 0, 1]])


# Taken from Parallel Inversion
def add_noise_to_transform_matrix(transform_matrix, delta_rot_range, delta_trans_range):
    transform_matrix_noisy = transform_matrix.copy()

    delta_phi = np.random.uniform(-delta_rot_range, delta_rot_range)
    delta_theta = np.random.uniform(-delta_rot_range, delta_rot_range)
    delta_psi = np.random.uniform(-delta_rot_range, delta_rot_range)
    delta_tx = np.random.uniform(-delta_trans_range, delta_trans_range)
    delta_ty = np.random.uniform(-delta_trans_range, delta_trans_range)
    delta_tz = np.random.uniform(-delta_trans_range, delta_trans_range)

    # We have to decompose the transform matrix, do rotation first, then translation
    transform_matrix_temp_rot = np.eye(4)
    transform_matrix_temp_rot[:3, :3] = transform_matrix_noisy[:3, :3]
    transform_matrix_temp_trans = np.eye(4)
    transform_matrix_temp_trans[:3, 3] = transform_matrix_noisy[:3, 3]

    transform_matrix_noisy = trans_t(delta_tx, delta_ty, delta_tz) @ transform_matrix_temp_trans @ rot_phi(
		delta_phi / 180. * np.pi) @ rot_theta(delta_theta / 180. * np.pi) @ rot_psi(delta_psi / 180. * np.pi) @ transform_matrix_temp_rot

    return transform_matrix_noisy


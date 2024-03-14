import cv2
import numpy as np


def reverse_transform(transform, centroid, avglen, totp):
    transform = np.array(transform)
    transform[:3, 3] += totp
    transform = switch_coord_system(transform)
    transform[:3, 3] /= 3. / avglen
    transform[:3, 3] += centroid
    return transform.tolist()


def apply_padding(img_path, output_img_path, tex_size, roi_size, roi_pos):
    # Load the ROI image, which is your full image
    roi = cv2.imread(img_path)

    print(f"shape of roi: {roi.shape}")

    # Resize the roi to roi_size
    # TODO: this assumes that roi images are squared! Fix later
    roi = cv2.resize(roi, (roi_size[0], roi_size[1]))

    print(f"shape of roi: {roi.shape}")

    # Create a new black image with the texture size
    new_image = np.zeros((tex_size[1], tex_size[0], 3), dtype=np.uint8)

    # Calculate the ending positions based on the starting positions and the size of the ROI
    end_y = roi_pos[1] + roi.shape[0]
    end_x = roi_pos[0] + roi.shape[1]

    # Place the ROI into the new image at the specified position
    # This is done by specifying the range in the new image where the ROI should be placed
    new_image[roi_pos[1]:end_y, roi_pos[0]:end_x] = roi

    # Save the new image with padding to the specified output path
    cv2.imwrite(output_img_path, new_image)


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


def ctw_to_wtc(transform_matrix):
    # Explanation for modification of rotation and translation below:
    #  - transform coming from pyvista (underlying VTK) describes camera-to-world
    #  - example: transform would map camera origin to its world coordinates
    #  - for NeRF, we want world-to-camera, as everything comes down to: what does the camera see?
    #    instead of where is the camera in the world?
    #  - therefore: rotation needs to be inverted (since orthonormal matrix -> transpose) to align world
    #    with camera; now, instead of translating the camera to the world, we translate the world to the
    #    camera by using -translation, however, we need to apply the rotation to the translation as well

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


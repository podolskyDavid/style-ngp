import cv2
import numpy as np


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
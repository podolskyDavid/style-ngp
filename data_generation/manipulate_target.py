import cv2
import numpy as np
import argparse
import os
import glob


def hist_match(source, templates):
    oldshape = source.shape
    source = source.ravel()

    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True, return_counts=True)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]

    avg_t_counts = np.zeros_like(s_counts, dtype=np.float64)
    for template in templates:
        template = template.ravel()
        t_values, t_counts = np.unique(template, return_counts=True)
        t_quantiles = np.cumsum(t_counts).astype(np.float64)
        t_quantiles /= t_quantiles[-1]
        interp_t_counts = np.interp(s_values, t_values, t_counts, left=0, right=0)
        avg_t_counts += interp_t_counts
    avg_t_counts /= len(templates)

    avg_t_quantiles = np.cumsum(avg_t_counts).astype(np.float64)
    avg_t_quantiles /= avg_t_quantiles[-1]

    interp_t_values = np.interp(s_quantiles, avg_t_quantiles, s_values)
    return interp_t_values[bin_idx].reshape(oldshape).astype(source.dtype)


def main():
    parser = argparse.ArgumentParser(description='Adjust the color distribution of target to match reference.')
    parser.add_argument('--reference_dir', help='Reference image directory path.')
    parser.add_argument('--target', help='Target image file path.')
    parser.add_argument('--exclude', type=str, default=None, help='Exclude images from reference directory.')
    args = parser.parse_args()

    # Get all image file paths in the reference directory
    ref_img_paths = glob.glob(os.path.join(args.reference_dir, '*'))

    # Read the reference images; potentially exclude something
    if args.exclude is not None:
        for ref in ref_img_paths:
            if args.exclude in ref:
                print(f"excluding {ref}")
                ref_img_paths.remove(ref)

    ref_imgs = [cv2.imread(ref, cv2.IMREAD_COLOR) for ref in ref_img_paths]
    target_img = cv2.imread(args.target, cv2.IMREAD_COLOR)

    matched_img = hist_match(target_img, ref_imgs)

    base_name, ext = os.path.splitext(args.target)
    adjusted_filename = f"{base_name}_adjusted{ext}"
    cv2.imwrite(adjusted_filename, matched_img)


if __name__ == "__main__":
    main()

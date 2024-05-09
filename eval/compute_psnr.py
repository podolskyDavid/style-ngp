import cv2
import os
import numpy as np
import argparse

def compute_psnr(img1, img2):
    # Compute PSNR between two images
    return cv2.PSNR(img1, img2)

def get_argparser():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--folder1', type=str, required=True, help='Path to the first folder')
    argparser.add_argument('--folder2', type=str, required=True, help='Path to the second folder')
    return argparser

if __name__ == "__main__":
    # Get args
    args = get_argparser().parse_args()

    # Get a list of all image files in the two directories
    img_files1 = [os.path.join(args.folder1, f) for f in os.listdir(args.folder1) if f.endswith('.png')]
    img_files2 = [os.path.join(args.folder2, f) for f in os.listdir(args.folder2) if f.endswith('.png')]

    # Initialize a list to store the PSNR values
    psnr_values = []

    # Generate all pairs of images
    for img_file1, img_file2 in zip(img_files1, img_files2):
        # Load the images
        img1 = cv2.imread(img_file1)
        img2 = cv2.imread(img_file2)

        # Compute the PSNR and add it to the list
        psnr = compute_psnr(img1, img2)
        psnr_values.append(psnr)

    # Compute the average PSNR
    avg_psnr = np.mean(psnr_values)

    print(f'Average PSNR: {avg_psnr}')
import json
import numpy as np
import argparse


# Code adapted from https://github.com/NVlabs/instant-ngp/issues/658
def nerfstudio_to_ngp(current_transform):
    # Assuming that I am in OpenGL with NerfStudio

    # Transformation matrix for coordinate system conversion to ngp
    transformation_matrix = np.array([[0, 1,  0, 0],
                                      [1, 0,  0, 0],
                                      [0, 0, -1, 0],
                                      [0, 0,  0, 1]])
    new_pose_matrix = transformation_matrix @ current_transform
    return new_pose_matrix


def nerf_to_ngp(xf):
    mat = np.copy(xf)
    mat[:, 1] *= -1 #flip axis
    mat[:, 2] *= -1
    mat = mat[[1, 2, 0, 3], :]
    return mat


def ngp_to_nerf(xf):
    mat = np.copy(xf)
    # mat[:,3] -= 0.025
    mat = mat[[2, 0, 1], :]  # swap axis
    mat[:, 1] *= -1  # flip axis
    mat[:, 2] *= -1

    mat[:, 3] -= [0.5, 0.5, 0.5]  # translation and re-scale
    mat[:, 3] /= 0.33

    return mat


def closest_point_2_lines(oa, da, ob, db): # returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
	da=da/np.linalg.norm(da)
	db=db/np.linalg.norm(db)
	c=np.cross(da,db)
	denom=(np.linalg.norm(c)**2)
	t=ob-oa
	ta=np.linalg.det([t,db,c])/(denom+1e-10)
	tb=np.linalg.det([t,da,c])/(denom+1e-10)
	if ta<0:
		ta=0
	if tb<0:
		tb=0
	return (oa+ta*da+ob+tb*db)*0.5,denom


def scale_for_ngp(frames):
    centroid = np.zeros(3)

    new_frames = np.copy(frames)

    for i, frame in enumerate(frames):
        current_transform = np.array(frame['transform_matrix'])
        print(f"shape of current transform is {current_transform.shape}")
        centroid += current_transform[:3, 3]

    nframes = len(frames)
    centroid *= 1 / nframes
    print(f"centroid is {centroid}")

    avglen = 0.
    for i, frame in enumerate(frames):
        current_transform = np.array(frame['transform_matrix'])

        # Subtract centroid
        current_transform[:3, 3] -= centroid

        # Add norm to average length
        avglen += np.linalg.norm(current_transform[:3, 3])

        # Add to new frames
        new_frames[i]["transform_matrix"] = current_transform.tolist()

    avglen /= nframes
    print("avg camera distance from origin ", avglen)

    for i in range(nframes):
        np_new_frame = np.array(new_frames[i]["transform_matrix"])
        np_new_frame[:3,3] *= 3. / avglen # scale to "nerf sized"
        # Sneaking in flipping of axes to go from nerfstudio to ngp coord system
        new_frames[i]["transform_matrix"] = nerfstudio_to_ngp(np_new_frame).tolist()

    # find a central point they are all looking at
    print("computing center of attention...")
    totw = 0
    totp = [0, 0, 0]
    for i in range(nframes):
        mf = np.array(new_frames[i]["transform_matrix"])[0:3, :]
        for j in range(nframes):
            mg = np.array(new_frames[j]["transform_matrix"])[0:3, :]
            p, w = closest_point_2_lines(mf[:, 3], mf[:, 2], mg[:, 3], mg[:, 2])
            if w > 0.01:
                totp += p * w
                totw += w
    totp /= totw
    print(totp)  # the cameras are looking at totp
    for i in range(nframes):
        current_transform = np.array(new_frames[i]["transform_matrix"])
        current_transform[0:3, 3] -= totp
        new_frames[i]["transform_matrix"] = current_transform.tolist()

    return new_frames.tolist()


def convert_transform(input_filename):
    # Load the original transformation matrix from the JSON file
    with open(input_filename, 'r') as file:
        data = json.load(file)

    frames = data['frames']

    transformed_frames = scale_for_ngp(frames)

    data['frames'] = transformed_frames

    return data


def get_argparser():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--file', type=str, help='Path to the input JSON file')
    return argparser


if __name__ == '__main__':
    # Get args
    args = get_argparser().parse_args()

    # Example usage
    transformed_data = convert_transform(args.file)

    # Write the updated transformation matrix to a new JSON file
    output_filename = args.file.replace('.json', '_ngp.json')
    with open(output_filename, 'w') as file:
        json.dump(transformed_data, file, indent=4)
    print(f"Converted transformation matrix written to {output_filename}")


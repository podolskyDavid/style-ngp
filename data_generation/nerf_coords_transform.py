import json
import numpy as np
import argparse
from util import switch_coord_system


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
        new_frames[i]["transform_matrix"] = switch_coord_system(np_new_frame).tolist()

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

    return new_frames.tolist(), centroid, avglen, totp


def convert_transform(input_filename, gt_filename, targets_filenames):
    # Load the original transformation matrix from the JSON file
    with open(input_filename, 'r') as file:
        data = json.load(file)

    frames = data['frames']

    transformed_frames, centroid, avglen, totp = scale_for_ngp(frames)

    data['frames'] = transformed_frames

    # If a ground truth file is provided, load and transform it
    if gt_filename is not None:
        with open(gt_filename, 'r') as file:
            gt_data = json.load(file)

        gt_transform = np.array(gt_data['transform_matrix'])

        # Subtract the centroid, flip axes, and scale
        gt_transform[:3, 3] -= centroid
        gt_transform[:3, 3] *= 3. / avglen
        gt_transform = switch_coord_system(gt_transform)
        gt_transform[:3, 3] -= totp

        # Write the transformed ground truth pose to a new JSON file
        gt_output_filename = gt_filename.replace('.json', '_ngp.json')
        with open(gt_output_filename, 'w') as file:
            json.dump(
                {'transform_matrix': gt_transform.tolist(),
                 'centroid': centroid.tolist(),
                 'avglen': avglen,
                 'totp': totp.tolist()},
                file,
                indent=4
            )
        print(f"Converted ground truth transformation matrix written to {gt_output_filename}")

    # If a targets file is provided, load and transform it
    if targets_filenames is not None:
        for targets_filename in targets_filenames:
            with open(targets_filename, 'r') as file:
                targets_data = json.load(file)

            # Iterate through the targets
            for target in targets_data['targets']:
                # Extract the init_pose and target_pose
                init_pose = np.array(target['init_pose'])
                target_pose = np.array(target['target_pose'])

                # Subtract the centroid, flip axes, and scale
                init_pose[:3, 3] -= centroid
                init_pose[:3, 3] *= 3. / avglen
                init_pose = switch_coord_system(init_pose)
                init_pose[:3, 3] -= totp

                target_pose[:3, 3] -= centroid
                target_pose[:3, 3] *= 3. / avglen
                target_pose = switch_coord_system(target_pose)
                target_pose[:3, 3] -= totp

                # Add the transformed poses back to the target dictionary
                target['init_pose_ngp'] = init_pose.tolist()
                target['target_pose_ngp'] = target_pose.tolist()

            # Also add the transformation parameters
            targets_data['centroid'] = centroid.tolist()
            targets_data['avglen'] = avglen
            targets_data['totp'] = totp.tolist()

            # Write the transformed targets to a new JSON file
            targets_output_filename = targets_filename.replace('.json', '_ngp.json')
            with open(targets_output_filename, 'w') as file:
                json.dump(targets_data, file, indent=4)
            print(f"Converted targets written to {targets_output_filename}")

    return data

def get_argparser():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--transforms_file', type=str, help='Path to the input JSON file')
    argparser.add_argument('--gt_file', type=str, help='Path to the ground truth JSON file that can also be transformed', default=None)
    argparser.add_argument('--targets_files', type=str, nargs='+', help='Paths to the targets JSON files that can be transformed', default=None)
    return argparser

if __name__ == '__main__':
    # Get args
    args = get_argparser().parse_args()

    # Example usage
    transformed_data = convert_transform(args.transforms_file, args.gt_file, args.targets_files)

    # Write the updated transformation matrix to a new JSON file
    output_filename = args.transforms_file.replace('.json', '_ngp.json')
    with open(output_filename, 'w') as file:
        json.dump(transformed_data, file, indent=4)
    print(f"Converted transformation matrix written to {output_filename}")


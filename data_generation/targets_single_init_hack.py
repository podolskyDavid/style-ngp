import json
import os
import glob


def replace_init_pose(target_file, new_init_pose):
    # Load the target JSON file
    with open(target_file, 'r') as f:
        data = json.load(f)

    # Replace the init_pose with the new_init_pose
    for target in data['targets']:
        target['init_pose_ngp'] = new_init_pose

    # Save the modified data to a new JSON file
    output_file = target_file.replace('.json', '_single_init.json')
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)


# Use this script to pick a single reasonable pose (e.g., central shot of the mesh) for all pose estimation runs on
#  all targets
#  This is akin to real surgery where preop surgical planning would yield this first estimate
if __name__ == "__main__":
    folder = '/home/maximilian_fehrentz/Documents/MICCAI/000/data/synth_targets'
    new_init_pose = [
        [
            -0.9934905400834416,
            -0.06321570912610927,
            0.09476455500024572,
            0.7580934290044564
        ],
        [
            -0.10972578928313549,
            0.30753697895796056,
            -0.9451884773629035,
            -1.0542691910300133
        ],
        [
            0.030607154897252353,
            -0.9494339264493838,
            -0.31247147289952976,
            0.08015873239998061
        ],
        [
            0.0,
            0.0,
            0.0,
            1.0
        ]
    ]

    # Get a list of all files in the folder that end with targets_ngp
    target_files = glob.glob(os.path.join(folder, '*targets_ngp.json'))

    # Iterate over the target files and replace the init pose
    for target_file in target_files:
        replace_init_pose(target_file, new_init_pose)
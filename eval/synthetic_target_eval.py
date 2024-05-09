import json
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.table as tbl
import os
from utils import reverse_transform


def get_argparser():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--record_files', type=str, nargs='+', required=True, help='Paths to the record JSON files')
    return argparser


def load_scale_and_centroid(targets_file):
    with open(targets_file, 'r') as f:
        targets_data = json.load(f)
    avglen = targets_data['avglen']
    centroid = np.array(targets_data['centroid'])
    totp = np.array(targets_data['totp'])
    return avglen, centroid, totp


def load_record(record_file, avglen, centroid, totp):
    with open(record_file, 'r') as f:
        record_data = json.load(f)
    for result in record_data['results']:
        result['pred_pose'] = np.array(result['pred_pose'])
        result['gt_pose'] = np.array(result['gt_pose'])
        # Rotations (and their errors computed by Parallel Inversion) are ok but translations from NGP need to be
        # converted back to VTK where they have a physical meaning (in mm)
        result['pred_pose'][:3, 3] = np.array(reverse_transform(result['pred_pose'], centroid, avglen, totp))[:3, 3]
        result['gt_pose'][:3, 3] = np.array(reverse_transform(result['gt_pose'], centroid, avglen, totp))[:3, 3]
        result['error_translation'] = np.linalg.norm(result['pred_pose'][:3, 3] - result['gt_pose'][:3, 3])
    return record_data


def compute_accuracy_threshold_curve(record_data, threshold_type):
    errors = [result['error_' + threshold_type] for result in record_data['results']]
    if threshold_type == "rotation":
        cutoff_degree = 5
        thresholds = np.linspace(0, cutoff_degree, 100)
    if threshold_type == "translation":
        cutoff_trans = 5
        thresholds = np.linspace(0, cutoff_trans, 100)
    accuracies = [np.mean(np.array(errors) <= threshold) for threshold in thresholds]
    return [round(t, 2) for t in thresholds], [round(a, 2) for a in accuracies]


def compute_avg_error(record_data, rot_threshold, trans_threshold):
    rot_errors = [result['error_rotation'] for result in record_data['results']]
    trans_errors = [result['error_translation'] for result in record_data['results']]

    # Filter out the outliers based on both rotation and translation thresholds
    rot_errors_corrected = [error for i, error in enumerate(rot_errors) if error <= rot_threshold and (trans_threshold is None or trans_errors[i] <= trans_threshold)]
    trans_errors_corrected = [error for i, error in enumerate(trans_errors) if (trans_threshold is None or error <= trans_threshold) and rot_errors[i] <= rot_threshold]

    avg_trans_error = np.mean(trans_errors_corrected)
    avg_rot_error = np.mean(rot_errors_corrected)
    max_trans_error = np.max(trans_errors)
    max_rot_error = np.max(rot_errors)
    num_outliers = len(rot_errors) - len(rot_errors_corrected)

    num_outliers_sanity = len(trans_errors) - len(trans_errors_corrected)
    assert num_outliers == num_outliers_sanity

    return avg_trans_error, avg_rot_error, max_trans_error, max_rot_error, num_outliers


if __name__ == "__main__":
    # Get args
    args = get_argparser().parse_args()

    # TODO: hardcoded, change later
    scene_names = ["Style1", "Style2", "Style3", "Style4"]

    # Get the directory of target_files
    config_files = [os.path.join(os.path.dirname(record_file), 'config.json') for record_file in args.record_files]

    # Initialize lists to store the average and maximum errors for all scenes
    avg_trans_errors = []
    avg_rot_errors = []
    max_trans_errors = []
    max_rot_errors = []
    num_outliers_list = []
    scenes = []

    # Initialize figures for the plots
    fig_trans, ax_trans = plt.subplots()
    # 1 to 5
    plt.xticks(np.arange(0, 6, step=1))
    fig_rot, ax_rot = plt.subplots()
    # 1 to 5
    plt.xticks(np.arange(0, 6, step=1))

    # Iterate over the pairs of record and config files
    for i, (record_file, config_file) in enumerate(zip(args.record_files, config_files)):
        # Load the config file
        with open(config_file, 'r') as f:
            config_data = json.load(f)

        # # Extract the scene and target file from the config data
        # scene = config_data['scene']

        scene = scene_names[i]
        scenes.append(scene)
        target_file = config_data['targets_json_path']

        # Load the scale, centroid, and totp from the target file
        avglen, centroid, totp = load_scale_and_centroid(target_file)

        # Load the record data
        record_data = load_record(record_file, avglen, centroid, totp)
        trans_thresholds, trans_accuracies = compute_accuracy_threshold_curve(record_data, 'translation')
        rot_thresholds, rot_accuracies = compute_accuracy_threshold_curve(record_data, 'rotation')

        # Add the accuracy threshold curves to the plots
        ax_trans.plot(trans_thresholds, trans_accuracies, label=scene)
        ax_rot.plot(rot_thresholds, rot_accuracies, label=scene)

        avg_trans_error, avg_rot_error, max_trans_error, max_rot_error, num_outliers = compute_avg_error(record_data, rot_threshold=20, trans_threshold=None)
        avg_trans_errors.append(avg_trans_error)
        avg_rot_errors.append(avg_rot_error)
        max_trans_errors.append(max_trans_error)
        max_rot_errors.append(max_rot_error)
        num_outliers_list.append(num_outliers)

    # Add legends to the plots
    ax_trans.legend()
    ax_rot.legend()

    # Set the font size and weight for all text elements in the plot
    plt.rcParams.update({'font.size': 30, 'font.weight': 'bold'})

    # Add grid to the plots
    ax_trans.grid(True)
    ax_rot.grid(True)

    # Add labels to the axes
    ax_trans.set_xlabel('Translation Threshold (in mm)')
    ax_trans.set_ylabel('Percentage of Predictions < Threshold')
    ax_rot.set_xlabel('Rotation Threshold (in deg)')
    ax_rot.set_ylabel('Percentage of Predictions < Threshold')

    # Save the plots
    fig_trans.savefig('translation.svg')
    fig_rot.savefig('rotation.svg')

    # Create a table with the average and maximum errors for all scenes
    all_cases_table = {
        'Scene': scenes,
        'Avg Trans Error (mm)': [round(e, 2) for e in avg_trans_errors],
        'Avg Rot Error (deg)': [round(e, 2) for e in avg_rot_errors],
        # 'Max Trans Error (mm)': [round(e, 2) for e in max_trans_errors],
        # 'Max Rot Error (deg)': [round(e, 2) for e in max_rot_errors],
        'Num Outliers': num_outliers_list
    }

    # Compute the average and maximum errors averaged over all scenes
    avg_all_scenes_table = {
        'Avg Trans Error (mm)': [round(np.mean(avg_trans_errors), 2)],
        'Avg Rot Error (deg)': [round(np.mean(avg_rot_errors), 2)],
        # 'Max Trans Error (mm)': round(np.mean(max_trans_errors), 2),
        # 'Max Rot Error (deg)': round(np.mean(max_rot_errors), 2),
        'Num Outliers': [np.sum(num_outliers_list)]
    }

    # Create a larger figure for the table
    fig_table, ax_table = plt.subplots(figsize=(16, 12))

    ax_table.axis('tight')
    ax_table.axis('off')
    table = tbl.table(ax_table, cellText=list(all_cases_table.values()), rowLabels=list(all_cases_table.keys()), loc='center')

    # Save the table
    fig_table.savefig('table_all_cases.png')

    # Create a table averaged over all scenes
    fig_table_avg, ax_table_avg = plt.subplots(figsize=(16, 12))

    ax_table_avg.axis('tight')
    ax_table_avg.axis('off')
    table_avg = tbl.table(ax_table_avg, cellText=list(avg_all_scenes_table.values()), rowLabels=list(avg_all_scenes_table.keys()), loc='center')

    # Save the table
    fig_table_avg.savefig('table_avg_all_scenes.png')
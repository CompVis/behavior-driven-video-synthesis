#!/usr/bin/env python3
# based on the file 'process_all.py' from https://github.com/anibali/h36m-fetch

import argparse
from os import path, makedirs, listdir
from shutil import move
import traceback
import cdflib
import numpy as np
from subprocess import call
from tempfile import TemporaryDirectory
from tqdm import tqdm

from data.metadata import load_h36m_metadata


metadata = load_h36m_metadata()

# Subjects to include when preprocessing
included_subjects = {
    "S1": 1,
    "S5": 5,
    "S6": 6,
    "S7": 7,
    "S8": 8,
    "S9": 9,
    "S11": 11,
}


# Rather than include every frame from every video, we can instead wait for the pose to change
# significantly before storing a new example.
def select_frame_indices_to_include(subject, poses_3d_univ):
    # To process every single frame, uncomment the following line:
    return np.arange(0, len(poses_3d_univ))


def infer_camera_intrinsics(points2d, points3d):
    """Infer camera instrinsics from 2D<->3D point correspondences."""
    pose2d = points2d.reshape(-1, 2)
    pose3d = points3d.reshape(-1, 3)
    x3d = np.stack([pose3d[:, 0], pose3d[:, 2]], axis=-1)
    x2d = (pose2d[:, 0] * pose3d[:, 2])
    alpha_x, x_0 = list(np.linalg.lstsq(x3d, x2d, rcond=-1)[0].flatten())
    y3d = np.stack([pose3d[:, 1], pose3d[:, 2]], axis=-1)
    y2d = (pose2d[:, 1] * pose3d[:, 2])
    alpha_y, y_0 = list(np.linalg.lstsq(y3d, y2d, rcond=-1)[0].flatten())
    return np.array([alpha_x, x_0, alpha_y, y_0])


def process_view(ddir, out_dir, subject, action, subaction, camera):
    subj_dir = path.join(ddir,'extracted', subject)

    base_filename = metadata.get_base_filename(subject, action, subaction, camera)
    poses_3d_univ = cdflib.CDF(path.join(subj_dir, 'Poses_D3_Positions_mono_universal', base_filename + '.cdf'))
    poses_3d_univ = np.array(poses_3d_univ['Pose'])
    poses_3d_univ = poses_3d_univ.reshape(poses_3d_univ.shape[1], 32, 3)

    frame_indices = select_frame_indices_to_include(subject, poses_3d_univ)
    frames = frame_indices + 1
    video_file = path.join(subj_dir, 'Videos', base_filename + '.mp4')
    frames_dir = path.join(out_dir, 'imageSequence', camera)
    makedirs(frames_dir, exist_ok=True)

    # Check to see whether the frame images have already been extracted previously
    existing_files = {f for f in listdir(frames_dir)}
    frames_are_extracted = True
    for i in frames:
        filename = 'img_%06d.jpg' % i
        if filename not in existing_files:
            frames_are_extracted = False
            break

    if not frames_are_extracted:
        with TemporaryDirectory() as tmp_dir:
            # Use ffmpeg to extract frames into a temporary directory
            call([
                'ffmpeg',
                '-nostats', '-loglevel', 'error',
                '-i', video_file,
                '-qscale:v', '3',
                path.join(tmp_dir, 'img_%06d.jpg')
            ])

            # Move included frame images into the output directory
            for i in frames:
                filename = 'img_%06d.jpg' % i
                move(
                    path.join(tmp_dir, filename),
                    path.join(frames_dir, filename)
                )




def process_subaction(ddir,subject, action, subaction):
    datasets = {}

    out_dir = path.join(ddir,'processed','all', subject, metadata.action_names[action] + '-' + subaction)
    makedirs(out_dir, exist_ok=True)

    for camera in tqdm(metadata.camera_ids, ascii=True, leave=False):
        try:
            process_view(ddir, out_dir, subject, action, subaction, camera)
        except:
            tqdm.write('!!! Error processing sequence, skipping: ' + \
                       repr((subject, action, subaction, camera)))
            tqdm.write(traceback.format_exc())
            continue


def process_all(ddir):
    sequence_mappings = metadata.sequence_mappings

    subactions = []

    for subject in included_subjects.keys():
        subactions += [
            (subject, action, subaction)
            for action, subaction in sequence_mappings[subject].keys()
            if int(action) > 1  # Exclude '_ALL'
        ]

    for subject, action, subaction in tqdm(subactions, ascii=True, leave=False):
        process_subaction(ddir,subject, action, subaction)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--datadir", type=str,
                        help="path to the data",required=True)
    args = parser.parse_args()
    ddir = args.datadir
    process_all(ddir)


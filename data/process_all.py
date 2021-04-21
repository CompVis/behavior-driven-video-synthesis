#!/usr/bin/env python3

from os import path, makedirs, listdir
from shutil import move
from spacepy import pycdf
import numpy as np
import h5py
from subprocess import call
from tempfile import TemporaryDirectory
from tqdm import tqdm
import imagesize
from glob import glob

from metadata import load_h36m_metadata

"action_name",
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

# Sequences with known issues
blacklist = {("S11", "2", "2", "54138969")}  # Video file is corrupted


# Rather than include every frame from every video, we can instead wait for the pose to change
# significantly before storing a new example.
def select_frame_indices_to_include(subject, poses_3d_univ):
    # To process every single frame, uncomment the following line:
    # return np.arange(0, len(poses_3d_univ))

    # Take every 64th frame for the protocol #2 test subjects
    # (see the "Compositional Human Pose Regression" paper)
    # if subject == "S9" or subject == "S11":
    #     return np.arange(0, len(poses_3d_univ), 64)

    # Take only frames where movement has occurred for the protocol #2 train subjects
    frame_indices = []
    prev_joints3d = None
    # threshold = (
    #     40 ** 2
    # )  # Skip frames until at least one joint has moved by 40mm
    for i, joints3d in enumerate(poses_3d_univ):
        # if prev_joints3d is not None:
        #     max_move = ((joints3d - prev_joints3d) ** 2).sum(axis=-1).max()
        #     # if max_move < threshold:
        #     #     continue
        # prev_joints3d = joints3d
        frame_indices.append(i)
    return np.array(frame_indices)


def infer_camera_intrinsics(points2d, points3d):
    """Infer camera instrinsics from 2D<->3D point correspondences."""
    pose2d = points2d.reshape(-1, 2)
    pose3d = points3d.reshape(-1, 3)
    x3d = np.stack([pose3d[:, 0], pose3d[:, 2]], axis=-1)
    x2d = pose2d[:, 0] * pose3d[:, 2]
    alpha_x, x_0 = list(np.linalg.lstsq(x3d, x2d, rcond=-1)[0].flatten())
    y3d = np.stack([pose3d[:, 1], pose3d[:, 2]], axis=-1)
    y2d = pose2d[:, 1] * pose3d[:, 2]
    alpha_y, y_0 = list(np.linalg.lstsq(y3d, y2d, rcond=-1)[0].flatten())
    return np.array([alpha_x, x_0, alpha_y, y_0])


def process_view(out_dir, subject, action, subaction, camera):
    subj_dir = path.join(
        "/export/data/ablattma/Datasets/human3.6M/Poses3D/extracted", subject
    )

    base_filename = metadata.get_base_filename(
        subject, action, subaction, camera
    )
    # world_filnames_pose = glob(
    #     path.join(
    #         subj_dir,
    #         "Poses_D3_Positions",
    #         out_dir.split("/")[-1][:-2] + "*.cdf",
    #     )
    # )
    # world_filenames_angles = glob(
    #     path.join(
    #         subj_dir, "Poses_D3_Angles", out_dir.split("/")[-1][:-2] + "*.cdf"
    #     )
    # )
    world_name = base_filename.split(".")[0]
    # Load joint position annotations
    # with pycdf.CDF(
    #     path.join(subj_dir, "Poses_D2_Positions", base_filename + ".cdf")
    # ) as cdf:
    #     poses_2d = np.array(cdf["Pose"])
    #     poses_2d = poses_2d.reshape((poses_2d.shape[1], 32, 2))
    with pycdf.CDF(
        path.join(
            subj_dir,
            "Poses_D3_Positions_mono_universal",
            base_filename + ".cdf",
        )
    ) as cdf:
        poses_3d_univ = np.array(cdf["Pose"])
        poses_3d_univ = poses_3d_univ.reshape((poses_3d_univ.shape[1], 32, 3))
    # with pycdf.CDF(
    #     path.join(subj_dir, "Poses_D3_Positions_mono", base_filename + ".cdf")
    # ) as cdf:
    #     poses_3d = np.array(cdf["Pose"])
    #     poses_3d = poses_3d.reshape((poses_3d.shape[1], 32, 3))
    # with pycdf.CDF(
    #     path.join(subj_dir, "Poses_D3_Angles_mono", base_filename + ".cdf")
    # ) as cdf:
    #     angles_3d = np.array(cdf["Pose"])
    #     angles_3d = angles_3d.reshape((angles_3d.shape[1], 78))
    # poses_3d_world = None
    # for f in world_filnames_pose:
    with pycdf.CDF(path.join(subj_dir,"Poses_D3_Positions",world_name + ".cdf")) as cdf:
        poses_3d_world = np.array(cdf["Pose"])
        poses_3d_world = poses_3d_world.reshape(
            (poses_3d_world.shape[1], 32, 3)
        )
    assert poses_3d_world.shape[0] == poses_3d_univ.shape[0]
    angles_3d_world = []
    try:
        with pycdf.CDF(path.join(subj_dir,"Poses_D3_Angles",world_name + ".cdf")) as cdf:
            angles_3d_world = np.array(cdf["Pose"])
            angles_3d_world = angles_3d_world.reshape(
                (angles_3d_world.shape[1], 78)
            )
        if not angles_3d_world.shape[0] == poses_3d_univ.shape[0]:
            print("shape of poses univ: ",poses_3d_univ.shape[0])
            print("shape of poses: ",poses_3d_world.shape[0])
            print("shape of angles: ",angles_3d_world.shape[0])
            exit(-1)

    except pycdf.CDFError as e:
        print(e)
        print("name: " + path.join(subj_dir,"Poses_D3_Angles",world_name + ".cdf"))
        exit(-1)
    # with pycdf.CDF(path.join(subj_dir, 'Poses_D3_Angles', base_filename + '.cdf')) as cdf:
    #     angles_3d_world = np.array(cdf['Pose'])
    #     angles_3d_world = poses_3d.reshape((angles_3d_world.shape[1], 32, 3))

    # Infer camera intrinsics
    # camera_int = infer_camera_intrinsics(poses_2d, poses_3d)
    # camera_int_univ = infer_camera_intrinsics(poses_2d, poses_3d_univ)

    frame_indices = select_frame_indices_to_include(subject, poses_3d_univ)
    frames = frame_indices + 1
    video_file = path.join(subj_dir, "Videos", base_filename + ".mp4")
    frames_dir = path.join(out_dir, "imageSequence", camera)
    if not path.isdir(frames_dir):
        raise NotADirectoryError(
            f'Desired frames dir "{frames_dir}" not found. Frames do not seem to be entirely extracted.'
        )
    # makedirs(frames_dir, exist_ok=True)

    # Check to see whether the frame images have already been extracted previously
    existing_files = {f for f in listdir(frames_dir)}
    frames_are_extracted = True
    # frame_paths = []
    # imsizes = []
    # intrinsics = []
    # intrinsics_univ = []
    for i in tqdm(frames, ascii=True, leave=False):
        filename = "img_%06d.jpg" % i
        if filename not in existing_files:
            raise Exception("Frames not entirely extracted.")
        # fp = path.join(frames_dir, filename)
        # frame_paths.append(fp)
        # imsize = imagesize.get(fp)
        # imsizes.append(imsize)
        # intrinsics.append(camera_int)
        # intrinsics_univ.append(camera_int_univ)

    # if not frames_are_extracted:
    #     with TemporaryDirectory() as tmp_dir:
    #         # Use ffmpeg to extract frames into a temporary directory
    #         call([
    #             'ffmpeg',
    #             '-nostats', '-loglevel', '0',
    #             '-i', video_file,
    #             '-qscale:v', '3',
    #             path.join(tmp_dir, 'img_%06d.jpg')
    #         ])
    #
    #         # Move included frame images into the output directory
    #         for i in frames:
    #             filename = 'img_%06d.jpg' % i
    #             move(
    #                 path.join(tmp_dir, filename),
    #                 path.join(frames_dir, filename)
    #             )
    # imsizes = np.asarray(imsizes, dtype=np.int)
    # frame_paths = np.asarray(frame_paths, dtype=np.dtype("S256"))
    # poses_2d_norm = np.divide(poses_2d, np.expand_dims(imsizes, axis=1))
    # intrinsics = np.stack(intrinsics, axis=0)
    # intrinsics_univ = np.stack(intrinsics_univ, axis=0)
    return {
        # "pose_2d": poses_2d[frame_indices],
        # "pose_normalized_2d": poses_2d_norm[frame_indices],
        # "pose_3d_univ": poses_3d_univ[frame_indices],
        # "pose_3d": poses_3d[frame_indices],
        # "angle_3d": angles_3d[frame_indices],
        # "frame_path": frame_paths,
        # "image_size": imsizes,
        # "intrinsics": intrinsics,
        # "intrinsics_univ": intrinsics_univ,
        # "frame": frames,
        # "camera": np.full(frames.shape, int(camera)),
        # "subject": np.full(frames.shape, int(included_subjects[subject])),
        # "action": np.full(frames.shape, int(action)),
        # "subaction": np.full(frames.shape, int(subaction)),
        "pose_3d_world": poses_3d_world[frame_indices],
        "angle_3d_world": angles_3d_world[frame_indices],
    }


def process_subaction(subject, action, subaction):
    datasets = {}

    out_dir = path.join(
        "processed",
        "all",
        subject,
        metadata.action_names[action] + "-" + subaction,
    )
    if not path.isdir(out_dir):
        raise NotADirectoryError(f"Desired out_dir {out_dir} not found.")
    # makedirs(out_dir, exist_ok=True)

    for nr, camera in enumerate(
        tqdm(metadata.camera_ids, ascii=True, leave=False)
    ):
        if (subject, action, subaction, camera) in blacklist:
            continue

        # try:
        annots = process_view(out_dir, subject, action, subaction, camera)
        # except Exception as e:
        #     print('Error processing sequence, skipping: ', repr((subject, action, subaction, camera)))
        #     print("Exception: ",e)
        #     continue

        for k, v in annots.items():
            if k in datasets:
                datasets[k].append(v)
            else:
                datasets[k] = [v]

    if len(datasets) == 0:
        return

    datasets = {k: np.concatenate(v) for k, v in datasets.items()}

    return datasets

    # with h5py.File(path.join(out_dir, 'annot.h5'), 'w') as f:
    #     for name, data in datasets.items():
    #         f.create_dataset(name, data=data)


def process_all():
    sequence_mappings = metadata.sequence_mappings
    save_dir = "/export/data/ablattma/Datasets/human3.6M/processed/all/"
    subactions = []

    for subject in included_subjects.keys():
        subactions += [
            (subject, action, subaction)
            for action, subaction in sequence_mappings[subject].keys()
            if int(action) > 1  # Exclude '_ALL'
        ]

    datasets = {}
    for subject, action, subaction in tqdm(subactions, ascii=True, leave=False):
        current_data = process_subaction(subject, action, subaction)

        for k in current_data:
            if k in datasets:
                datasets[k].append(current_data[k])
            else:
                datasets[k] = [current_data[k]]

    for k in datasets:
        # if "intrinsics" in k:
        #     datasets[k] = np.stack(datasets[k], axis=0)
        # else:
        datasets[k] = np.concatenate(datasets[k], axis=0)

        # if k == "frame_path":
        #     datasets[k] = datasets.astype(np.dtype("S256"))

    with h5py.File(
        path.join(save_dir, "annot_complete.h5"), "r+"
    ) as f:
        for name, data in datasets.items():
            f.create_dataset(name, data=data)


if __name__ == "__main__":
    process_all()
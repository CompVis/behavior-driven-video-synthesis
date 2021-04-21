"""Functions that help with data processing for human3.6m"""
import numpy as np
import copy
from tqdm.autonotebook import tqdm
import matplotlib as mpl
import xml.etree.ElementTree as ET
from os import path
from lib.utils import add_joints_to_img
from data.base_dataset import BaseDataset
import cv2
from lib.utils import make_joint_img, scale_img
from models.vunets import VunetOrg
import torch
# required for 3d box plot of matplotlib, though not used
# package seems not to be exported
from mpl_toolkits.mplot3d import axes3d, Axes3D


def euler_to_rotation_matrix(angles, deg=True, format="zxy"):
    """

    :param angles: angles around axes (zxy) or xyz
    :return:
    """
    if deg:
        angles = np.radians(angles)

    cx = np.cos(angles[0])
    cy = np.cos(angles[1])
    cz = np.cos(angles[2])
    sx = np.sin(angles[0])
    sy = np.sin(angles[1])
    sz = np.sin(angles[2])

    # init = np.eye(3)
    if format == "zxy":
        R = np.asarray(
            [
                [cy * cz - sx * sy * sz, cy * sz + sx * sy * cz, -sy * cx],
                [-cx * sz, cx * cz, sx],
                [sy * cz + cy * sx * sz, sy * sz - cy * sx * cz, cy * cx],
            ]
        )
    elif format == "xyz":
        R = (
            np.asarray([[cz, sz, 0.0], [-sz, cz, 0.0], [0.0, 0.0, 1.0]])
            @ np.asarray([[cy, 0, -sy], [0.0, 1.0, 0.0], [sy, 0.0, cy]])
            @ np.asarray([[1.0, 0, 0.0], [0.0, cx, sx], [0.0, -sx, cx]])
        )
    else:
        raise NotImplementedError()

    return R


def rotmat2euler(R):
    """
Converts a rotation matrix to Euler angles
Matlab port to python for evaluation purposes
https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/RotMat2Euler.m#L1

Args
  R: a 3x3 rotation matrix
Returns
  eul: a 3x1 Euler angle representation of R
"""
    if R[0, 2] == 1 or R[0, 2] == -1:
        # special case
        E3 = 0  # set arbitrarily
        dlta = np.arctan2(R[0, 1], R[0, 2])

        if R[0, 2] == -1:
            E2 = np.pi / 2
            E1 = E3 + dlta
        else:
            E2 = -np.pi / 2
            E1 = -E3 + dlta

    else:
        E2 = -np.arcsin(R[0, 2])
        E1 = np.arctan2(R[1, 2] / np.cos(E2), R[2, 2] / np.cos(E2))
        E3 = np.arctan2(R[0, 1] / np.cos(E2), R[0, 0] / np.cos(E2))

    eul = np.array([E1, E2, E3])
    return eul


def quat2expmap(q):
    """
Converts a quaternion to an exponential map
Matlab port to python for evaluation purposes
https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/quat2expmap.m#L1

Args
  q: 1x4 quaternion
Returns
  r: 1x3 exponential map
Raises
  ValueError if the l2 norm of the quaternion is not close to 1
"""
    if np.abs(np.linalg.norm(q) - 1) > 1e-3:
        # raise ValueError
        print("corrupteeeeed....")

    sinhalftheta = np.linalg.norm(q[1:])
    coshalftheta = q[0]

    r0 = np.divide(q[1:], (np.linalg.norm(q[1:]) + np.finfo(np.float32).eps))
    theta = 2 * np.arctan2(sinhalftheta, coshalftheta)
    theta = np.mod(theta + 2 * np.pi, 2 * np.pi)

    if theta > np.pi:
        theta = 2 * np.pi - theta
        r0 = -r0

    r = r0 * theta
    return r


def rotmat2quat(R):
    """
Converts a rotation matrix to a quaternion
Matlab port to python for evaluation purposes
https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/rotmat2quat.m#L4

Args
  R: 3x3 rotation matrix
Returns
  q: 1x4 quaternion
"""
    rotdiff = R - R.T

    r = np.zeros(3)
    r[0] = -rotdiff[1, 2]
    r[1] = rotdiff[0, 2]
    r[2] = -rotdiff[0, 1]
    sintheta = np.linalg.norm(r) / 2
    r0 = np.divide(r, np.linalg.norm(r) + np.finfo(np.float32).eps)

    costheta = (np.trace(R) - 1) / 2

    theta = np.arctan2(sintheta, costheta)

    q = np.zeros(4)
    q[0] = np.cos(theta / 2)
    q[1:] = r0 * np.sin(theta / 2)
    return q


def rotmat2expmap(R):
    return quat2expmap(rotmat2quat(R))


def expmap2rotmat(r):
    """
Converts an exponential map angle to a rotation matrix
Matlab port to python for evaluation purposes
I believe this is also called Rodrigues' formula
https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/expmap2rotmat.m

Args
  r: 1x3 exponential map
Returns
  R: 3x3 rotation matrix
"""
    theta = np.linalg.norm(r)
    r0 = np.divide(r, theta + np.finfo(np.float32).eps)
    r0x = np.array([0, -r0[2], r0[1], 0, 0, -r0[0], 0, 0, 0]).reshape(3, 3)
    r0x = r0x - r0x.T
    R = (
        np.eye(3, 3)
        + np.sin(theta) * r0x
        + (1 - np.cos(theta)) * (r0x).dot(r0x)
    )
    return R


def unNormalizeData(normalizedData, data_mean, data_std, dimensions_to_ignore):
    """Borrowed from SRNN code. Reads a csv file and returns a float32 matrix.
https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/generateMotionData.py#L12

Args
  normalizedData: nxd matrix with normalized data
  data_mean: vector of mean used to normalize the data
  data_std: vector of standard deviation used to normalize the data
  dimensions_to_ignore: vector with dimensions not used by the model
  actions: list of strings with the encoded actions
  one_hot: whether the data comes with one-hot encoding
Returns
  origData: data originally used to
"""
    T = normalizedData.shape[0]
    D = data_mean.shape[0]

    origData = np.zeros((T, D), dtype=np.float32)
    dimensions_to_use = []
    for i in range(D):
        if i in dimensions_to_ignore:
            continue
        dimensions_to_use.append(i)
    dimensions_to_use = np.array(dimensions_to_use)

    origData[:, dimensions_to_use] = normalizedData

    # potentially ineficient, but only done once per experimentdata_conversions
    stdMat = data_std.reshape((1, D))
    stdMat = np.repeat(stdMat, T, axis=0)
    meanMat = data_mean.reshape((1, D))
    meanMat = np.repeat(meanMat, T, axis=0)
    origData = np.multiply(origData, stdMat) + meanMat
    return origData


def revert_output_format(poses, data_mean, data_std, dim_to_ignore):
    """
Converts the output of the neural network to a format that is more easy to
manipulate for, e.g. conversion to other format or visualization

Args
  poses: Poses, given in an array of shape (n, dim_poses) where n denotes the number of overall samples
Returns
  poses_out: A tensor of size (batch_size, seq_length, dim) output. Each
  batch is an n-by-d sequence of poses.data_conversions
"""
    # seq_len = len(poses)
    # if seq_len == 0:
    #     return []
    #
    # batch_size, dim = poses[0].shape
    #
    # poses_out = np.concatenate(poses)
    # poses_out = np.reshape(poses_out, (seq_len, batch_size, dim))
    # poses_out = np.transpose(poses_out, [1, 0, 2])

    # poses_out_list = []

    poses_unnorm = unNormalizeData(poses, data_mean, data_std, dim_to_ignore)

    return poses_unnorm


def readCSVasFloat(filename):
    """
Borrowed from SRNN code. Reads a csv and returns a float matrix.
https://github.com/asheshjain399/NeuralModels/blob/master/neuralmodels/utils.py#L34

Args
  filename: string. Path to the csv file
Returns
  returnArray: the read data in a float32 matrix
"""
    returnArray = []
    lines = open(filename).readlines()
    for line in lines:
        line = line.strip().split(",")
        if len(line) > 0:
            returnArray.append(np.array([np.float32(x) for x in line]))

    returnArray = np.array(returnArray)
    return returnArray


def load_data(path_to_dataset, subjects, actions, one_hot):
    """
Borrowed from SRNN code. This is how the SRNN code reads the provided .txt files
https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/processdata.py#L270

Args
  path_to_dataset: string. directory where the data resides
  subjects: list of numbers. The subjects to load
  actions: list of string. The actions to load
  one_hot: Whether to add a one-hot encoding to the data
Returns
  trainData: dictionary with k:v
    k=(subject, action, subaction, 'even'), v=(nxd) un-normalize.format(
                    , , ,
                )d data
  completeData: nxd matrix with all the data. Used to normlization stats
"""
    nactions = len(actions)

    trainData = {}
    completeData = []
    for subj in tqdm(subjects):
        print(f"processing subject # {subj}")
        for action_idx in tqdm(np.arange(len(actions))):
            print(f"processing action #{action_idx}")
            action = actions[action_idx]

            for subact in [1, 2]:  # subactions

                print(
                    f"Reading subject {subj}, action {action}, subaction {subact}"
                )

                filename = f"{path_to_dataset}/S{subj}/{action}_{subact}.txt"
                action_sequence = readCSVasFloat(filename)

                n, d = action_sequence.shape
                even_list = list(range(0, n, 1))

                if one_hot:
                    # Add a one-hot encoding at the end of the representation
                    the_sequence = np.zeros(
                        (len(even_list), d + nactions), dtype=float
                    )
                    the_sequence[:, 0:d] = action_sequence[even_list, :]
                    the_sequence[:, d + action_idx] = 1
                    trainData[(subj, action, subact, "even")] = the_sequence
                else:
                    trainData[(subj, action, subact, "even")] = action_sequence[
                        even_list, :
                    ]

                if len(completeData) == 0:
                    completeData = copy.deepcopy(action_sequence)
                else:
                    completeData = np.append(
                        completeData, action_sequence, axis=0
                    )

    return trainData, completeData


def normalize_data(data, data_mean, data_std, dim_to_use, actions, one_hot):
    """
Normalize input data by removing unused dimensions, subtracting the mean and
dividing by the standard deviation

Args
  data: nx99 matrix with data to normalize
  data_mean: vector of mean used to normalize the data
  data_std: vector of standard deviation used to normalize the data
  dim_to_use: vector with dimensions used by the model
  actions: list of strings with the encoded actions
  one_hot: whether the data comes with one-hot encoding
Returns
  data_out: the passed data matrix, but normalized
"""
    data_out = {}
    nactions = len(actions)

    if not one_hot:
        # No one-hot encoding... no need to do anything special
        for key in list(data.keys()):
            data_out[key] = np.divide((data[key] - data_mean), data_std)
            data_out[key] = data_out[key][:, dim_to_use]

    else:
        # TODO hard-coding 99 dimensions for un-normalized human poses
        for key in list(data.keys()):
            data_out[key] = np.divide(
                (data[key][:, 0:99] - data_mean), data_std
            )
            data_out[key] = data_out[key][:, dim_to_use]
            data_out[key] = np.hstack((data_out[key], data[key][:, -nactions:]))

    return data_out


def normalization_stats(completeData):
    """"
Also borrowed for SRNN code. Computes mean, stdev and dimensions to ignore.
https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/processdata.py#L33

Args
  completeData: nx99 matrix with data to normalize
Returns
  data_mean: vector of mean used to normalize the data
  data_std: vector of standard deviation used to normalize the data
  dimensions_to_ignore: vector with dimensions not used by the model
  dimensions_to_use: vector with dimensions used by the model
"""
    data_mean = np.mean(completeData, axis=0)
    data_std = np.std(completeData, axis=0)

    dimensions_to_ignore = []
    dimensions_to_use = []

    dimensions_to_ignore.extend(list(np.where(data_std < 1e-4)[0]))
    dimensions_to_use.extend(list(np.where(data_std >= 1e-4)[0]))

    data_std[dimensions_to_ignore] = 1.0

    return data_mean, data_std, dimensions_to_ignore, dimensions_to_use


def _some_variables(use_posInd=False):
    """
  We define some variables that are useful to run the kinematic tree

  Args
    None
  Returns
    parent: 32-long vector with parent-child relationships in the kinematic tree
    offset: 96-long vector with bone lenghts
    rotInd: 32-long list with indices into angles
    expmapInd: 32-long list with indices into expmap angles
  """

    parent = (
        np.array(
            [
                0,
                1,
                2,
                3,
                4,
                5,
                1,
                7,
                8,
                9,
                10,
                1,
                12,
                13,
                14,
                15,
                13,
                17,
                18,
                19,
                20,
                21,
                20,
                23,
                13,
                25,
                26,
                27,
                28,
                29,
                28,
                31,
            ]
        )
        - 1
    )

    offset = np.array(
        [
            0.000000,
            0.000000,
            0.000000,
            -132.948591,
            0.000000,
            0.000000,
            0.000000,
            -442.894612,
            0.000000,
            0.000000,
            -454.206447,
            0.000000,
            0.000000,
            0.000000,
            162.767078,
            0.000000,
            0.000000,
            74.999437,
            132.948826,
            0.000000,
            0.000000,
            0.000000,
            -442.894413,
            0.000000,
            0.000000,
            -454.206590,
            0.000000,
            0.000000,
            0.000000,
            162.767426,
            0.000000,
            0.000000,
            74.999948,
            0.000000,
            0.100000,
            0.000000,
            0.000000,
            233.383263,
            0.000000,
            0.000000,
            257.077681,
            0.000000,
            0.000000,
            121.134938,
            0.000000,
            0.000000,
            115.002227,
            0.000000,
            0.000000,
            257.077681,
            0.000000,
            0.000000,
            151.034226,
            0.000000,
            0.000000,
            278.882773,
            0.000000,
            0.000000,
            251.733451,
            0.000000,
            0.000000,
            0.000000,
            0.000000,
            0.000000,
            0.000000,
            99.999627,
            0.000000,
            100.000188,
            0.000000,
            0.000000,
            0.000000,
            0.000000,
            0.000000,
            257.077681,
            0.000000,
            0.000000,
            151.031437,
            0.000000,
            0.000000,
            278.892924,
            0.000000,
            0.000000,
            251.728680,
            0.000000,
            0.000000,
            0.000000,
            0.000000,
            0.000000,
            0.000000,
            99.999888,
            0.000000,
            137.499922,
            0.000000,
            0.000000,
            0.000000,
            0.000000,
        ]
    )

    offset = offset.reshape(-1, 3)

    rotInd = [
        [5, 6, 4],
        [8, 9, 7],
        [11, 12, 10],
        [14, 15, 13],
        [17, 18, 16],
        [],
        [20, 21, 19],
        [23, 24, 22],
        [26, 27, 25],
        [29, 30, 28],
        [],
        [32, 33, 31],
        [35, 36, 34],
        [38, 39, 37],
        [41, 42, 40],
        [],
        [44, 45, 43],
        [47, 48, 46],
        [50, 51, 49],
        [53, 54, 52],
        [56, 57, 55],
        [],
        [59, 60, 58],
        [],
        [62, 63, 61],
        [65, 66, 64],
        [68, 69, 67],
        [71, 72, 70],
        [74, 75, 73],
        [],
        [77, 78, 76],
        [],
    ]

    # definitions are originating from matlab file --> bring them to zero based indexing
    rotInd = [[e - 1 for e in s if len(s) > 0] for s in rotInd]
    posInd = [0, 1, 2] if use_posInd else None

    expmapInd = np.split(np.arange(4, 100) - 1, 32)

    return parent, offset, rotInd, expmapInd, posInd


def apply_affine_transform(x, M):
    """
    Returns affine transformation R*x + t where M = [R,t] \in R^3x4
    :param x:
    :param M:
    :return:
    """
    is1d = len(x.shape) == 1
    if is1d:
        x = np.expand_dims(x, axis=0)

    x_hom = np.concatenate(
        [x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=-1
    )
    x_out = x_hom @ M.T
    if is1d:
        x_out = np.squeeze(x_out, axis=0)
    return x_out


def fkl(
    angles, parent, offset, rotInd, expmapInd, posInd=None, use_euler=False
):
    """
  Convert joint angles and bone lenghts into the 3d points of a person.

  Args
    angles: vector with 3d position and 3d joint angles in expmap format (99-long) if use_euler=False, or in bvh-motion-capture format (78-long) if use_euler=True
    parent: 32-long vector with parent-child relationships in the kinematic tree
    offset: 96-long vector with bone lenghts
    rotInd: 32-long list with indices into angles
    expmapInd: 32-long list with indices into expmap angles
  Returns
    xyz: 32x3 3d points that represent a person in 3d space
  """
    if use_euler:
        assert len(angles) == 78
    else:
        assert len(angles) == 99

    # Structure that indicates parents for each joint
    njoints = 32
    xyzStruct = [dict() for x in range(njoints)]

    for i in np.arange(njoints):

        if use_euler:
            current_eulers = (
                np.asarray([0.0, 0.0, 0.0])
                if len(rotInd[i]) == 0
                else angles[rotInd[i]]
            )
            thisRotation = euler_to_rotation_matrix(current_eulers, deg=True)
        else:
            r = angles[expmapInd[i]]
            thisRotation = expmap2rotmat(r)

        if parent[i] == -1:  # Root node
            if posInd is not None:
                # note this is currently not in expmap format, since it's a translation and no rotation
                thisPosition = np.array(
                    [angles[posInd[0]], angles[posInd[1]], angles[posInd[2]]]
                )
            else:
                thisPosition = np.asarray([0.0, 0.0, 0.0])
            xyzStruct[i]["rotation"] = thisRotation
            xyzStruct[i]["xyz"] = (
                np.reshape(offset[i, :], (1, 3)) + thisPosition
            )
        else:
            xyzStruct[i]["xyz"] = (offset[i, :]).dot(
                xyzStruct[parent[i]]["rotation"]
            ) + xyzStruct[parent[i]]["xyz"]
            xyzStruct[i]["rotation"] = thisRotation.dot(
                xyzStruct[parent[i]]["rotation"]
            )

    xyz = [xyzStruct[i]["xyz"] for i in range(njoints)]
    xyz = np.array(xyz).squeeze()

    return np.reshape(xyz, [-1])


def revert_coordinate_space(channels, R0, T0):
    """
    Aranges the translation and rotation of the root joint such that the series of poses appears
    to well aranged for visualization in the corrdinate frame
    :param channels: poses, in expmap format; shape (seq_len,99)
    :param R0: initial rotation of root joint
    :param T0: initial translation of root joint
    :param use_euler: whether to use euler angle parametrization or expmap representation
    :return: series of aranged poses
    """
    n, d = channels.shape

    channels_rec = copy.copy(channels)
    R_prev = R0
    T_prev = T0
    rootRotInd = np.arange(3, 6)

    # Loop through the passed posses
    for ii in range(n):

        R_diff = expmap2rotmat(channels[ii, rootRotInd])
        R = R_diff.dot(R_prev)

        channels_rec[ii, rootRotInd] = rotmat2expmap(R)
        T = T_prev + (
            (R_prev.T).dot(np.reshape(channels[ii, :3], [3, 1]))
        ).reshape(-1)
        channels_rec[ii, :3] = T

        T_prev = T
        R_prev = R

    return channels_rec


def kinematic_tree():

    # define mappings
    mappings = {
        "name": str,
        "id": int,
        "offset": float,
        "parent": int,
        "order": str,
        "rotInd": int,
        "children": int,
    }

    p = path.split(path.abspath(__file__))[0]

    tree = ET.parse(path.join(p, "metadata.xml"))
    root = tree.getroot()

    # angles_dict = etree_to_dict(angles)
    skel_tree = root.find("skel_angles")
    kin_tree = {
        "root": "Hips",
        "name": [],
        "id": [],
        "offset": [],
        "parent": [],
        "order": [],
        "rotInd": [],
        "children": [],
    }
    for i, tr in enumerate(skel_tree):
        if not tr.tag == "tree":
            continue
        for i, item in enumerate(tr):
            childs = list(item.iter())

            if i == 0:
                # x,y,z

                posInd = np.asarray(
                    list(map(int, childs[21].text[1:-1].split())), dtype=np.int
                )
                # I HATE MATLAB INDEXING
                kin_tree["posInd"] = {
                    "ids": [posInd[0] - 1, posInd[1] - 1, posInd[2] - 1],
                    "order": "xyz",
                }

            for c in childs:
                if c.tag in mappings.keys():
                    if c.text != "None" and c.text is not None:
                        kin_tree[c.tag].append(
                            list(map(mappings[c.tag], c.text[1:-1].split()))
                            if c.text.startswith("[")
                            else mappings[c.tag](c.text)
                        )
                    else:
                        kin_tree[c.tag].append([])

    kin_tree["order"] = list(
        map(lambda x: None if x == [] else x, kin_tree["order"])
    )
    # same (0-based) indexing is in _some_variables()-function
    kin_tree["parent"] = [e - 1 for e in kin_tree["parent"]]
    kin_tree["parent"] = np.asarray(kin_tree["parent"])
    kin_tree["offset"] = np.asarray(kin_tree["offset"]) * 10
    # kin_tree["offset"] = [[e * 10 for e in r] for r in kin_tree["offset"]]
    kin_tree["expmapInd"] = np.split(np.arange(4, 100) - 1, 32)
    # did I already mention that I HATE MATLAB INDEXING?!!!!!!!!!!!!!!
    kin_tree["rotInd"] = [
        [e - 1 for e in s if len(s) > 0] for s in kin_tree["rotInd"]
    ]
    kin_tree["children"] = [
        [e - 1 for e in s] if isinstance(s, list) else s - 1
        for s in kin_tree["children"]
    ]

    # this offset is more accurate when kps are projected to the image plane
    offset = np.array(
        [
            0.000000,
            0.000000,
            0.000000,
            -132.948591,
            0.000000,
            0.000000,
            0.000000,
            -442.894612,
            0.000000,
            0.000000,
            -454.206447,
            0.000000,
            0.000000,
            0.000000,
            162.767078,
            0.000000,
            0.000000,
            74.999437,
            132.948826,
            0.000000,
            0.000000,
            0.000000,
            -442.894413,
            0.000000,
            0.000000,
            -454.206590,
            0.000000,
            0.000000,
            0.000000,
            162.767426,
            0.000000,
            0.000000,
            74.999948,
            0.000000,
            0.100000,
            0.000000,
            0.000000,
            233.383263,
            0.000000,
            0.000000,
            257.077681,
            0.000000,
            0.000000,
            121.134938,
            0.000000,
            0.000000,
            115.002227,
            0.000000,
            0.000000,
            257.077681,
            0.000000,
            0.000000,
            151.034226,
            0.000000,
            0.000000,
            278.882773,
            0.000000,
            0.000000,
            251.733451,
            0.000000,
            0.000000,
            0.000000,
            0.000000,
            0.000000,
            0.000000,
            99.999627,
            0.000000,
            100.000188,
            0.000000,
            0.000000,
            0.000000,
            0.000000,
            0.000000,
            257.077681,
            0.000000,
            0.000000,
            151.031437,
            0.000000,
            0.000000,
            278.892924,
            0.000000,
            0.000000,
            251.728680,
            0.000000,
            0.000000,
            0.000000,
            0.000000,
            0.000000,
            0.000000,
            99.999888,
            0.000000,
            137.499922,
            0.000000,
            0.000000,
            0.000000,
            0.000000,
        ]
    )

    offset = offset.reshape(-1, 3)

    kin_tree["offset"] = offset

    return kin_tree


def camera_projection(poses, camera_parameters):
    """

    :param poses:
    :param camera_parameters: camera parameters as (f_x, x_0, f_y, y_0)
    :return:
    """
    camera_mat = np.asarray(
        [
            [camera_parameters[0], 0.0, camera_parameters[1]],
            [0.0, camera_parameters[2], camera_parameters[3]],
            [0.0, 0.0, 1.0],
        ]
    )
    if len(poses.shape) not in [2, 3]:
        raise Exception("Poses array has to be of dim 2 or 3.")
    # divide by z
    poses3d = poses / np.expand_dims(poses[..., -1], axis=-1)
    poses2d_homogenous = poses3d @ camera_mat.T
    # poses2d_homogenous = np.moveaxis(poses2d_homogenous,[0,1],[1,0])
    return poses2d_homogenous[..., :-1]


class Ax3DPose(object):
    def __init__(
        self,
        ax,
        dataset: BaseDataset = None,
        ticks=True,
        marker_color=None,
        lcolor="#3498db",
        rcolor="#e74c3c",
        limits=None,
    ):
        """
    Create a 3d pose visualizer that can be updated with new poses.

    Args
      ax: 3d axis to plot the 3d pose on
      lcolor: String. Colour for the left part of the body
      rcolor: String. Colour for the right part of the body
    """
        if marker_color is not None:
            self.marker_color = marker_color

        # Start and endpoints of our representation

        # self.I = (
        #     np.array([1, 2, 3, 1, 7, 8, 1, 13, 14, 15, 14, 18, 19, 14, 26, 27])
        #     - 1
        # )
        # self.J = (
        #     np.array([2, 3, 4, 7, 8, 9, 13, 14, 15, 16, 18, 19, 20, 26, 27, 28])
        #     - 1
        # )
        # # Left / right indicator
        # self.LR = np.array(
        #     [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool
        # )
        self.I = [i[0] for i in dataset.joint_model.total_relative_joints]
        self.J = [i[1] for i in dataset.joint_model.total_relative_joints]
        self.LR = []
        self.ax = ax

        self.n_kps = len(dataset.joint_model.kps_to_use)

        vals = np.zeros((self.n_kps, 3))
        self.limits = limits

        # Make connection matrix
        self.plots = []
        for i in np.arange(len(self.I)):
            x = np.array([vals[self.I[i], 0], vals[self.J[i], 0]])
            y = np.array([vals[self.I[i], 1], vals[self.J[i], 1]])
            z = np.array([vals[self.I[i], 2], vals[self.J[i], 2]])
            if self.marker_color is not None:
                self.plots.append(
                    self.ax.plot(x, y, z, lw=2, c=self.marker_color)
                )
            else:
                self.plots.append(
                    self.ax.plot(
                        x, y, z, lw=2, c=lcolor if self.LR[i] else rcolor
                    )
                )
        self.ticks = ticks
        if self.ticks:
            self.ax.set_xlabel("x")
            self.ax.set_ylabel("y")
            self.ax.set_zlabel("z")
        else:
            self.hide_axes()

    def hide_axes(self):
        # remove grid
        self.ax.grid(False)
        # remove axis
        self.ax.set_axis_off()
        # remomve labels
        self.ax.set_label("")
        # remove surrounding box
        self.ax.set_frame_on(False)

    def update(self, channels, lcolor="#3498db", rcolor="#e74c3c"):
        """
    Update the plotted 3d pose.

    Args
      channels: 96-dim long np array. The pose to plot.
      lcolor: String. Colour for the left part of the body.
      rcolor: String. Colour for the right part of the body.
    Returns
      Nothing. Simply updates the axis with the new pose.
    """

        if hasattr(self, "marker_color"):
            lcolor = rcolor = self.marker_color

        if len(channels.shape) < 3:
            vals = np.reshape(channels, (self.n_kps, -1))
        else:
            vals = channels

        for i in np.arange(len(self.I)):
            x = np.array([vals[self.I[i], 0], vals[self.J[i], 0]])
            y = np.array([vals[self.I[i], 1], vals[self.J[i], 1]])
            z = np.array([vals[self.I[i], 2], vals[self.J[i], 2]])
            self.plots[i][0].set_xdata(x)
            self.plots[i][0].set_ydata(y)
            self.plots[i][0].set_3d_properties(z)
            self.plots[i][0].set_color(lcolor if len(self.LR) == len(self.I) else rcolor)

        r = .75
        xroot, yroot, zroot = vals[0, 0], vals[0, 1], vals[0, 2]
        if self.limits is None:

            self.ax.set_xlim3d([-r + xroot, r + xroot])
            self.ax.set_zlim3d([-r + zroot, r + zroot])
            self.ax.set_ylim3d([-r + yroot, r + yroot])
        else:
            r = 0.2
            self.ax.set_xlim3d(
                [-r + self.limits["x"][0], r + self.limits["x"][1]]
            )
            self.ax.set_ylim3d(
                [-r + self.limits["y"][0], r + self.limits["y"][1]]
            )
            self.ax.set_zlim3d(
                [-r + self.limits["z"][0], r + self.limits["z"][1]]
            )

        if not self.ticks:
            self.hide_axes()


def convert_to_3d(poses_as_angles, kinematic_tree, swap_yz):
    poses_3d = []
    for pose in poses_as_angles:
        poses_3d.append(
            fkl(
                pose,
                kinematic_tree["parent"],
                kinematic_tree["offset"],
                kinematic_tree["rotInd"],
                kinematic_tree["expmapInd"],
                kinematic_tree["posInd"]["ids"],
            )
        )

    poses_3d = np.stack(poses_3d, axis=0)
    poses_3d = poses_3d.reshape((poses_3d.shape[0], 32, -1))
    if swap_yz:
        poses_3d = poses_3d[:, :, [0, 2, 1]]

    return poses_3d

def project_onto_image_plane(
    poses,
    image_sizes,
    color,
    text:str,
    org:tuple,
    extrinsics,
    intrinsics,
    dataset: BaseDataset,
    target_size: tuple = None,
    background_color=None,
    synth_model: VunetOrg = None,  # ,thicken=False
    app_img=None,
    thickness=3,
    font_size=2,
    crop=False,
    target_width=128,
    yield_mask=False,
    overlay=False,
    cond_id=None
):
    """

    :param poses:
    :param image_sizes:
    :param extrinsics:
    :param intrinsics:
    :param color: the color in which to display the generated poses
    :param joint_model:
    :param target_size:
    :param app_img:
    :return:
    """
    if color == None:
        color = [[255, 0, 0],[255, 0, 0],[0, 0, 255],[0, 0, 255],[0, 0, 255],[255, 0, 0],[0, 0, 255],[0, 0, 255],[0, 0, 255],[0, 0, 255],[0, 0, 255],[255, 0, 0],[0, 0, 255],[0, 0, 255],[255, 0, 0],[255, 0, 0]]
        assert len(color) == len(dataset.joint_model.total_relative_joints)

    elif len(color)==2:
        color = [color[0], color[0], color[1], color[1],
                 color[1], color[0], color[1], color[1],
                 color[1], color[1], color[1], color[0],
                 color[1], color[1], color[0], color[0]]
        assert len(color) == len(dataset.joint_model.total_relative_joints)


    if overlay and len(color) != 2:
        color = [[255, 0, 0],[255, 0, 0],[0, 102, 255],[0, 102, 255],[0, 102, 255],[255, 0, 0],[0, 102, 255],[0, 102, 255],[0, 102, 255],[0, 102, 255],[0, 102, 255],[255, 0, 0],[0, 102, 255],[0, 102, 255],[255, 0,    0],[255, 0, 0]]
    yield_mask &= crop
    masks = None

    if background_color is None:
        background_color = 0

    if synth_model is not None:
        assert app_img is not None
        img_size = [synth_model.spatial_size, synth_model.spatial_size]
        size_arr = np.full((1, 2), synth_model.spatial_size,
                           dtype=np.float)
        dev = app_img.get_device() if app_img.get_device() >= 0 else "cpu"

    imgs = []
    imgs_rgb = []
    for id_count,(p, s) in enumerate(zip(poses, image_sizes)):
        pose_c = apply_affine_transform(p, extrinsics)
        pose_i = camera_projection(pose_c, intrinsics)

        img = add_joints_to_img(
            np.full(
                (s[0], s[1], 3), fill_value=background_color, dtype=np.uint8
            ),
            pose_i[dataset.joint_model.kps_to_use] if dataset.small_joint_model else pose_i,
            dataset.joint_model.total_relative_joints,
            color_kps=color,
            color_joints=color,
        )
        if cond_id is not None:
            if id_count < cond_id:
                text = text + "; COND"
            else:
                text = text + "; PRED"

        img = cv2.putText(img,text,org,cv2.FONT_HERSHEY_SIMPLEX,font_size,(0,0,0),thickness=thickness)
        if synth_model is not None:
            # rescale joints
            joint_scaling = size_arr / np.expand_dims(s, axis=0)
            pose_i_rescaled = pose_i * joint_scaling
            stickman = make_joint_img(
                img_size,
                pose_i_rescaled,
                dataset.joint_model,
                line_colors=dataset.line_colors,
                scale_factor=dataset.stickman_scale,
            )
            stickman = (
                dataset.stickman_transforms(stickman).unsqueeze(dim=0).to(dev)
            )
            with torch.no_grad():
                rgb_img = synth_model.transfer(app_img, stickman)
            rgb_img = (
                (scale_img(rgb_img) * 255.0)
                .permute((0, 2, 3, 1))
                .detach()
                .cpu()
                .numpy()
                .astype(np.uint8)
            )
            # rgb_img = cv2.putText(rgb_img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_size,
            #                   (255, 255, 255), thickness=thickness)
            if crop:
                kps = pose_i[
                    dataset.joint_model.kps_to_use] if dataset.small_joint_model else pose_i
                kps_rescaled = kps * joint_scaling
                center_kps = np.mean(kps_rescaled, axis=0)
                min_x = int(min(max(0, center_kps[0] - target_width // 2),
                                rgb_img.shape[1] - target_width))
                max_x = min_x + target_width
                rgb_img = rgb_img[:, :, min_x:max_x]
            imgs_rgb.append(rgb_img.squeeze(0))


        if target_size is not None:
            scaling = np.asarray(img.shape[:2],dtype=np.float) / np.asarray(target_size,dtype=np.float)
            img = cv2.resize(img, target_size,interpolation=cv2.INTER_NEAREST)
        else:
            scaling = np.ones(2,dtype=np.float)

        if crop:
            kps = pose_i[dataset.joint_model.kps_to_use] if dataset.small_joint_model else pose_i
            kps_rescaled = kps / np.expand_dims(scaling,axis=0)
            center_kps = np.mean(kps_rescaled,axis=0)
            min_x = int(min(max(0, center_kps[0] - target_width// 2),
                        img.shape[1] - target_width))
            max_x = min_x + target_width
            img = img[:,min_x:max_x]
            if masks is None:
                masks = np.expand_dims(np.asarray([min_x,max_x]),axis=0)
            else:
                masks = np.concatenate([masks,np.expand_dims(np.asarray([min_x,max_x]),axis=0)],axis=0)

        imgs.append(img)


    if synth_model is not None and overlay:
        overlays = []
        for stick,r in zip(imgs,imgs_rgb):
            o = np.where(np.expand_dims(np.all(stick==255,axis=2),axis=-1),r,stick)
            overlays.append(o)

    outs = [np.stack(imgs, axis=0)]
    if synth_model is not None:
        outs.append(np.stack(imgs_rgb, axis=0))
    else:
        outs.append(np.zeros_like(outs[0]))

    if yield_mask:
        outs.append(masks)

    if overlay:
        outs.append(np.stack(overlays,axis=0))

    return outs

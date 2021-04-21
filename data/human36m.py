import h5py as h5
import numpy as np
from tqdm.autonotebook import tqdm  #
import torch
from copy import deepcopy
import cv2
import imagesize
from functools import partial
from os import path


from data.data_conversions_3d import (
    kinematic_tree,
    rotmat2expmap,
    euler_to_rotation_matrix,
    normalization_stats,
    revert_output_format,
    apply_affine_transform,
    camera_projection
)
from lib.utils import JointModel, t2p, t3p, t4p, t5p, make_joint_img
from data.base_dataset import BaseDataset

__data_split__ = {
    "train": ["S1", "S5", "S6", "S7", "S8"],
    "test": ["S9", "S11"],
}

__actionID_to_action__ = {
    2: "Directions",
    3: "Discussion",
    4: "Eating",
    5: "Greeting",
    6: "Phoning",
    7: "Posing",
    8: "Purchases",
    9: "Sitting",
    10: "SittingDown",
    11: "Smoking",
    12: "TakingPhoto",
    13: "Waiting",
    14: "Walking",
    15: "WalkingDog",
    16: "WalkTogether",
}


class Human36mDataset(BaseDataset):
    def __init__(
        self, transforms, data_keys, seq_length, mode="train", **kwargs
    ):
        assert mode in ["train", "test"]
        self.small_joint_model = (
            kwargs["small_joint_model"]
            if "small_joint_model" in kwargs.keys()
            else False
        )
        self.action_split_type = kwargs["action_split_type"] if "action_split_type" in kwargs else "default"
        self.valid_keypoint_types = [
            "angle_euler",
            "norm_keypoints",
            "keypoints_3d",
            "keypoints_3d_univ",
            "angle_expmap",
            "angle_world_euler",
            "angle_world_expmap",
            "keypoints_3d_world",
        ]

        if "keypoint_type" in kwargs:
            assert kwargs["keypoint_type"] in self.valid_keypoint_types
            self.keypoint_key = kwargs["keypoint_type"]
        else:
            self.keypoint_key = None

        if self.small_joint_model:
            joint_model = JointModel(
                body=[25, 17, 6, 1],
                right_lines=[(3, 2), (2, 1), (1, 25), (25, 26), (26, 30)],
                left_lines=[(8, 7), (7, 6), (6, 17), (17, 18), (18, 22)],
                head_lines=[],
                face=[],
                rshoulder=25,
                lshoulder=17,
                headup=15,
                kps_to_use=[1, 2, 3, 6, 7, 8, 15, 17, 18, 22, 25, 26, 30],
                total_relative_joints=[
                    [0, 1],
                    [1, 2],
                    [3, 4],
                    [4, 5],
                    [0, 3],
                    [3, 7],
                    [0, 10],
                    [7, 10],
                    [7, 8],
                    [8, 9],
                    [10, 11],
                    [11, 12],
                ],
                right_hand=[],
                left_hand=[],
                head_part=[],
                kp_to_joint=[
                    "r_hip",
                    "r_knee",
                    "r_foot",
                    "l_hip",
                    "l_knee",
                    "l_foot",
                    "head",
                    "l_shoulder",
                    "l_elbow",
                    "l_hand",
                    "r_shoulder",
                    "r_elbow",
                    "r_hand",
                ],
                kps_to_change=[1, 2, 3, 6, 7, 8, 15, 17, 18, 22, 25, 26, 30],
                kps_to_change_rel=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                norm_T=[
                    t3p,
                    t4p,
                    partial(t2p, ids=[25, 26]),
                    partial(t2p, ids=[26, 30]),
                    partial(t2p, ids=[17, 18]),
                    partial(t2p, ids=[18, 22]),
                    partial(t2p, ids=[1, 2]),
                    partial(t2p, ids=[2, 3]),
                    partial(t2p, ids=[6, 7]),
                    partial(t2p, ids=[7, 8]),
                ],
            )
        else:
            # detailed joint model
            joint_model = JointModel(
                body=[1, 25, 13, 17, 6]
                if self.keypoint_key != "keypoints_3d_world"
                else [0, 14, 8, 11, 3],
                right_lines=[(3, 2), (2, 1), (1, 25), (25, 26), (26, 27)]
                if self.keypoint_key != "keypoints_3d_world"
                else [(0, 1), (1, 2), (0, 14), (14, 15), (15, 16)],
                left_lines=[(8, 7), (7, 6), (6, 17), (17, 18), (18, 19)]
                if self.keypoint_key != "keypoints_3d_world"
                else [(3, 4), (4, 5), (3, 11), (11, 12), (12, 13)],
                head_lines=[(13, 14), (14, 15)]
                if self.keypoint_key != "keypoints_3d_world"
                else [(8, 9), (9, 10)],
                face=[],
                rshoulder=25,
                lshoulder=17,
                headup=15,
                kps_to_use=[
                    1,  # 0
                    2,
                    3,  # 2
                    6,
                    7,  # 4
                    8,
                    11,  # 6
                    12,
                    13,  # 8
                    14,
                    15,  # 10
                    17,
                    18,  # 12
                    19,
                    25,  # 14
                    26,
                    27,  # 16
                ],
                total_relative_joints=[
                    [0, 1],
                    [1, 2],
                    [3, 4],
                    [4, 5],
                    [3, 6],
                    [0, 6],
                    [6, 7],
                    [7, 8],
                    [8, 9],
                    [9, 10],
                    [8, 11],
                    [8, 14],
                    [11, 12],
                    [12, 13],
                    [14, 15],
                    [15, 16],
                ],
                right_hand=[],
                left_hand=[],
                head_part=[],
                kp_to_joint=[
                    "r_hip",
                    "r_knee",
                    "r_foot",
                    "l_hip",
                    "l_knee",
                    "l_foot",
                    "pelvis",
                    "thorax",
                    "neck",
                    "nose",
                    "head",
                    "l_shoulder",
                    "l_elbow",
                    "l_wirst",
                    "r_shoulder",
                    "r_elbow",
                    "r_wrist",
                ],
                kps_to_change=[],
                kps_to_change_rel=[],
                norm_T=[
                    t3p,  # head
                    t5p,  # body
                    partial(t2p, ids=[25, 26]),  # right upper arm
                    partial(t2p, ids=[26, 30]),  # right lower arm
                    partial(t2p, ids=[17, 18]),  # left upper arm
                    partial(t2p, ids=[18, 22]),  # left lower arm
                    partial(t2p, ids=[1, 2]),  # right upper leg
                    partial(t2p, ids=[2, 3]),  # right lower leg
                    partial(t2p, ids=[6, 7]),  # left upper leg
                    partial(t2p, ids=[7, 8]),  # left lower leg
                ],
            )
        self.debug = "debug" in kwargs.keys() and kwargs["debug"]

        super().__init__(
            transforms, mode, seq_length, data_keys, joint_model, **kwargs
        )

        self._output_dict.update(
            {
                "intrinsics": self._get_intrinsic_params,
                "intrinsics_paired": partial(
                    self._get_intrinsic_params, use_map_ids=True
                ),
                "extrinsics": self._get_extrinsic_params,
                "extrinsics_paired": partial(
                    self._get_extrinsic_params, use_map_ids=True
                ),
            }
        )

        # self.datapath = (
        #     "/export/scratch/compvis/datasets/human3.6M/processed/all/"
        # )

        # sequence matching only enabled when using 2d keypoints
        if not (
            self.keypoint_key == "norm_keypoints" or self.keypoint_key is None
        ):
            self.prepare_seq_matching = False
            self.use_matched_map_ids = True

        self.train_synthesis = (
            kwargs["train_synthesis"] if "train_synthesis" in kwargs else False
        )

        self.use_3d_for_stickman = (
            kwargs["use_3d_for_stickman"]
            if "use_3d_for_stickman" in kwargs
            else False
        )
        if self.use_3d_for_stickman:
            self._output_dict["stickman"] = self._get_stickman_from_3d
            assert self.keypoint_key in [
                "angle_world_expmap",
                "keypoints_3d_world",
            ]
            if self.keypoint_key == "keypoints_3d_world":
                assert not self.small_joint_model
            assert self.train_synthesis

        self.label_type = "action"

        self.single_app = "target_apperance" in kwargs
        self.target_app = (
            kwargs["target_appearance"] if self.single_app else None
        )

        # adjust action categories used for training
        if "all_actions" in kwargs and kwargs["all_actions"]:
            self.actions_to_use = list(__actionID_to_action__.values())
        else:
            self.actions_to_use = (
                kwargs["actions_to_use"]
                if "actions_to_use" in kwargs
                and len(kwargs["actions_to_use"]) > 1
                else [
                    "Greeting",
                    "Purchases",
                    "SittingDown",
                    "TakingPhoto",
                    "Walking",
                    "WalkingDog",
                    "WalkTogether",
                ]
            )

        self.actions_to_discard = (
            kwargs["actions_to_discard"]
            if "actions_to_discard" in kwargs
            else None
        )

        self.prepare_seq_matching = (
            kwargs["prepare_seq_matching"]
            if "prepare_seq_matching" in kwargs
            else False
        )

        # if self.motion_based_sampling:
        #     assert not self.prepare_seq_matching

        # load data (default = h36m_full)
        dataset_version = "dataset_version" in kwargs
        dataset_version = (
            kwargs["dataset_version"] if dataset_version else "h36m_full"
        )
        self._load_data(self.datapath, dataset_version)

        # set appearance mapping ids
        if mode == "train":
            self.datadict["map_ids"] = np.zeros_like(
                self.datadict["p_ids"], dtype=np.int
            )

            self.complete_datadict["map_ids"] = np.zeros_like(
                self.complete_datadict["p_ids"], dtype=np.int
            )

        elif mode == "test":
            ids = np.linspace(
                0,
                len(self.datadict["p_ids"]),
                len(self.datadict["p_ids"]),
                endpoint=False,
                dtype=np.int64,
            )
            # if not self.single_app:
            np.random.shuffle(ids)
            self.datadict["map_ids"] = ids

            ids_c = np.linspace(
                0,
                len(self.complete_datadict["p_ids"]),
                len(self.complete_datadict["p_ids"]),
                endpoint=False,
                dtype=np.int64,
            )
            # if not self.single_app:
            np.random.shuffle(ids_c)
            self.complete_datadict["map_ids"] = ids_c

        # self.transforms = transforms

        # set unique appearance image for inference
        if self.single_app:
            # this contains the name of the image path of the target image
            target_app_path = self._get_target_app_image()
            target_id = np.where(self.datadict["img_paths"] == target_app_path)[
                0
            ]

            self.datadict["map_ids"] = np.full_like(
                self.datadict["map_ids"], target_id
            )

        if self.use_matched_map_ids:  # or self.motion_based_sampling:
            self.matched_map_ids = np.zeros_like(self.datadict["map_ids"])
            unique_aids = np.unique(self.datadict["action"])
            for id in unique_aids:
                valid_ids = np.nonzero(np.equal(self.datadict["action"], id))[0]
                map_ids = deepcopy(valid_ids)
                np.random.shuffle(map_ids)
                self.matched_map_ids[valid_ids] = map_ids

        # get sequence lengths per video
        self._get_sequence_end_ids()
        self._get_sequence_start_ids()
        self._check_seq_len_and_frame_lag()
        self.action_id_to_action = __actionID_to_action__
        # if self.motion_based_sampling:
        #     # compute motion scores which are required for motion based sampling
        #     self._compute_motion_scores()

        self.resample_map_ids()

        # compute image_shapes
        uniqe_vids, first_vid_occs = np.unique(
            self.datadict["v_ids"], return_index=True
        )
        self.image_shapes = {
            vid: imagesize.get(self.datadict["img_paths"][fo])
            for (vid, fo) in zip(uniqe_vids, first_vid_occs)
        }

        print(
            f"Constructed Human3.6m Dataset in {self.mode}-mode, which consists of {self.__len__()} samples."
        )

    def get_test_app_images(self) -> dict:
        return {
            "S1": path.join(
                self.datapath,
                "S1/Walking-2/imageSequence/54138969/img_000189.jpg",
            ),
            "S5": path.join(
                self.datapath,
                "S5/Walking-1/imageSequence/55011271/img_000048.jpg",
            ),
            "S6": path.join(
                self.datapath,
                "S6/Walking-2/imageSequence/55011271/img_000206.jpg",
            ),
            "S7": path.join(
                self.datapath,
                "S7/Walking-2/imageSequence/58860488/img_000277.jpg",
            ),
            "S8": path.join(
                self.datapath,
                "S8/Walking-1/imageSequence/60457274/img_000001.jpg",
            ),
            "S9": path.join(
                self.datapath,
                "S9/Walking-1/imageSequence/58860488/img_000321.jpg",
            ),
            "S11": path.join(
                self.datapath,
                "S11/Walking-2/imageSequence/58860488/img_000193.jpg",
            ),
        }

    def __len__(self):
        return self.datadict["img_paths"].shape[0]

    def _get_target_app_image(self):
        if not self.target_app in self.get_test_app_images().keys():
            raise TypeError(
                f"The target appearance has to be a string object in {list(self.get_test_app_images().keys())}."
            )

        return self.get_test_app_images()[self.target_app]

    ### dataset loading ###
    def _load_data(self, basepath, version):
        if version == "h36m_small20":
            self._load_h36m_small20(basepath)
        elif version == "h36m_full":
            self._load_h36m_full(basepath)
        else:
            raise Exception(f"Dataset version not valid.")

    # human3.6m full dataset
    def _load_h36m_full(self, basepath):
        # load and convert meta data
        # if self.normalize_keypoints:
        attribute_mapping = {
            "frame_path": "img_paths",
            "pose_2d": "keypoints",
            "subject": "p_ids",
            "frame": "f_ids",  # frame id, 0,....,len(seq)
            "action": "action",
            "subaction": "subaction",
            "pose_normalized_2d": "norm_keypoints",
            "camera": "camera_id",
            # "angle_3d": "angle_euler",
            "image_size": "image_size",
            # "intrinsics": "intrinsics",
            "intrinsics_univ": "intrinsics_univ",
            "pose_3d": "keypoints_3d",
            # "pose_3d_univ": "keypoints_3d_univ",
            # "angle_3d_expmap": "angle_expmap",
            # "angle_3d_world": "angle_world_euler",
            "pose_3d_world": "keypoints_3d_world",
            # "angle_expmap_world": "angle_world_expmap",
            # "extrinsics": "extrinsics",
            "extrinsics_univ": "extrinsics_univ",
        }
        h5_file = path.join(basepath, "annot_export.h5")
        # else:
        #     attribute_mapping = {
        #         "frame_path": "img_paths",
        #         "keypoints": "keypoints",
        #         "subject": "p_ids",
        #         "fid": "f_ids",  # frame id, 0,....,len(seq)
        #         # 'frame': 'frame',  # original frame name, e.g. 0001.png
        #         "action": "action",
        #         "subaction": "subaction",
        #     }
        #     h5_file = path.join(basepath, "annot.h5")
        with h5.File(h5_file, "r") as f:
            for k in tqdm(f.keys(), desc="Constructing Human3.6m dataset..."):
                if k not in self.valid_keypoint_types or k == self.keypoint_key:
                    # if self.debug:
                    #     self.datadict[attribute_mapping[k]] = np.asarray(f[k])[:100000]
                    # else:
                    # self.datadict[attribute_mapping[k]] = np.asarray(f[k])
                    self.datadict[k] = np.asarray(f[k])
        # load kinematic tree

        if self.debug:
            # load small dataset
            n_samples_per_person = len(__actionID_to_action__) * 100
            unique_pids, pids_first = np.unique(
                self.datadict["p_ids"], return_index=True
            )
            unique_aids = np.unique(self.datadict["action"])
            ids = np.zeros(
                n_samples_per_person * unique_pids.shape[0], dtype=np.int
            )
            count = 0
            for pid in tqdm(
                unique_pids,
                desc=f"Debug-mode: Generating small data set which contains {ids.shape[0]} samples",
            ):
                for aid in unique_aids:
                    ids[count : count + 100] = np.nonzero(
                        np.logical_and(
                            self.datadict["action"] == aid,
                            self.datadict["p_ids"] == pid,
                        )
                    )[0][:100]
                    count += 100

            self.datadict = {
                key: self.datadict[key][ids]
                for key in self.datadict
                if self.datadict[key].size > 0
            }

        self.kinematic_tree = kinematic_tree()

        # get unique person ids
        self.person_ids = list(np.unique(self.datadict["p_ids"]))

        # add base path to img_paths
        base_path_tmp = "/" + path.join(
            *self.datapath.split("/")[
                0 : np.where(np.array(self.datapath.split("/")) == "processed")[
                    0
                ][0]
            ]
        )
        self.datadict["img_paths"] = [
            path.join(base_path_tmp, p.decode("utf-8"))
            for p in self.datadict["img_paths"]
        ]
        self.datadict["img_paths"] = np.asarray(self.datadict["img_paths"])

        # for k in self.datadict:
        #     self.datadict[k] = np.asarray(self.datadict[k])

        self.datadict["f_ids"] = self.datadict["f_ids"] - 1

        self.complete_datadict = deepcopy(self.datadict)
        # reduce dataset size if world coordinates (angles or poses) are used
        if "world" in self.keypoint_key and not self.train_synthesis:
            target_cam_id = np.unique(self.datadict["camera_id"])[0]
            t_sample_ids = self.datadict["camera_id"] == target_cam_id
            for key in self.datadict:
                if self.datadict[key].size > 0:
                    self.datadict[key] = self.datadict[key][t_sample_ids]

        pre_vids = (
            1000000 * self.datadict["camera_id"]
            + 10000 * self.datadict["action"]
            + 1000 * self.datadict["subaction"]
            + self.datadict["p_ids"]
        )

        vid_mapping = {u: i for i, u in enumerate(np.unique(pre_vids))}

        self.datadict["v_ids"] = np.full_like(pre_vids, -1)

        for key in vid_mapping:
            self.datadict["v_ids"][pre_vids == key] = vid_mapping[key]

        pre_vids_c = (
            1000000 * self.complete_datadict["camera_id"]
            + 10000 * self.complete_datadict["action"]
            + 1000 * self.complete_datadict["subaction"]
            + self.complete_datadict["p_ids"]
        )

        vid_mapping = {u: i for i, u in enumerate(np.unique(pre_vids_c))}

        self.complete_datadict["v_ids"] = np.full_like(pre_vids_c, -1)

        for key in vid_mapping:
            self.complete_datadict["v_ids"][pre_vids_c == key] = vid_mapping[
                key
            ]

        assert not np.any(self.datadict["v_ids"] == -1)
        if (
            "angle" in self.keypoint_key
            or self.keypoint_key == "keypoints_3d_world"
        ):
            keypoints_shape = self.datadict[self.keypoint_key].shape
            if self.keypoint_key == "keypoints_3d_world":
                self.datadict[self.keypoint_key] = (
                    self.datadict[self.keypoint_key] / 1000.0
                )  # m to comply with test setting
                self.datadict["extrinsics_univ"][:, :, -1] = (
                    self.datadict["extrinsics_univ"][:, :, -1] / 1000
                )  # translation mm to m
                self.complete_datadict[self.keypoint_key] = (
                    self.complete_datadict[self.keypoint_key] / 1000.0
                )  # m to comply with test setting
                self.complete_datadict["extrinsics_univ"][:, :, -1] = (
                    self.complete_datadict["extrinsics_univ"][:, :, -1] / 1000
                )
                self.datadict[self.keypoint_key] = self.datadict[
                    self.keypoint_key
                ][:, self.joint_model.kps_to_use]
                self.complete_datadict[
                    self.keypoint_key
                ] = self.complete_datadict[self.keypoint_key][
                    :, self.joint_model.kps_to_use
                ]
                self.datadict[self.keypoint_key] = self.datadict[
                    self.keypoint_key
                ].reshape(keypoints_shape[0], -1)
                self.complete_datadict[
                    self.keypoint_key
                ] = self.complete_datadict[self.keypoint_key].reshape(
                    self.complete_datadict[self.keypoint_key].shape[0], -1
                )
            self.data_mean, self.data_std, self.dim_to_ignore, self.dim_to_use = normalization_stats(
                self.datadict[self.keypoint_key]
            )

            # normalize keypoints
            self.datadict[self.keypoint_key] = self.__normalize_poses(
                self.datadict[self.keypoint_key]
            )
            self.complete_datadict[self.keypoint_key] = self.__normalize_poses(
                self.complete_datadict[self.keypoint_key]
            )

            self.maxs_normalized = np.expand_dims(
                np.expand_dims(
                    np.expand_dims(
                        np.asarray(
                            [
                                np.amax(
                                    self.datadict[self.keypoint_key][:, ::3]
                                ),
                                np.amax(
                                    self.datadict[self.keypoint_key][:, 1::3]
                                ),
                                np.amax(
                                    self.datadict[self.keypoint_key][:, 2::3]
                                ),
                            ]
                        ),
                        axis=0,
                    ),
                    axis=0,
                ),
                axis=0,
            )
            self.mins_normalized = np.expand_dims(
                np.expand_dims(
                    np.expand_dims(
                        np.asarray(
                            [
                                np.amin(
                                    self.datadict[self.keypoint_key][:, ::3]
                                ),
                                np.amin(
                                    self.datadict[self.keypoint_key][:, 1::3]
                                ),
                                np.amin(
                                    self.datadict[self.keypoint_key][:, 2::3]
                                ),
                            ]
                        ),
                        axis=0,
                    ),
                    axis=0,
                ),
                axis=0,
            )

        # get data split
        if self.overall_split:
            print("Using overall datasplit....")
            self._make_overall_split()

        else:
            split_indices = set(self._get_split_full()[self.mode])
            for k, v in tqdm(
                self.datadict.items(),
                desc="Selecting desired subset of overall data...",
            ):
                self.datadict[k] = np.asarray(
                    [p for i, p in enumerate(v) if i in split_indices]
                )

        # select or discard individual action categories
        if (
            self.actions_to_discard is not None
            or self.actions_to_use is not None
        ):
            if (
                self.actions_to_discard is not None
                and self.actions_to_use is not None
            ):
                raise ValueError(
                    "Please only consider actions_to_use OR actions_to_discard"
                )

            if self.actions_to_discard is not None:  # discard actions ...
                indices_to_use = [
                    i
                    for i, e in enumerate(self.datadict["action"])
                    if __actionID_to_action__[e] not in self.actions_to_discard
                ]
            elif self.actions_to_use is not None:  # select actions ...
                indices_to_use = [
                    i
                    for i, e in enumerate(self.datadict["action"])
                    if __actionID_to_action__[e] in self.actions_to_use
                ]

            # filter
            indices_to_use_set = set(indices_to_use)
            for k, v in tqdm(
                self.datadict.items(), desc="Filtering actions..."
            ):
                self.datadict[k] = np.asarray(
                    [p for i, p in enumerate(v) if i in indices_to_use_set]
                )
            print(
                f'Actions to be used: {[__actionID_to_action__[i] for i in np.unique(self.datadict["action"]).tolist()]}'
            )

            if self.prepare_seq_matching:
                self.pose_encodings = self.pose_encodings[
                    np.asarray(indices_to_use)
                ]

        # get sequences per action if required
        if self.prepare_seq_matching:
            seqs_per_action = dict()
            v_id_curr = self.datadict["v_ids"][0]
            curr_seq = []
            for k, v_id in tqdm(
                enumerate(self.datadict["v_ids"]),
                desc="Get sequences per action...",
            ):

                if v_id == v_id_curr:
                    curr_seq.append(k)
                    action_id = self.datadict["action"][k]
                else:
                    if action_id not in seqs_per_action.keys():
                        seqs_per_action[action_id] = []
                    seqs_per_action[action_id].append(curr_seq)
                    curr_seq = []
                    v_id_curr = v_id
            self.seqs_per_action = seqs_per_action

    def _get_split_full(self):
        if self.use_person_split:
            _data_split_int_ = {"train": [1, 5, 6, 7, 8], "test": [9, 11]}
            target_data = self.datadict["p_ids"]
        else:
            if self.action_split_type == "generalize_sitting":
                _data_split_int_ = {
                    "train": [2, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 16],
                    "test": [9, 8, 10],
                }
            elif self.action_split_type == "generalize_walking":
                _data_split_int_ = {
                    "train": [2, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 16],
                    "test": [14, 15, 16],
                }
            else:
                _data_split_int_ = {
                    "train": [2, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 16],
                    "test": [8, 12, 13, 14],
                }
            target_data = self.datadict["action"]

        split_indices_train = [
            i
            for i, e in enumerate(target_data)
            if e in _data_split_int_["train"]
        ]
        split_indices_test = [
            i
            for i, e in enumerate(target_data)
            if e in _data_split_int_["test"]
        ]

        return {"train": split_indices_train, "test": split_indices_test}

    def __normalize_poses(self, poses):
        poses_out = np.divide((poses - self.data_mean), self.data_std)
        poses_out = poses_out[:, self.dim_to_use]
        return poses_out

    def _get_stickman_from_3d(self, ids):
        stickmans = []
        for i in ids:
            ang = self.datadict[self.keypoint_key][i]
            intr = self.datadict["intrinsics_univ"][i]
            extr = self.datadict["extrinsics_univ"][i]
            imsize = self.datadict["image_size"][i]

            ang = revert_output_format(
                np.expand_dims(ang, axis=0),
                self.data_mean,
                self.data_std,
                self.dim_to_ignore,
            )
            if self.keypoint_key != "keypoints_3d_world":
                kps3d_w = convert_to_3d(ang, self.kinematic_tree, swap_yz=False)
            else:
                kps3d_w = ang.reshape(
                    ang.shape[0], len(self.joint_model.kps_to_use), 3
                )

            kps3d_c = apply_affine_transform(kps3d_w.squeeze(axis=0), extr)
            kps2d = camera_projection(kps3d_c, intr)

            scale_x = float(self.spatial_size) / imsize[0]
            scale_y = float(self.spatial_size) / imsize[1]

            kps_rescaled = np.multiply(
                kps2d[:, :2], np.asarray([scale_x, scale_y], dtype=np.float)
            )

            stickman = make_joint_img(
                [self.spatial_size, self.spatial_size],
                kps_rescaled,
                self.joint_model,
                line_colors=self.line_colors,
                scale_factor=self.stickman_scale,
            )
            stickmans.append(self.stickman_transforms(stickman))

        return torch.stack(stickmans, dim=0).squeeze()

    def _get_keypoints(self, ids, use_map_ids=False):
        kpts = []
        if use_map_ids:
            ids = self._sample_valid_seq_ids(
                [self.datadict["map_ids"][ids[0]], len(ids) - 1]
            )
        if self.keypoint_key is None:
            key = "norm_keypoints"
        else:
            key = self.keypoint_key
        for id in ids:
            if self.keypoint_key == "keypoints_3d_world":
                kps = self.datadict[key][id]
            else:
                kps = self.datadict[key][id]
            # if key == "angles_3d":
            #     # convert to expmap format
            #     kps = self.__euler2expmap(kps)
            if self.train_reg:
                # keypoints need to be converted to normalized image coordinates
                kps3d_w = revert_output_format(
                np.expand_dims(kps, axis=0),
                self.data_mean,
                self.data_std,
                self.dim_to_ignore,
                    )
                kps3d_w = kps3d_w.reshape(kps3d_w.shape[0], len(self.joint_model.kps_to_use), 3)

                extr = self.datadict["extrinsics_univ"][id]
                intr = self.datadict["intrinsics_univ"][id]
                imsize = self.datadict["image_size"][id]
                # to camera
                kps3d_c = apply_affine_transform(kps3d_w.squeeze(axis=0), extr)
                # to image
                kps2d = camera_projection(kps3d_c, intr)
                # normalize
                kps = np.divide(kps2d[:, :2], imsize)

            kpts.append(kps)

        kpts = np.stack(kpts, axis=0).squeeze()
        # if self.keypoint_key == "keypoints_3d_world":
        #     kpts = kpts.reshape(kpts.shape[0],-1)
        return kpts

    def _get_intrinsic_params(self, ids, use_map_ids=False):
        if use_map_ids:
            ids = self._sample_valid_seq_ids(
                [self.datadict["map_ids"][ids[0]], len(ids) - 1]
            )

        cam_params = self.datadict["intrinsics_univ"][ids].squeeze()
        return cam_params

    def _get_extrinsic_params(self, ids, use_map_ids=False):
        if use_map_ids:
            ids = self._sample_valid_seq_ids(
                [self.datadict["map_ids"][ids[0]], len(ids) - 1]
            )
        extrs = self.datadict["extrinsics_univ"][ids]
        return extrs


def _euler2expmap(angles, kin_tree):
    """
    Transforms a angle array from euler to expmap representation
    :param angles: shape (n_samples,78)
    :param kin_tree:
    :return:
    """
    black_list = {"sample": [], "joint_id": [], "input_angle": []}
    posInd = kin_tree["posInd"]["ids"]
    rotInd = kin_tree["rotInd"]
    expmap_ind = kin_tree["expmapInd"]
    # order = self.kinematic_tree["order"]

    expmaps = np.zeros((angles.shape[0], 99), dtype=np.float)
    for n, ang in enumerate(tqdm(angles)):

        expmaps[n, posInd] = ang[posInd]
        for i, pack in enumerate(zip(rotInd, expmap_ind)):
            ri, ei = pack
            # if ri = []
            if not ri:
                ea = np.asarray([0.0, 0.0, 0.0])
            else:
                ea = ang[ri]
            R = euler_to_rotation_matrix(ea, deg=True)
            try:
                expmaps[n, ei] = rotmat2expmap(R)
            except ValueError as e:
                print("Error : Quaternion not unit quaternion!")
                black_list["sample"].append(n)
                black_list["joint_id"].append(i)
                black_list["input_angle"].append(ea)

    return expmaps  # black_list


def eval_black_list(dataset, save_path):
    from data.data_conversions_3d import fkl, camera_projection
    import pandas as pd
    from lib.utils import add_joints_to_img

    parent = dataset.kinematic_tree["parent"]
    offset = dataset.kinematic_tree["offset"]
    rotInd = dataset.kinematic_tree["rotInd"]
    expmapInd = dataset.kinematic_tree["expmapInd"]
    posInd = dataset.kinematic_tree["posInd"]["ids"]

    eulers = dataset.datadict["angle_euler"]
    df = pd.read_csv("black_list_expmap.csv")

    corrupted_sample_ids = np.asarray(df["sample"])

    corrupted_poses = eulers[corrupted_sample_ids]
    healthy_ones = eulers[corrupted_sample_ids + 1]
    img_paths = dataset.datadict["img_paths"][corrupted_sample_ids]
    img_paths_valid = dataset.datadict["img_paths"][corrupted_sample_ids + 1]
    camera_parameters = dataset.datadict["intrinsics_univ"]
    for i, tup in enumerate(
        tqdm(
            zip(
                img_paths,
                corrupted_poses,
                camera_parameters,
                healthy_ones,
                img_paths_valid,
            )
        )
    ):
        # cam params are the same for valid and invalid samples
        img_p, pose, cam_params, pose_valid, img_valid_p = tup
        as_exp = _euler2expmap(
            np.expand_dims(pose, axis=0), dataset.kinematic_tree
        )
        as_exp_valid = _euler2expmap(
            np.expand_dims(pose_valid, axis=0), dataset.kinematic_tree
        )

        as_kps = fkl(
            np.squeeze(as_exp, axis=0),
            parent,
            offset,
            rotInd,
            expmapInd,
            posInd,
        )
        as_kps = as_kps.reshape((32, 3))
        as_kps_valid = fkl(
            np.squeeze(as_exp_valid, axis=0),
            parent,
            offset,
            rotInd,
            expmapInd,
            posInd,
        )
        as_kps_valid = as_kps_valid.reshape((32, 3))

        projected = camera_projection(as_kps, cam_params)
        projected_valid = camera_projection(as_kps_valid, cam_params)

        img = cv2.imread(img_p)
        img_valid = cv2.imread(img_valid_p)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = add_joints_to_img(
            img,
            projected[dataset.joint_model.kps_to_use],
            dataset.joint_model.total_relative_joints,
        )
        img_valid = add_joints_to_img(
            img_valid,
            projected_valid[dataset.joint_model.kps_to_use],
            dataset.joint_model.total_relative_joints,
            color_kps=[[0, 255, 0]],
            color_joints=[[0, 255, 0]],
        )

        cv2.imwrite(
            path.join(save_path, f"corrupted_sample_nr{i + 1}.jpg"), img
        )
        cv2.imwrite(
            path.join(save_path, f"valid_sample_nr{i + 1}.jpg"), img_valid
        )


def get_intrinsic_mat(params):
    """
    Linear projection matrix withut distortion parameters
    :param params:
    :return:
    """
    return np.asarray(
        [
            [params[0], 0.0, params[1]],
            [0.0, params[2], params[3]],
            [0.0, 0.0, 1.0],
        ]
    )


def estimate_extrinsics(dataset):
    """
    Estimate Extrinsic parameters from world to cam point correspondences
    :param dataset:
    :return:
    """
    # extrinsics are matrices M of shape (3,4) for every datapoint --> M = [R,t] where R=rotation matrix and t = translation vector
    camera_extrinsics_univ = np.zeros(
        (dataset.datadict["keypoints_3d_univ"].shape[0], 3, 4), dtype=np.float
    )
    camera_extrinsics = np.zeros(
        (dataset.datadict["keypoints_3d"].shape[0], 3, 4), dtype=np.float
    )

    for i, vid in enumerate(
        tqdm(
            np.unique(dataset.datadict["v_ids"]),
            desc="Estimate extrinsics per video",
        )
    ):
        ids = dataset.datadict["v_ids"] == vid
        kps3d_c = dataset.datadict["keypoints_3d"][ids]
        kps3d_c_univ = dataset.datadict["keypoints_3d_univ"][ids]
        kps3d_w = dataset.datadict["keypoints_3d_world"][ids]
        kps3d_c = np.reshape(kps3d_c, (-1, 3))
        kps3d_c_univ = np.reshape(kps3d_c_univ, (-1, 3))
        kps3d_w = np.reshape(kps3d_w, (-1, 3))

        _, M, _ = cv2.estimateAffine3D(
            kps3d_w, kps3d_c, ransacThreshold=10, confidence=0.999
        )
        _, M_univ, _ = cv2.estimateAffine3D(
            kps3d_w, kps3d_c_univ, ransacThreshold=10, confidence=0.999
        )

        # returned values correspond to [R,t]^T
        camera_extrinsics[ids] = M
        camera_extrinsics_univ[ids] = M_univ

    return camera_extrinsics_univ, camera_extrinsics

if __name__ == "__main__":
    import torchvision.transforms as tt
    from os import path, makedirs
    from PIL import Image
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from lib.logging import create_video_3d
    from skvideo import io as vio
    from data.data_conversions_3d import (
        fkl,
        camera_projection,
        apply_affine_transform,
        revert_output_format,
        convert_to_3d,
    )

    from lib.utils import parallel_data_prefetch, add_joints_to_img
    import yaml

    save_path = "./test_data/human36m_full"
    makedirs(save_path, exist_ok=True)

    with open("../config/test_datasets.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if config["general"]["mode"] == "visualize_projection":
        transforms = tt.Compose([tt.ToTensor()])

        print("preparing dataset...")

        dataset = Human36mDataset(
            transforms,
            data_keys=["keypoints"],  # "kp_change", "keypoints"
            mode="test",
            **config["data"]
        )

        # complete data as euler angles
        angles_euler = dataset.datadict["angle_euler"]
        parent = dataset.kinematic_tree["parent"]
        offset = dataset.kinematic_tree["offset"]
        rotInd = dataset.kinematic_tree["rotInd"]
        expmapInd = dataset.kinematic_tree["expmapInd"]
        posInd = dataset.kinematic_tree["posInd"]["ids"]

        # visualize as a test
        time_points = np.random.choice(
            np.arange(0, len(dataset) - 50), 5, replace=False
        )
        for nr, i in enumerate(tqdm(time_points, leave=False)):
            #
            frame_ids = np.arange(i, i + 50)
            vid = []

            intrinsics = dataset.datadict["intrinsics_univ"][frame_ids]
            expmaps = dataset.datadict["angle_world_expmap"][frame_ids]
            extrinsics = dataset.datadict["extrinsics_univ"][frame_ids]
            img_paths = dataset.datadict["img_paths"][frame_ids]
            for img_path, pose, intrs, extrs in zip(
                    img_paths, expmaps, intrinsics, extrinsics
            ):
                pose = revert_output_format(
                    np.expand_dims(pose, axis=0),
                    dataset.data_mean,
                    dataset.data_std,
                    dataset.dim_to_ignore,
                )
                keypoints_world = fkl(
                    np.squeeze(pose, axis=0),
                    parent,
                    offset,
                    rotInd,
                    expmapInd,
                    posInd,
                )

                keypoints_world = keypoints_world.reshape((-1, 3))

                keypoints_camera = apply_affine_transform(
                    keypoints_world, extrs
                )
                keypoints_2d = camera_projection(keypoints_camera, intrs)

                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                combined_img = add_joints_to_img(
                    img,
                    keypoints_2d[dataset.joint_model.kps_to_use],
                    dataset.joint_model.total_relative_joints,
                )
                vid.append(combined_img)

            restored = revert_output_format(
                expmaps,
                dataset.data_mean,
                dataset.data_std,
                dataset.dim_to_ignore,
            )

            restored = convert_to_3d(restored, dataset.kinematic_tree, False)

            vid_3D = create_video_3d(
                restored,
                [0, 0, 1],
                dataset,
                "test_3d",
                use_limits=True,
                use_posInd=True,
            )

            writer = vio.FFmpegWriter(
                path.join(save_path, f"kps_3d_world_unproc#{nr}.mp4")
            )

            for frame in vid_3D:
                writer.writeFrame(frame)

            writer.close()

            vid = np.stack(vid, axis=0)

            writer = vio.FFmpegWriter(
                path.join(save_path, f"test_video_kps_img_expmap#{nr}.mp4")
            )

            for frame in vid:
                writer.writeFrame(frame)

            writer.close()

    elif config["general"]["mode"] == "test_synth":
        font_size = 0.7
        font_thickness = 2
        transforms = tt.Compose(
            [
                tt.ToPILImage(),
                tt.Resize(
                    [config["data"]["spatial_size"],
                     config["data"]["spatial_size"]], Image.BICUBIC
                ),
                tt.ToTensor(),
            ]
        )

        dataset = Human36mDataset(
            transforms,
            data_keys=["pose_img", "app_img", "stickman", "sample_ids"],
            mode="test",
            **config["data"]
        )

        print(f"number of data samples = {len(dataset)}")

        loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

        for i, batch in enumerate(
                tqdm(
                    loader,
                    total=config["general"]["n_vid_to_generate"],
                    desc=f"Generating keypoint images for vunet training",
                )
        ):
            if i >= config["general"]["n_vid_to_generate"]:
                break
            pimg = (
                (batch["pose_img"].squeeze(dim=0) * 255.0)
                    .permute(1, 2, 0)
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
            )
            aimg = (
                (batch["app_img"].squeeze(dim=0) * 255.0)
                    .permute(1, 2, 0)
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
            )
            stickman = (
                (batch["stickman"].squeeze(dim=0) * 255.0)
                    .permute(1, 2, 0)
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
            )
            overlay = cv2.addWeighted(pimg, 0.5, stickman, 0.5, 0)

            img = np.concatenate([aimg, stickman, pimg, overlay], axis=1)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(path.join(save_path, f"vunet_train_image#{i}.jpg"), img)

    else:

        transforms = tt.Compose(
            [
                tt.ToPILImage(),
                tt.Resize(
                    [config["data"]["spatial_size"],
                     config["data"]["spatial_size"]], Image.BICUBIC
                ),
                tt.ToTensor(),
                # tt.Lambda(lambda x: (x.float() / 127.5) - 1.0),
            ]
        )

        # data input
        print("preparing dataset...")
        # datapath = "/export/scratch/tmilbich/Datasets/human3.6M/processed/min_kp_move_20_plusEval/"
        dataset = Human36mDataset(
            transforms,
            data_keys=["pose_img", "stickman"],  # "kp_change", "keypoints"
            mode="test",
            label_transfer=True,
            debug=True,
            **config["data"]
        )

        print(f"Dataset contains {len(dataset)} samples.")

        unique_aids, counts = np.unique(
            dataset.datadict["action"], return_counts=True
        )
        weights = np.ones_like(dataset.datadict["action"], dtype=np.float)

        for n, aid in enumerate(unique_aids):
            weights[dataset.datadict["action"] == aid] = 1.0 / counts[n]

        # sample with weights to obtain balanced dataset
        # sampler = WeightedRandomSampler(
        #     weights=weights, num_samples=len(dataset), replacement=True
        # )
        # seq_sampler = SequenceSampler(
        #     dataset, sampler=sampler, batch_size=5, drop_last=True
        # )
        loader = DataLoader(dataset, num_workers=0, batch_size=5, shuffle=True)

        nex = 20
        for i, batch in enumerate(tqdm(loader, total=nex)):

            if i == nex:
                break

            stickman = (
                    batch["stickman"].permute(0, 2, 3, 1).cpu().numpy() * 255.0
            ).astype(np.uint8)
            img = (
                    batch["pose_img"].permute(0, 2, 3, 1).cpu().numpy() * 255.0
            ).astype(np.uint8)

            img = np.concatenate([p for p in img], axis=1)
            stickman = np.concatenate([p for p in stickman], axis=1)
            overlay = cv2.addWeighted(img, 0.5, stickman, 0.5, 0)
            comb = np.concatenate([stickman, img, overlay], axis=0)

            comb = cv2.cvtColor(comb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(path.join(save_path, f"cvbae_train_imgs#{i}.jpg"), comb)

        # visualize_matched_poses_3d(
        #     loader,
        #     logwandb=False,
        #     save_path=save_path,
        #     revert_coord_space=False,
        # )
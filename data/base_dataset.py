import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
from lib.utils import (
    get_bounding_box,
    make_joint_img,
    JointModel,
    get_line_colors,
)
from abc import abstractmethod
import imagesize
from copy import deepcopy
from torchvision.transforms import Resize, Compose
from functools import partial
from numpy import linalg as LA
from collections.abc import Iterable
import warnings


class BaseDataset(Dataset):
    def __init__(
        self,
        transforms,
        mode: str,
        seq_length,
        data_keys: list,
        joint_model: JointModel,
        **kwargs,
    ):
        super().__init__()

        if 'datapath' not in kwargs or 'datapath' == 'none':
            raise ValueError(f'When intending to load the data, the respective datapath must be set in config-file for {self.__class__.__name__}.')

        self.datapath = kwargs['datapath']

        self.complete_datadict = None
        # default spatial size is 256 (only used for image/video generation
        self.spatial_size = (
            kwargs["spatial_size"] if "spatial_size" in kwargs else 256
        )
        self.use_person_split = (
            kwargs["use_person_split"] if "use_person_split" in kwargs else True
        )
        # use crops if required; Note: this is currently only used in synthesis module
        # for computation of image quality metrics
        self.use_crops = (
            kwargs["use_crops"] if "use_crops" in kwargs.keys() else False
        )
        self.datakeys = data_keys
        self.use_crops_for_app = (
            kwargs["crop_app"] if "crop_app" in kwargs.keys() else False
        )

        self.prepare_seq_matching = (
            kwargs["prepare_seq_matching"]
            if "prepare_seq_matching" in kwargs.keys()
            else False
        )

        self.train_reg = (
            kwargs["train_regressor"] if "train_regressor" in kwargs else False
        )
        self.reg_steps = (
            kwargs["reg_steps"]
            if self.train_reg and "reg_steps" in kwargs
            else -1
        )

        if self.train_reg and self.reg_steps > 1:
            addkeys = ["reg_imgs", "reg_targets"]
            self.datakeys.extend(addkeys)

        self.train_synthesis = (
            kwargs["train_synthesis"] if "train_synthesis" in kwargs else False
        )

        # overall data split
        self.overall_split = (
            kwargs["overall_split"] if "overall_split" in kwargs else False
        )

        # sequence length related stuff
        self.sequential_frame_lag = (
            kwargs["sequential_frame_lag"]
            if "sequential_frame_lag" in kwargs.keys()
            else 1
        )
        assert self.sequential_frame_lag >= 1

        self.seq_length = (
            seq_length
            if isinstance(seq_length, tuple) or isinstance(seq_length, list)
            else (seq_length, seq_length)
        )

        if "label_transfer" in kwargs.keys() and kwargs["label_transfer"]:
            self.label_transfer = True
        else:
            self.label_transfer = False

        if "inplane_normalize" in kwargs.keys():
            self.inplane_norm = kwargs["inplane_normalize"]
        else:
            self.inplane_norm = False

        if "box_factor" in kwargs.keys():
            self.box_factor = kwargs["box_factor"]
        else:
            self.box_factor = -1

        if self.inplane_norm and self.box_factor < 0:
            raise ValueError(
                f"The box factor must be larger than zero if inplane normalization should be applied to the appearance images but is actually {self.box_factor}"
            )

        # other
        self.joint_model = joint_model
        self._output_dict = {
            "pose_img": partial(self._get_pose_img, use_crops=self.use_crops),
            "keypoints": partial(self._get_keypoints, use_map_ids=False),
            "stickman": self._get_stickman,
            "app_img": partial(
                self._get_app_img, inplane_norm=self.inplane_norm
            ),
            "sample_ids": self._get_sample_ids,
            "synth_weights": self._get_synth_weights,
            "cropped_pose_img": partial(self._get_pose_img, use_crops=True),
            "img_size": self._get_img_size,
            "action": self._get_action,
            "matched_keypoints": self._get_matched_keypoints,
            "paired_keypoints": partial(self._get_keypoints, use_map_ids=True),
            "paired_sample_ids": partial(
                self._get_sample_ids, use_map_ids=True
            ),
        }

        if "pose_img" in self.datakeys:
            self._output_dict.update(
                {"pose_img_inplane": self._get_pose_img_inplane}
            )
            self.datakeys.append("pose_img_inplane")
        try:
            self.root_joint_idx = self.joint_model.kp_to_joint.index("head")
        except ValueError:
            self.root_joint_idx = None
        # set exp weight for sampling kp to change
        self.line_colors = (
            get_line_colors(
                (
                    len(self.joint_model.left_lines),
                    len(self.joint_model.right_lines),
                    len(self.joint_model.head_lines)
                    + len(self.joint_model.face),
                )
            )
            if "diff_line_colors" in kwargs.keys()
            and kwargs["diff_line_colors"]
            else None
        )

        self.stickman_scale = (
            kwargs["stickman_scale"]
            if "stickman_scale" in kwargs.keys()
            and kwargs["stickman_scale"] > 0
            else None
        )

        # check validity of data keys
        if not all(map(lambda x: x in self._output_dict.keys(), data_keys)):
            warnings.warn(
                f"Not all data_keys are in {list(self._output_dict.keys())}. Check if this is intended."
            )

        if "cropped_pose_img" in data_keys and self.use_crops:
            warnings.warn(
                f"use_crops is enabled and cropped_pose_img is in data_keys! Loading of the cropped pose image is hence duplicated. remove one option for faster training."
            )

        self.transforms = transforms
        # remove resizing transformation for stickman, if such one is contained in transforms
        self.stickman_transforms = Compose(
            list(
                filter(
                    lambda x: not isinstance(x, Resize),
                    self.transforms.transforms,
                )
            )
        )
        self.mode = mode

        if "synth_weights" in self.datakeys and ("synth_weights" not in kwargs):
            raise ValueError(
                f'If synth_weights shall be used, a "synth_weights"-parameter has to define the weight value which shall be used within the bounding box'
            )

        self.synth_weights = (
            kwargs["synth_weights"] if "synth_weights" in kwargs else None
        )

        self.datadict = {
            "img_paths": np.asarray([], dtype=np.object),
            "keypoints": np.asarray([], dtype=np.float),
            "v_ids": np.asarray([], dtype=np.int),
            # vid is the id of the video
            "p_ids": np.asarray([], dtype=np.int),
            # pid is the id of the person, shall be unique
            "f_ids": np.asarray([], dtype=np.int),
            # frame id is the number of the frame within the related sequence
            "map_ids": np.asarray([], dtype=np.int),
            "seq_lengths": np.asarray([], dtype=np.int),
            # this is set if keypoints are intended to be normalized
            "norm_keypoints": np.asarray([], dtype=np.float),
            "action": np.asarray([], dtype=np.int),
        }
        # if self.motion_based_sampling:
        #     self.datadict.update(
        #         {"motion_scores": np.asarray([], dtype=np.float)}
        #     )
        self.person_ids = []

        # data container for data prefetching
        self.prefetched_datadict = {"sample_ids": None, "kp_change": None}

        self.sequence_end_ids = None
        self.sequence_start_ids = None
        self.action_id_to_action = None
        self.image_shapes = None
        self.seqs_per_action = None
        self.matched_map_ids = None
        self.use_matched_map_ids = (
            self.label_transfer and not self.prepare_seq_matching
        )

    def __getitem__(self, idx):

        # get sequence indices
        if self.prefetched_datadict["sample_ids"] is not None:
            # use prefetched ids if available
            ids = self.prefetched_datadict["sample_ids"][idx]
        else:
            ids = self._sample_valid_seq_ids(idx)

        # collect outputs
        data = {
            key: self._output_dict[key](ids)
            for key in self.datakeys
            if key not in ["reg_imgs", "reg_targets"]
        }
        if self.train_reg:
            data = self._add_reg_imgs(ids, data)

        return data

    def _add_reg_imgs(self, ids, data):
        reg_ids = (
            ids
            + list(
                np.random.choice(
                    self.__len__(), self.reg_steps - 1, replace=False
                )
            )
            if self.reg_steps > 1
            else ids
        )
        reg_img_fn = (
            self._get_pose_img_inplane
            if self.inplane_norm
            else partial(self._get_pose_img, use_crops=self.use_crops)
        )
        data.update(
            {
                "reg_imgs": reg_img_fn(reg_ids),
                "reg_targets": self._get_keypoints(reg_ids, use_map_ids=False),
            }
        )

        return data

    def _get_pose_img(self, ids, use_crops, use_complete_ddict=False):
        if use_complete_ddict:
            assert not use_crops
            img_paths = self.complete_datadict["img_paths"][ids]
        else:
            img_paths = self.datadict["img_paths"][ids]
        prep_img = []
        if use_crops:
            keypoints = self.datadict["keypoints"][ids]
            for (p, kps) in zip(img_paths, keypoints):
                pimg = cv2.imread(p)
                pimg = cv2.cvtColor(pimg, cv2.COLOR_BGR2RGB)

                crop_dict = get_bounding_box(kps, pimg.shape)
                cr_box = crop_dict["bbox"]

                if np.any(crop_dict["pads"] > 0):
                    pimg = cv2.copyMakeBorder(
                        pimg,
                        crop_dict["pads"][0],
                        crop_dict["pads"][1],
                        crop_dict["pads"][2],
                        crop_dict["pads"][3],
                        borderType=cv2.BORDER_REFLECT,
                    )
                pimg = pimg[cr_box[2] : cr_box[3], cr_box[0] : cr_box[1]]

                prep_img.append(self.transforms(pimg))
        else:
            for p in img_paths:
                pimg = cv2.imread(p)
                pimg = cv2.cvtColor(pimg, cv2.COLOR_BGR2RGB)
                prep_img.append(self.transforms(pimg))

        return torch.stack(prep_img, dim=0).squeeze()

    def _get_img_size(self, ids):
        img_paths = self.datadict["img_paths"][ids]
        sizes = []
        for p in img_paths:
            # image shape
            sizes.append(torch.tensor(imagesize.get(p), dtype=torch.int))

        return torch.stack(sizes, dim=0).squeeze(dim=0)

    def _get_keypoints(self, ids, use_map_ids=False):
        kpts = []
        if use_map_ids:
            ids = self._sample_valid_seq_ids(
                [self.datadict["map_ids"][ids[0]], len(ids) - 1]
            )
        if self.use_crops:
            for id in ids:
                kps = self.datadict["keypoints"][
                    id, self.joint_model.kps_to_use, :2
                ]
                imsize = imagesize.get(self.datadict["img_paths"][id])
                crop_dict = get_bounding_box(kps, [imsize[1], imsize[0]])
                bbx = crop_dict["bbox"]

                kps_rescaled = kps - np.asarray(
                    [bbx[0], bbx[2]], dtype=np.float
                )
                scale_x = 1.0 / np.abs(bbx[1] - bbx[0])
                scale_y = 1.0 / np.abs(bbx[3] - bbx[2])

                kps_rescaled = np.multiply(
                    kps_rescaled[:, :2],
                    np.asarray([scale_x, scale_y], dtype=np.float),
                )

                kpts.append(kps_rescaled)
        else:
            key = "norm_keypoints"
            for id in ids:
                kps = self.datadict[key][id, self.joint_model.kps_to_use, :2]
                kps = np.clip(kps, a_min=0.0, a_max=1.0)

                kpts.append(kps)

        kpts = np.stack(kpts, axis=0).squeeze()

        return kpts

    def _get_matched_keypoints(self, ids):

        if hasattr(self, "pose_encodings"):
            # get random action sequence
            action_id = self.datadict["action"][ids[0]]
            ids_target = self.get_action_sequence(int(action_id))
            embeds_base = self.pose_encodings[ids, :]
            embeds_target = self.pose_encodings[ids_target, :]

            # find nn sequence
            dists = []
            ids_start = []
            seq_len1 = embeds_base.shape[0]
            seq_len2 = embeds_target.shape[0]
            offset = 5
            for k in range(
                0, seq_len2 - (seq_len1 * self.sequential_frame_lag) + 1, offset
            ):
                ids_tmp = np.arange(
                    k,
                    k + (self.sequential_frame_lag * seq_len1),
                    self.sequential_frame_lag,
                )

                dists.append(
                    np.mean(LA.norm(embeds_target[ids_tmp, :] - embeds_base))
                )
                ids_start.append(k)

            # find nn subseq
            dists = np.stack(dists)
            id_nn = np.argmin(dists)

            # id2use = ids_start[id_nn]
            # seq_nn = ids_target[id2use : id2use + seq_len1]

            ids2use = np.arange(
                ids_start[id_nn],
                ids_start[id_nn] + (self.sequential_frame_lag * seq_len1),
                self.sequential_frame_lag,
            )
            seq_nn = np.asarray(ids_target)[ids2use]
        else:
            seq_nn = self._sample_valid_seq_ids(
                [self.matched_map_ids[ids[0]], len(ids) - 1]
            )

        kps = self._get_keypoints(seq_nn)

        return kps, seq_nn

    def _get_app_img(self, ids, inplane_norm, use_complete_ddict=False):
        # this part always uses crops
        if use_complete_ddict and self.complete_datadict is not None:
            ddict = self.complete_datadict
        else:
            ddict = self.datadict
        if not isinstance(ids, Iterable):
            ids = [ids]
        app_paths = ddict["img_paths"][ddict["map_ids"][ids]]
        if not isinstance(app_paths, np.ndarray):
            app_paths = [app_paths]
        prep_imgs = []
        if inplane_norm:
            kpts = ddict["keypoints"][ddict["map_ids"][ids]]
            for p, kps in zip(app_paths, kpts):
                orig_img = cv2.imread(p)
                orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
                # original height
                # oh = orig_img.shape[0]
                oh = self.spatial_size
                # target heights and widths
                # hw = [s // 2** self.box_factor for s in orig_img.shape[:2]]
                # wh = list(reversed(hw))
                hw = (
                    self.spatial_size // 2 ** self.box_factor,
                    self.spatial_size // 2 ** self.box_factor,
                )
                wh = hw
                part_imgs = []
                for t in self.joint_model.norm_T:
                    part_img = np.zeros([hw[0], hw[1], 3], dtype=np.uint8)
                    # get transformation
                    T = t(kps, jm=self.joint_model, wh=wh, oh=oh)
                    if T is not None:
                        part_img = cv2.warpPerspective(
                            orig_img, T, hw, borderMode=cv2.BORDER_REPLICATE
                        )
                    else:
                        part_img = np.zeros((hw[0], hw[1], 3), dtype=np.uint8)
                    part_imgs.append(self.stickman_transforms(part_img))

                # since part_imgs are already torch.tensors, concatenate in first axis
                pimg = torch.cat(part_imgs, dim=0)
                prep_imgs.append(pimg)

        else:
            # use image cropped around the keypoints of the specific person
            if self.use_crops_for_app:
                kpts = ddict["keypoints"][ddict["map_ids"][ids]]
                for p, kps in zip(app_paths, kpts):
                    pimg = cv2.imread(p)
                    pimg = cv2.cvtColor(pimg, cv2.COLOR_BGR2RGB)

                    crop_dict = get_bounding_box(kps, pimg.shape)
                    cr_box = crop_dict["bbox"]

                    if np.any(crop_dict["pads"] > 0):
                        pimg = cv2.copyMakeBorder(
                            pimg,
                            crop_dict["pads"][0],
                            crop_dict["pads"][1],
                            crop_dict["pads"][2],
                            crop_dict["pads"][3],
                            borderType=cv2.BORDER_REFLECT,
                        )
                    pimg = pimg[cr_box[2] : cr_box[3], cr_box[0] : cr_box[1]]
                    prep_imgs.append(self.transforms(pimg))
            else:
                for p in app_paths:
                    pimg = cv2.imread(p)
                    pimg = cv2.cvtColor(pimg, cv2.COLOR_BGR2RGB)
                    prep_imgs.append(self.transforms(pimg))

        return torch.stack(prep_imgs, dim=0).squeeze()

    def _get_stickman(self, ids, sscale=None):
        kpts = self.datadict["keypoints"][ids]
        img_paths = self.datadict["img_paths"][ids]
        if "img_size" in self.datadict and self.datadict["img_size"].size > 0:
            img_shapes = self.datadict["img_size"][ids]
        else:
            img_shapes = None

        stickmans = []

        for i, (kps, p) in enumerate(zip(kpts, img_paths)):
            # attention imagesize returns size as (width,height)
            if img_shapes is not None:
                img_shape = [img_shapes[i][1], img_shapes[i][0]]
            else:
                img_shape = imagesize.get(p)
            if self.use_crops:

                crop_dict = get_bounding_box(kps, [img_shape[1], img_shape[0]])
                bbx = crop_dict["bbox"]
                # new coordinate origin
                kps_rescaled = kps - np.asarray(
                    [bbx[0], bbx[2]], dtype=np.float
                )
                # scale to desired img size
                scale_x = float(self.spatial_size) / np.abs(bbx[1] - bbx[0])
                scale_y = float(self.spatial_size) / np.abs(bbx[3] - bbx[2])

                kps_rescaled = np.multiply(
                    kps_rescaled[:, :2],
                    np.asarray([scale_x, scale_y], dtype=np.float),
                )

            else:
                scale_x = float(self.spatial_size) / img_shape[0]
                scale_y = float(self.spatial_size) / img_shape[1]

                kps_rescaled = np.multiply(
                    kps[:, :2], np.asarray([scale_x, scale_y], dtype=np.float)
                )
            stickman = make_joint_img(
                [self.spatial_size, self.spatial_size],
                kps_rescaled,
                self.joint_model,
                line_colors=self.line_colors,
                scale_factor=self.stickman_scale if sscale is None else sscale,
            )
            if np.all(stickman == 0):
                print("zeroimg")
            stickmans.append(self.stickman_transforms(stickman))

        return torch.stack(stickmans, dim=0).squeeze()

    def _get_sample_ids(self, ids, use_map_ids=False):
        if use_map_ids:
            ids = self._sample_valid_seq_ids(
                [self.datadict["map_ids"][ids[0]], len(ids) - 1]
            )
            return np.asarray(ids)
        else:
            return np.asarray(ids)

    def _get_action(self, ids):
        return self.datadict["action"][ids]

    def _sample_valid_seq_ids(self, input_data):
        if all(map(lambda x: x == 0, self.seq_length)) and isinstance(
            input_data, int
        ):
            return [input_data]
        elif all(map(lambda x: x == 0, self.seq_length)) and isinstance(
            input_data, list
        ):
            return [input_data[0]]

        if isinstance(input_data, int):
            idx = input_data
            seq_len = np.random.choice(
                range(self.seq_length[0], self.seq_length[1] + 1), 1
            )[0]

        elif isinstance(input_data, list) and len(input_data) > 1:
            idx = input_data[0]
            seq_len = input_data[-1]
        else:
            raise ValueError("Unsupported input datatype.")

        # get inital valid lags
        seq_end_id = self.sequence_end_ids[self.datadict["v_ids"][idx]]
        frame_lag = self.sequential_frame_lag
        # use predefined seq len
        idx_start = idx
        idx_end = idx_start + frame_lag * seq_len + 1  # seq = anchor + seq len!

        # assert staying inside sequence
        if idx_end > seq_end_id:
            seq_start_id = self.sequence_start_ids[self.datadict["v_ids"][idx]]
            idx_start = idx_start - (idx_end - seq_end_id) + 1
            idx_end = seq_end_id + 1  # + 1 as range ends end-1
            if idx_start < seq_start_id:
                frame_lag = max(1, int((idx_end - seq_start_id) / seq_len))
                idx_start = idx_end - frame_lag * seq_len - 1

        return np.arange(
            start=idx_start, stop=idx_end, step=frame_lag
        )  # assert sequences of equal sequence lengths !

    def _get_sequence_end_ids(self):

        self.sequence_end_ids = dict()
        for k in list(np.unique(self.datadict["v_ids"])):
            self.sequence_end_ids[k] = np.max(
                np.where(self.datadict["v_ids"] == k)[0]
            )

    def _get_sequence_start_ids(self):
        self.sequence_start_ids = dict()
        for k in list(np.unique(self.datadict["v_ids"])):
            self.sequence_start_ids[k] = np.min(
                np.where(self.datadict["v_ids"] == k)[0]
            )

    def _check_seq_len_and_frame_lag(self):
        assert (
            self.sequence_end_ids is not None
            and self.sequence_start_ids is not None
            and self.sequential_frame_lag >= 1
        )
        assert len(self.sequence_start_ids) == len(self.sequence_end_ids)
        assert set(self.sequence_end_ids.keys()).intersection(
            self.sequence_start_ids.keys()
        ) == set(self.sequence_end_ids.keys())

        sequence_lengths = [
            self.sequence_end_ids[vid] - self.sequence_start_ids[vid]
            for vid in self.sequence_end_ids
        ]
        min_seq_len = np.min(sequence_lengths)

        if self.seq_length[1] * self.sequential_frame_lag > min_seq_len:
            print(
                "WARNING: Frame lag and sequence lengths are too long for the dataset. Trying to reduce frame lag"
            )
            print(f"Frame lag before reduction: {self.sequential_frame_lag}")
            self.sequential_frame_lag = max(
                1, int(min_seq_len / self.seq_length[1])
            )
            print(f"Frame lag after reduction: {self.sequential_frame_lag}")

            if self.seq_length[1] > min_seq_len:
                print(
                    f"WARNING: Sequence length too long even the minimum frame lag, reducing maximum sequence length to {min_seq_len}."
                )
                self.seq_length = (self.seq_length[0], min_seq_len)
                if self.seq_length[0] >= self.seq_length[1]:
                    self.seq_length = (
                        self.seq_length[1] - 1,
                        self.seq_length[1],
                    )

            print(
                f"Sequence length and frame lag after adaptation: Sequence length = {self.seq_length}; Frame lag = {self.sequential_frame_lag}"
            )

    def _get_synth_weights(self, ids):
        kpts = self.datadict["keypoints"][ids]
        img_paths = self.datadict["img_paths"][ids]
        weight_maps = []
        for kps, p in zip(kpts, img_paths):
            img_shape = imagesize.get(p)
            scale_x = float(self.spatial_size) / img_shape[0]
            scale_y = float(self.spatial_size) / img_shape[1]

            kps_rescaled = np.multiply(
                kps[:, :2], np.asarray([scale_x, scale_y]), dtype=np.float32
            )

            bb = get_bounding_box(kps_rescaled, img_shape)["bbox"]
            weight_map = np.ones(
                shape=[self.spatial_size, self.spatial_size, 1], dtype=np.float
            )
            weight_map[
                max(0, bb[2]) : min(self.spatial_size, bb[3]),
                max(0, bb[0]) : min(self.spatial_size, bb[1]),
            ] = self.synth_weights
            weight_map = cv2.GaussianBlur(weight_map, (9, 9), sigmaX=3.0)
            weight_maps.append(torch.tensor(np.expand_dims(weight_map, axis=0)))
            # channel dimension squeezed after gaussian blur
            # weight_maps.append(
            #     torch.tensor(weight_map)
            # )

        return torch.stack(weight_maps, dim=0).squeeze(dim=0)

    def _get_pose_img_inplane(self, ids):
        # this part always uses crops
        app_paths = self.datadict["img_paths"][ids]
        if not isinstance(app_paths, np.ndarray):
            app_paths = [app_paths]
        prep_imgs = []

        kpts = self.datadict["keypoints"][ids]
        for p, kps in zip(app_paths, kpts):
            orig_img = cv2.imread(p)
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
            # original height
            # oh = orig_img.shape[0]
            oh = self.spatial_size
            # target heights and widths
            # hw = [s // 2** self.box_factor for s in orig_img.shape[:2]]
            # wh = list(reversed(hw))
            hw = (
                self.spatial_size // 2 ** self.box_factor,
                self.spatial_size // 2 ** self.box_factor,
            )
            wh = hw
            part_imgs = []
            for t in self.joint_model.norm_T:
                part_img = np.zeros([hw[0], hw[1], 3], dtype=np.uint8)
                # get transformation
                T = t(kps, jm=self.joint_model, wh=wh, oh=oh)
                if T is not None:
                    part_img = cv2.warpPerspective(
                        orig_img, T, hw, borderMode=cv2.BORDER_REPLICATE
                    )
                else:
                    part_img = np.zeros((hw[0], hw[1], 3), dtype=np.uint8)
                # don't resize
                part_imgs.append(self.stickman_transforms(part_img))

            # since part_imgs are already torch.tensors, concatenate in first axis
            pimg = torch.cat(part_imgs, dim=0)
            prep_imgs.append(pimg)

        return torch.stack(prep_imgs, dim=0).squeeze()

    def resample_map_ids(self):
        print("resampling map ids!")
        self.__resample_map(self.datadict, use_matched=True)
        if self.complete_datadict is not None:
            self.__resample_map(self.complete_datadict)

    def __resample_map(self, ddict, use_matched=False):
        assert ddict["action"].size > 0 and ddict["map_ids"].size > 0
        unique_aids = np.unique(ddict["action"])
        if self.label_transfer:

            for id in unique_aids:
                same_ids = np.nonzero(np.equal(ddict["action"], id))[0]
                diff_ids = np.nonzero(np.not_equal(ddict["action"], id))[0]
                # if diff ids are more than same ids, draw with replacement (this is only expected if dataset is highly imbalanced
                replacement = same_ids.size > diff_ids.size
                map_ids = np.random.choice(
                    diff_ids, same_ids.size, replace=replacement
                )
                ddict["map_ids"][same_ids] = map_ids

                # this is only applied, if label transfer is enabled and
                if self.matched_map_ids is not None and use_matched:
                    valid_ids = np.nonzero(np.equal(ddict["action"], id))[0]
                    same_map_ids = deepcopy(valid_ids)
                    np.random.shuffle(same_map_ids)
                    self.matched_map_ids[valid_ids] = same_map_ids
        else:
            for id in unique_aids:
                valid_ids = np.nonzero(np.equal(ddict["action"], id))[0]
                map_ids = deepcopy(valid_ids)
                np.random.shuffle(map_ids)
                ddict["map_ids"][valid_ids] = map_ids

    def get_action_sequence(self, action_label, return_kps=False):

        id_tmp = int(
            np.random.choice(
                list(range(len(self.seqs_per_action[action_label]))), size=1
            )
        )
        if not return_kps:
            return self.seqs_per_action[action_label][id_tmp]
        else:
            return self._get_keypoints(
                self.seqs_per_action[action_label][id_tmp]
            )

    def _make_overall_split(self):
        # make overall person split
        if isinstance(self.datadict["img_paths"], list):
            if len(self.datadict["img_paths"]) == 0:
                raise Exception(
                    "The datadict has to be filled before calling make_overall_split."
                )
        elif isinstance(self.datadict["img_paths"], np.ndarray):
            if self.datadict["img_paths"].size == 0:
                raise Exception(
                    "The datadict has to be filled before calling make_overall_split."
                )
        else:
            raise ValueError(
                'The fields of the datadict have to be either of type "list" or "np.ndarray".'
            )

        self.datadict = {
            key: np.asarray(self.datadict[key]) for key in self.datadict
        }

        n_train = int(0.8 * self.datadict["img_paths"].shape[0])
        ids = np.arange(self.datadict["img_paths"].shape[0])
        np.random.seed(42)
        np.random.shuffle(ids)
        target_ids = ids[:n_train]

        self.datadict = {
            key: self.datadict[key][target_ids]
            for key in self.datadict
            if self.datadict[key].size != 0
        }

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def get_test_app_images(self) -> dict:
        pass

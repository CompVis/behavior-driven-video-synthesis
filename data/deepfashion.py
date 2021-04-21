import numpy as np
from tqdm.autonotebook import tqdm
from os import path
import torch
import cv2
from functools import partial
import pickle
from copy import deepcopy
from torchvision import transforms as tt
from PIL import Image

from data.base_dataset import BaseDataset
from lib.utils import JointModel, t2p, t3p, t4p, valid_joints


class DeepFashionDataset(BaseDataset):
    def __init__(
        self, transforms, data_keys, seq_length, mode="train", **kwargs
    ):
        assert mode in ["train", "test"]

        # data_keys = ["pose_img","stickman", "app_img", "sample_ids"]
        kwargs["cropp_app"] = False

        joint_model = JointModel(
            body=[8, 2, 5, 11],
            right_lines=[(10, 9), (9, 8), (2, 3), (3, 4)],
            left_lines=[(13, 12), (12, 11), (5, 6), (6, 7)],
            head_lines=[],
            face=[(0, 14), (0, 15), (14, 16), (15, 17)],
            rshoulder=2,
            lshoulder=5,
            headup=0,
            kps_to_use=list(range(18)),
            total_relative_joints=[],
            right_hand=[],
            left_hand=[],
            head_part=[],
            kp_to_joint=[
                "nose",
                "neck",
                "rshoulder",
                "relbow",
                "rwrist",
                "lshoulder",
                "lelbow",
                "lwrist",
                "rhip",
                "rknee",
                "rankle",
                "lhip",
                "lknee",
                "lfoot",
                "reye",
                "leye",
                "rear",
                "lear",
            ],
            kps_to_change=None,
            kps_to_change_rel=None,
            norm_T=[
                t4p,
                t3p,
                partial(t2p, ids=[2, 3]),
                partial(t2p, ids=[3, 4]),
                partial(t2p, ids=[5, 6]),
                partial(t2p, ids=[6, 7]),
                partial(t2p, ids=[8, 9]),
                partial(t2p, ids=[9, 10]),
                partial(t2p, ids=[11, 12]),
                partial(t2p, ids=[12, 13]),
            ],
        )

        self.random_rotation = (
            not kwargs["inplane_normalize"]
            if "inplane_normalize" in kwargs
            else True
        )

        super().__init__(
            transforms,
            mode,
            seq_length,
            data_keys,
            joint_model=joint_model,
            **kwargs,
        )

        self.train_reg = kwargs["train_regressor"] if "train_regressor" in kwargs else False
        self.reg_steps = kwargs["reg_steps"] if self.train_reg and "reg_steps" in kwargs else -1
        if self.random_rotation:
            self.extended_transforms = deepcopy(self.transforms)
            self.extended_transforms.transforms.insert(
                1,
                tt.RandomAffine(
                    degrees=30,
                    translate=(0.3, 0),
                    scale=(0.7, 1),
                    resample=Image.BILINEAR,
                    fillcolor=(255, 255, 255),
                ),
            )

            self._output_dict.update(
                {"pose_img_inplane": self._get_pose_image_rot}
            )
            self.datakeys.append("pose_img_inplane")

        if self.train_reg and self.reg_steps > 1:
            # self._output_dict.update({"reg_imgs":self._get_reg_imgs, "reg_targets": self._get_reg_targets})
            addkeys = ["reg_imgs", "reg_targets"]
            self.datakeys.extend(addkeys)

        self.label_transfer = False
        self.use_crops_for_app = False
        # self.datapath = "/export/scratch/compvis_datasets/deepfashion_vunet/"

        with open(path.join(self.datapath, "index.p"), "rb") as f:
            self.data = pickle.load(f)

        indices = np.asarray(
            [i for i in range(len(self.data["train"])) if self._filter(i)],
            dtype=np.int,
        )
        self.datadict["img_paths"] = np.asarray(
            [path.join(self.datapath, p) for p in self.data["imgs"]],
            dtype=np.object,
        )[indices]

        self.datadict["norm_keypoints"] = self.data["joints"][indices]
        # keypoints are always normalized
        self.datadict["keypoints"] = self.data["joints"][indices] * 256
        self.datadict["img_size"] = np.full(
            (self.datadict["img_paths"].shape[0], 2), 256
        )
        # dummy pids
        self.datadict["p_ids"] = np.zeros(
            self.datadict["img_paths"].shape[0], dtype=np.int
        )
        self.person_ids = [0]
        self.datadict["train"] = np.asarray(self.data["train"], dtype=np.bool)[
            indices
        ]

        if self.mode == "train":
            self.datadict = {
                key: self.datadict[key][self.datadict["train"]]
                for key in self.datadict
                if self.datadict[key].size > 0
            }
        else:
            self.datadict = {
                key: self.datadict[key][np.logical_not(self.datadict["train"])]
                for key in self.datadict
                if self.datadict[key].size > 0
            }

        self.datadict["map_ids"] = np.arange(
            self.datadict["img_paths"].shape[0]
        )
        self.resample_map_ids()

        print(
            f'Constructed DeepFashion Dataset in "{self.mode}"-mode. Dataset contains of overall {self.__len__()}.'
        )



    def _get_pose_image_rot(self, ids):
        img_paths = self.datadict["img_paths"][ids]
        # rescale keypoints as these are normalized
        keypoints = self.datadict["keypoints"][ids] * 255.0
        prep_img = []
        for p in img_paths:
            # bb_dict =
            pimg = cv2.imread(p)
            pimg = cv2.cvtColor(pimg, cv2.COLOR_BGR2RGB)
            # prep_img.append(self.transforms(pimg))
            prep_img.append(self.extended_transforms(pimg))

        return torch.stack(prep_img, dim=0).squeeze()

    def _get_img_size(self, ids):
        return torch.from_numpy(self.datadict["img_size"][ids])

    def _filter(self, i):
        good = True
        joints = self.data["joints"][i]
        joints = np.float32(joints[self.joint_model.body])
        good = good and valid_joints(joints)
        return good

    def __len__(self):
        return self.datadict["img_paths"].shape[0]

    def resample_map_ids(self):
        print("resampling map_ids.")
        np.random.shuffle(self.datadict["map_ids"])


if __name__ == "__main__":

    from torchvision import transforms as tt
    from PIL import Image
    from data.samplers import PerPersonSampler
    from torch.utils.data import DataLoader
    from os import makedirs
    import yaml

    with open("../config/test_datasets.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    save_path = "./test_data/deepfashion"
    makedirs(save_path, exist_ok=True)

    transforms = tt.Compose(
        [
            tt.ToPILImage(),
            tt.Resize(
                [
                    config["data"]["spatial_size"],
                    config["data"]["spatial_size"],
                ],
                Image.BICUBIC,
            ),
            tt.ToTensor(),
            # tt.Lambda(lambda x: (x.float() / 127.5) - 1.0),
        ]
    )

    dataset = DeepFashionDataset(
        transforms,
        data_keys=["stickman", "pose_img", "app_img"],
        seq_length=0,
        mode="train",
        inplane_normalize=config["data"]["inplane_normalize"],
        box_factor=2,
    )
    sampler = PerPersonSampler(dataset)
    loader = DataLoader(dataset, batch_size=5, sampler=sampler, num_workers=10)

    n_ex = 100

    for i, batch in enumerate(tqdm(loader, total=n_ex)):
        if i >= n_ex:
            break

        imgs = {
            key: (batch[key].permute(0, 2, 3, 1).cpu().numpy() * 255.0).astype(
                np.uint8
            )
            if not isinstance(batch[key], list)
            else (
                batch[key][0].permute(0, 2, 3, 1).cpu().numpy() * 255.0
            ).astype(np.uint8)
            for key in batch
        }

        if config["data"]["inplane_normalize"]:
            app_grid = np.concatenate(
                [
                    imgs["app_img"][..., 3 * i : 3 * (i + 1)]
                    for i in range(imgs["app_img"].shape[-1] // 3)
                ],
                axis=2,
            )
            app_grid = np.concatenate(list(app_grid), axis=0)
            app_grid = cv2.cvtColor(app_grid, cv2.COLOR_RGB2BGR)
            tgt_grid = np.concatenate(
                [
                    imgs["pose_img_inplane"][..., 3 * i : 3 * (i + 1)]
                    for i in range(imgs["pose_img_inplane"].shape[-1] // 3)
                ],
                axis=2,
            )
            tgt_grid = np.concatenate(list(tgt_grid), axis=0)
            tgt_grid = cv2.cvtColor(tgt_grid, cv2.COLOR_RGB2BGR)
            stickmans = np.concatenate(list(imgs["stickman"]), axis=1)
            stickmans = cv2.cvtColor(stickmans, cv2.COLOR_RGB2BGR)

            cv2.imwrite(path.join(save_path, f"app_imgs_inpl{i}.jpg"), app_grid)
            cv2.imwrite(path.join(save_path, f"tgt_imgs_inpl{i}.jpg"), tgt_grid)
            cv2.imwrite(path.join(save_path, f"stickmans{i}.jpg"), stickmans)

        else:

            imgs = {key: list(imgs[key]) for key in imgs}

            if isinstance(batch["pose_img_inplane"], list):
                keypoints = list(
                    (
                        batch["pose_img_inplane"][1]
                        * config["data"]["spatial_size"]
                    ).numpy()
                )
                mask = list(batch["pose_img_inplane"][2].numpy())
                with_points = []
                for j, (img, kps, m) in enumerate(
                    zip(imgs["pose_img_inplane"], keypoints, mask)
                ):
                    img_p = img
                    for kp in kps[m]:
                        img_p = cv2.circle(
                            img_p, (int(kp[0]), int(kp[1])), 2, (0, 0, 255), -1
                        )
                        img_p = (
                            img_p.get()
                            if isinstance(img_p, cv2.UMat)
                            else img_p
                        )

                    with_points.append(img_p)

                imgs["pose_img_inplane"] = with_points

            grid = {
                key: np.concatenate(imgs[key], axis=1)
                for key in imgs
                if key != "app_img"
            }
            grid.update(
                {
                    "overlay": cv2.addWeighted(
                        grid["pose_img"], 0.5, grid["stickman"], 0.5, 0
                    )
                }
            )

            grid = np.concatenate([grid[key] for key in grid], axis=0)
            grid = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
            cv2.imwrite(path.join(save_path, f"test_grid_{i}.jpg"), grid)

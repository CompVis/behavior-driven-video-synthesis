from torchvision import transforms as tt
import numpy as np
from tqdm.autonotebook import tqdm  #
from lib.utils import JointModel, t2p, t3p, t4p, valid_joints
from os import path
from data.base_dataset import BaseDataset
import cv2
import pickle
from functools import partial
from PIL import Image
from copy import deepcopy
import torch


class MarketDataset(BaseDataset):
    def __init__(
        self, transforms, data_keys, seq_length, mode="train", **kwargs
    ):
        assert mode in ["train", "test"]

        # data_keys = ["pose_img","stickman", "app_img", "sample_ids"]
        kwargs["cropp_app"] = False

        joint_model = JointModel(
            body=[8,9,3,2],
            right_lines=[(0, 1), (1, 2), (6, 7), (7, 8)],
            left_lines=[(3, 4), (4, 5), (9, 10), (10, 11)],
            head_lines=[],
            face=[(13,14),(13,15),(14,16),(15,17),],
            rshoulder=8,
            lshoulder=9,
            headup=13,
            kps_to_use=list(range(18)),
            total_relative_joints=[],
            right_hand=[],
            left_hand=[],
            head_part=[],
            kp_to_joint=
            ["rankle","rknee","rhip","lhip","lknee","lankle","rwrist","relbow","rshoulder","lshoulder","lelbow","lwrist","neck","nose", "leye",
                "reye",
                "lear",
                "rear"],
            kps_to_change=None,
            kps_to_change_rel=None,
            norm_T=[t4p,t3p,partial(t2p, ids=[0,1]),partial(t2p, ids=[1,2]),partial(t2p, ids=[6,7]),partial(t2p, ids=[7,8]),partial(t2p,ids=[3,4]),partial(t2p, ids=[4,5]),partial(t2p, ids=[9,10]),partial(t2p, ids=[10,11])],
        )

        self.random_rotation = "inplane_normalize" not in kwargs or not kwargs["inplane_normalize"]

        super().__init__(transforms,mode,seq_length,data_keys,joint_model=joint_model,**kwargs)

        if self.random_rotation:
            self.extended_transforms = deepcopy(self.transforms)
            self.extended_transforms.transforms.insert(1, tt.Pad(
                self.spatial_size // 2,
                padding_mode="reflect"))
            self.extended_transforms.transforms.insert(2, tt.RandomRotation(90,
                                                                            resample=Image.BILINEAR))
            self.extended_transforms.transforms.insert(3, tt.CenterCrop(
                [128,128]))

            self._output_dict.update(
                {"pose_img_inplane": self._get_pose_image_rot})
            self.datakeys.append("pose_img_inplane")

        self.label_transfer = False
        self.use_crops_for_app = False
        # self.datapath = "/export/data/ablattma/Datasets/market"

        with open(path.join(self.datapath,"index.p"),"rb") as f:
            self.data = pickle.load(f)

        indices = np.asarray(
            [i for i in range(len(self.data["train"]))
             if self._filter(i)], dtype=np.int)

        self.datadict["img_paths"] = np.asarray([path.join(self.datapath,p) for p in self.data["imgs"]],dtype=np.object)[indices]
        self.datadict["norm_keypoints"] = np.asarray(self.data["joints"])[indices]
        # keypoints are always normalized
        self.datadict["keypoints"] = np.asarray(self.data["joints"])[indices] * 128
        self.datadict["img_size"] = np.full((self.datadict["img_paths"].shape[0],2),128)
        # dummy pids
        self.datadict["p_ids"] = np.zeros(self.datadict["img_paths"].shape[0],dtype=np.int)
        self.person_ids = [0]
        self.datadict["train"] = np.asarray(self.data["train"],dtype=np.bool)[indices]

        # filter_ids = np.load("/export/home/ablattma/projects/neural_pose_behavior/data/deepfashion_invalid_ids.npy")
        # ids = np.arange(self.datadict["img_paths"].shape[0])
        # self.datadict = {key: self.datadict[key][np.logical_not(np.isin(ids,filter_ids))] for key in self.datadict if self.datadict[key].size>0}

        if self.mode == "train":
            self.datadict = {key: self.datadict[key][self.datadict["train"]] for key in self.datadict if self.datadict[key].size>0}
        else:
            self.datadict = {key: self.datadict[key][np.logical_not(self.datadict["train"])] for
                             key in self.datadict if self.datadict[key].size > 0}

        self.datadict["map_ids"] = np.arange(self.datadict["img_paths"].shape[0])
        self.resample_map_ids()

        print(f"Constructed Market Dataset in \"{self.mode}\"-mode. Dataset contains of overall {self.__len__()}.")


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

    def _get_pose_image_rot(self,ids):
        img_paths = self.datadict["img_paths"][ids]
        prep_img = []
        for p in img_paths:
            pimg = cv2.imread(p)
            pimg = cv2.cvtColor(pimg, cv2.COLOR_BGR2RGB)
            prep_img.append(self.extended_transforms(pimg))

        return torch.stack(prep_img, dim=0).squeeze()



if __name__ == '__main__':
    from torchvision import transforms as tt
    from PIL import Image
    from data.samplers import PerPersonSampler
    from torch.utils.data import DataLoader
    from os import makedirs
    import yaml

    save_path = "./test_data/market"
    makedirs(save_path, exist_ok=True)

    with open("../config/test_datasets.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

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

    dataset = MarketDataset(transforms,
                            data_keys=["stickman", "pose_img", "app_img"],
                            seq_length=0, mode="train")
    sampler = PerPersonSampler(dataset)
    loader = DataLoader(dataset, batch_size=5, sampler=sampler, num_workers=0)
    #
    # # filter out invalid indices
    # invalid_ids = np.asarray([],dtype=np.int)
    # for i in tqdm(np.arange(len(dataset))):
    #     stickman = (dataset._get_stickman([i]).permute(1,2,0).numpy() * 255.).astype(np.uint8)
    #
    #     if np.all(stickman==0):
    #         invalid_ids = np.append(invalid_ids,i)
    #         print(f"Invalid id encountered: {i}, filtering....")
    #
    # np.save("/export/home/ablattma/projects/neural_pose_behavior/data/deepfashion_invalid_ids.npy",invalid_ids)

    n_ex = 100

    for i, batch in enumerate(tqdm(loader, total=n_ex)):
        if i >= n_ex:
            break

        imgs = {
            key: (batch[key].permute(0, 2, 3, 1).cpu().numpy() * 255.).astype(
                np.uint8) for key in batch}

        grid = {key: np.concatenate(list(imgs[key]), axis=1) for key in imgs if
                key != "app_img"}
        grid.update({"overlay": cv2.addWeighted(grid["pose_img"], 0.5,
                                                grid["stickman"], 0.5, 0)})

        grid = np.concatenate([grid[key] for key in grid], axis=0)
        grid = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path.join(save_path, f"test_grid_{i}.jpg"), grid)
import torch
from torch.utils.data import (
    Sampler,
    BatchSampler,
)
from data.base_dataset import BaseDataset
import numpy as np
from numpy import linalg as LA
from tqdm import tqdm

from lib.utils import set_np_random_seed
from copy import deepcopy

# from data.human36m import TimeConsistentDataset


class EntireSequenceSampler(Sampler):
    def __init__(self, dataset, n_unique_videos=1, target_vids=None):
        assert issubclass(
            dataset, BaseDataset
        ), "The dataset utilized in PerPersonSampler has to be of type PerPersonDataset!"
        self.dataset = dataset
        self.has_target_vids = target_vids is not None
        if self.has_target_vids:
            if isinstance(target_vids, np.ndarray):
                self.target_vids = target_vids
            elif isinstance(target_vids, list):
                self.target_vids = np.asarray(target_vids, type=np.int)
            else:
                raise TypeError(
                    "target_vids has to be either of type list or array."
                )

        else:
            self.n_unique_vids = min(
                n_unique_videos,
                np.unique(self.dataset.datadict["v_ids"]).shape[0],
            )
            self._choose_target_vids()

    def _choose_target_vids(self):
        # sample two keys which are subsequently used for testing
        if self.has_target_vids:
            assert np.all(
                np.isin(
                    self.target_vids, np.unique(self.dataset.datadict["v_ids"])
                )
            ), "The target vids specified in EntireSequenceSampler have to be contained in the vids of the dataset."
        else:
            unique_vids = np.unique(self.dataset.datadict["v_ids"])
            self.target_vids = np.random.choice(
                unique_vids, self.n_unique_vids, replace=False
            )

    def __iter__(self):
        self._choose_target_vids()

        print(
            f"New round of sequqences; The sequences with the following vids are chosen {self.target_vids}"
        )
        idxs = list(
            np.sort(
                np.argwhere(
                    np.isin(self.dataset.datadict["v_ids"], self.target_vids)
                )
            )
        )
        return iter([idx[0] for idx in idxs])

    def __len__(self):
        return np.argwhere(
            np.isin(self.dataset.datadict["v_ids"], self.target_vids)
        ).shape[0]


class PerPersonSampler(Sampler):
    def __init__(self, dataset, **kwargs):
        # super().__init__(dataset)
        if "manual_rseed" in kwargs.keys() and kwargs["manual_rseed"]:
            set_np_random_seed()

        if not issubclass(type(dataset), BaseDataset):
            raise TypeError(
                "The dataset utilized in PerPersonSampler must inherit from BaseDataset!"
            )

        if len(dataset.person_ids) == 0:
            raise ValueError(
                "person_ids is an empty list for this dataset. This list must not be empty if usage of PerPersonSampler is intended."
            )

        self.dataset = dataset
        if (
            "sampling_dist" in kwargs.keys()
            and kwargs["sampling_dist"] is not None
        ):
            self.sampling_dist = torch.tensor(kwargs["sampling_dist"])
        else:
            self.sampling_dist = None
        self._randomize_dataset()

    def __iter__(self):
        # randomize maps after every epoch. This should lead to a better generalization
        self._randomize_dataset()

        # randomize indices into the dataset
        n = self.dataset.datadict["img_paths"].shape[0]
        if self.sampling_dist is None:
            return iter(torch.randperm(n).tolist())
        else:
            n = len(self.dataset)
            assert self.sampling_dist.shape[0] == n

            return iter(
                torch.multinomial(
                    self.sampling_dist, n, replacement=True
                ).tolist()
            )

    def __len__(self):
        return self.dataset.datadict["img_paths"].shape[0]

    def _randomize_dataset(self):
        for id in self.dataset.person_ids:
            valid_ids = np.nonzero(
                np.equal(self.dataset.datadict["p_ids"], id)
            )[0]
            map_ids = deepcopy(valid_ids)
            np.random.shuffle(map_ids)
            self.dataset.datadict["map_ids"][valid_ids] = map_ids


# class VideoSequenceSampler(Sampler):
#     def __init__(self, dataset):
#
#         assert isinstance(
#             dataset, TimeConsistentDataset
#         ), 'The dataset must be of type "TimeConsistendDataset".'
#         self.dataset = dataset
#
#     def __iter__(self):
#         np.random.shuffle(self.dataset.id_list)
#         return iter(self.dataset.id_list.tolist())
#
#     def __len__(self):
#         return len(self.dataset.id_list)


class ReconstructionSampler(Sampler):
    def __init__(self, dataset):
        assert issubclass(type(dataset), BaseDataset)
        self.dataset = dataset
        self.dataset.datadict["map_ids"] = np.arange(
            self.dataset.datadict["img_paths"].shape[0]
        )

    def __len__(self):
        return self.dataset.datadict["img_paths"].shape[0]

    def __iter__(self):
        # set map ids for appearances exactly to pids such that reconstruction can be performed
        self.dataset.datadict["map_ids"] = np.arange(
            self.dataset.datadict["img_paths"].shape[0]
        )

        n = self.dataset.datadict["img_paths"].shape[0]
        return iter(torch.randperm(n).tolist())


class WeightedDataSampler(Sampler):
    def __init__(self, dataset, motion_sampling=False, alpha_data=1.0):

        self.dataset = deepcopy(dataset)
        self.motion_sampling = motion_sampling
        self.alpha_data = alpha_data
        self.sample_weights = None

    def __len__(self):
        return len(self.dataset)

    def _get_motion_weights(self):
        # assert required data info is available
        assert "keypoints" in self.dataset.datakeys

        max_valid_lag = self.dataset.seq_length * max(self.dataset.valid_lags)
        motion_per_sample = list()
        for idx in tqdm(
            range(len(self.dataset)),
            desc=f"Compute motion distribution weights: exp_weight={self.alpha_data}",
            total=len(self.dataset),
        ):

            if (
                idx >= len(self.dataset) - max_valid_lag
            ):  # set prob to zero if next frame out of dataset
                motion_tmp = 0.0
            else:
                if self.dataset.prefetched_datadict["sample_ids"] is not None:
                    seq_ids = self.dataset.prefetched_datadict["sample_ids"][
                        idx, :
                    ]
                    pose = self.dataset.datadict["keypoints"][seq_ids[0]][
                        self.dataset.joint_model.kps_to_use, :
                    ].astype(np.float32)
                    pose_next = self.dataset.datadict["keypoints"][seq_ids[-1]][
                        self.dataset.joint_model.kps_to_use, :
                    ].astype(np.float32)
                else:
                    pose = self.dataset.datadict["keypoints"][idx][
                        self.dataset.joint_model.kps_to_use, :
                    ].astype(np.float32)
                    pose_next = self.dataset.datadict["keypoints"][
                        idx + max_valid_lag
                    ][self.dataset.joint_model.kps_to_use, :].astype(np.float32)

                # compute motion magnitude
                pose_res = pose_next - pose
                motion_tmp = LA.norm(pose_res)

            motion_per_sample.append(np.power(motion_tmp, self.alpha_data))

        # compute distribution
        sampling_prob_motion = np.asarray(motion_per_sample)
        return sampling_prob_motion / np.sum(sampling_prob_motion)

    def __iter__(self):
        # choose data sampling
        if self.motion_sampling:
            self.sample_weights = deepcopy(self._get_motion_weights())
            return iter(
                torch.multinomial(
                    torch.tensor(self.sample_weights),
                    len(self.dataset),
                    replacement=True,
                ).tolist()
            )
        else:
            return iter(torch.randperm(len(self.dataset)).tolist())


class SequenceSampler(BatchSampler):
    def __init__(self, dataset ,sampler, batch_size, drop_last):
        super().__init__(sampler, batch_size, drop_last)


        self.dataset = dataset
        self.seq_lengths = self.dataset.seq_length
        self.randomize_map_ids = (
            "paired_keypoints" in self.dataset.datakeys
            or "paired_sample_ids" in self.dataset.datakeys
            or "paired_change" in self.dataset.datakeys
        )

    # def __len__(self):
    #     return len(self.sampler.data_source)

    def __iter__(self):
        if self.randomize_map_ids:
            self.dataset.resample_map_ids()

        batch = []

        # sample sequence length
        seq_len = np.random.choice(
            range(self.seq_lengths[0], self.seq_lengths[1]), 1
        )[0]

        for idx in self.sampler:
            batch.append([idx, seq_len])
            if len(batch) == self.batch_size:
                yield batch
                batch = []

                # sample sequence length
                seq_len = np.random.choice(
                    range(self.seq_lengths[0], self.seq_lengths[1]), 1
                )[0]

        if len(batch) > 0 and not self.drop_last:
            yield batch

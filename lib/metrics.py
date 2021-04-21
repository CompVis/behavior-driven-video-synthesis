import torch
from torch import nn
from torch.nn import DataParallel
from torch.utils.data import DataLoader
import numpy as np
from data import get_dataset
from data.samplers import ReconstructionSampler
from tqdm.autonotebook import tqdm
from lib.utils import scale_img
from skimage.metrics import structural_similarity
from models.imagenet_pretrained import FIDInceptionModel
from scipy import linalg
from scipy.stats import entropy
import cv2
from os import path
from copy import deepcopy
from data.data_conversions_3d import revert_output_format,rotmat2euler,expmap2rotmat
from models.vunets import VunetOrg
from torchvision.models import inception_v3


def compute_ssim(model: torch.nn.Module, devices, data_keys, debug=False, **kwargs):
    # if "mode" in kwargs.keys():
    #     kw = {key: kwargs[key] for key in kwargs if key != "mode"}
    # else:
    #     kw = kwargs

    test_batch_size=kwargs["test_batch_size"] if "test_batch_size" in kwargs else 8
    num_workers = kwargs["n_data_workers"] if "n_data_workers" in kwargs else 8
    max_n_samples = kwargs["max_n_samples"] if "max_n_samples" in kwargs else 8000
    inplane_normalize = kwargs["inplane_normalize"] if "inplane_normalize" in kwargs else False

    dataset, transforms = get_dataset(kwargs)


    ssim_dataset = dataset(
        transforms, data_keys=data_keys, mode="test", **kwargs
    )

    print(f"Length of dataset for ssim computation is {len(ssim_dataset)}")
    reconstruction_sampler = ReconstructionSampler(ssim_dataset)
    # use drop_last = True to have unbiased estimator
    ssim_loader = DataLoader(
        ssim_dataset,
        batch_size=test_batch_size,
        sampler=reconstruction_sampler,
        drop_last=True,
        num_workers=num_workers,
        # pin_memory=True,
    )
    ssim_it = iter(ssim_loader)

    n_max = (
        10
        if debug
        else min(max_n_samples, len(ssim_dataset))
    )

    # generate reconstructions
    model.eval()

    ssims = np.asarray([], dtype=np.float)
    for batch_nr, data in enumerate(
        tqdm(ssim_it, desc="Inferring images for ssim computation.")
    ):
        if not isinstance(model, DataParallel):
            imgs = {name: data[name].to(devices[0]) for name in data_keys}
        else:
            imgs = {name: data[name] for name in data_keys}

        if isinstance(model,VunetOrg):
            app_img = imgs["app_img"]
        else:
            app_img = imgs["pose_img_inplane"] if inplane_normalize else imgs["pose_img"]

        stickman = imgs["stickman"]
        target_img = imgs["pose_img"]

        with torch.no_grad():
            if "mode" in kwargs.keys():
                out = model(app_img, stickman, mode=kwargs["mode"])
            else:
                out = model(app_img, stickman)
            img_rec = out[0]

        # scale for ssim
        img_rec = scale_img(img_rec)
        target_img = scale_img(target_img)

        # to numpy format
        img_rec = img_rec.permute(0, 2, 3, 1).cpu().numpy()
        target_img = target_img.permute(0, 2, 3, 1).cpu().numpy()

        # compute ssim values for batch, implementation of Wang et. al.
        ssim_batch = np.asarray(
            [
                structural_similarity(
                    rimg,
                    timg,
                    multichannel=True,
                    data_range=1.0,
                    gaussian_weights=True,
                    use_sample_covariance=False,
                )
                for rimg, timg in zip(img_rec, target_img)
            ]
        )
        ssims = np.append(ssims, ssim_batch)

        if (batch_nr + 1) * test_batch_size >= n_max:
            break

    ssim = ssims.mean()
    print(f"Computed average SSIM between {n_max} samples: SSIM = {ssim}")

    return ssim


def compute_fid(model, data_keys, devices, debug=False, **kwargs):
    print("Compute FID score...")

    if "mode" in kwargs.keys():
        kw = {key: kwargs[key] for key in kwargs if key != "mode"}
    else:
        kw = kwargs

    assert "dataset" in kwargs
    dataset_name = kwargs["dataset"]
    test_batch_size = kwargs[
        "test_batch_size"] if "test_batch_size" in kwargs else 8
    num_workers = kwargs["n_data_workers"] if "n_data_workers" in kwargs else 8
    max_n_samples = kwargs[
        "max_n_samples"] if "max_n_samples" in kwargs else 8000
    inplane_normalize = kwargs[
        "inplane_normalize"] if "inplane_normalize" in kwargs else False


    # compute inception features for gt data
    inc_model = FIDInceptionModel()
    if isinstance(model, DataParallel):
        inc_model = DataParallel(inc_model, device_ids=devices)

    inc_model.to(devices[0])
    inc_model.eval()

    dataset, transforms = get_dataset(kwargs)


    fid_file_name = f"./{dataset_name}-fid-features.npy"
    fid_dataset = dataset(
        transforms, data_keys=data_keys, mode="test", **kw
    )

    n_max = min(12000, len(fid_dataset))
    if debug:
        n_max = 40
    print(f"n_max for fid computation is {n_max}")

    is_precomputed = path.isfile(fid_file_name)

    if is_precomputed:
        all_gt_features = np.load(fid_file_name)

        if debug:
            all_gt_features=all_gt_features[:n_max]
        else:
            n_max = all_gt_features.shape[0]
    else:
        reconstruction_sampler = ReconstructionSampler(fid_dataset)
        # use drop_last = True to have unbiased estimator
        fid_loader = DataLoader(
            fid_dataset,
            batch_size=test_batch_size,
            sampler=reconstruction_sampler,
            drop_last=True,
            num_workers=10,
            # pin_memory=True,
        )
        # fid_it = iter(fid_loader)

        # compute for 12000 samples

        all_gt_features = []
        for batch_nr, batch in enumerate(
            tqdm(
                fid_loader,
                desc="Compute inceptionv3 features on ground truth data...",
            )
        ):
            if isinstance(inc_model, DataParallel):
                imgs = {name: batch[name] for name in data_keys}
            else:
                imgs = {name: batch[name].to(devices[0]) for name in data_keys}

            gt = imgs["pose_img"]

            if not isinstance(inc_model, DataParallel):
                gt = gt.to(devices[0])
            with torch.no_grad():
                gt_features = inc_model(gt)

            all_gt_features.append(gt_features.cpu().numpy())

            if (batch_nr + 1) * test_batch_size >= n_max:
                break

        all_gt_features = np.concatenate(all_gt_features, axis=0)
        if not debug:
            np.save(fid_file_name, all_gt_features)

    mu_gt = np.mean(all_gt_features, axis=0)
    cov_gt = np.cov(all_gt_features, rowvar=False)

    # compute inception features for sythesized data
    model.eval()
    reconstruction_sampler = ReconstructionSampler(fid_dataset)
    # use drop_last = True to have unbiased estimator
    fid_loader = DataLoader(
        fid_dataset,
        batch_size=test_batch_size,
        sampler=reconstruction_sampler,
        drop_last=True,
        num_workers=10,
        # pin_memory=True,
    )
    # fid_it = iter(fid_loader)

    all_gen_features = []
    for batch_nr, batch in enumerate(
        tqdm(
            fid_loader,
            desc="Generate data and compute inceptionv3 features from that...",
        )
    ):



        if isinstance(model, DataParallel):
            imgs = {name: batch[name] for name in data_keys}
        else:
            imgs = {name: batch[name].to(devices[0]) for name in data_keys}

        if isinstance(model,VunetOrg):
            app_img = imgs["app_img"]
        else:
            app_img = imgs["pose_img_inplane"] if inplane_normalize else imgs["pose_img"]
        shape_img = imgs["stickman"]
        target_img = deepcopy(imgs["pose_img"])

        with torch.no_grad():
            # train is reconstruction mode
            if "mode" in kwargs.keys():
                out = model(app_img, shape_img, mode="train")
            else:
                out = model(app_img, shape_img)

            rec_img = out[0]

            rec_features = inc_model(rec_img)

        rec_features_cp = deepcopy(rec_features)
        all_gen_features.append(rec_features_cp.cpu().numpy())
        del rec_features
        del rec_img
        del target_img
        del out
        del shape_img
        if (batch_nr + 1) * test_batch_size >= n_max:
            break

    all_gen_features = np.concatenate(all_gen_features, axis=0)
    mu_gen = np.mean(all_gen_features, axis=0)
    cov_gen = np.cov(all_gen_features, rowvar=False)

    fid = _calculate_fid(mu_gt, cov_gt, mu_gen, cov_gen)

    print(
        f"Computed average FID between {n_max} generated and ground truth samples: FID = {fid}"
    )

    return fid


def _calculate_fid(mu1, cov1, mu2, cov2, eps=1e-6):
    # Taken from https://github.com/mseitzer/pytorch-fid/blob/master/fid_score.py
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(cov1)
    sigma2 = np.atleast_2d(cov2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = f"fid calculation produces singular product; adding {eps} to diagonal of cov estimates"

        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

def mse_euler(error_dict,rec,gt,actions,data_mean,data_std,dim_to_ignore,max_len):
    assert len(actions) == rec.shape[0]
    # loop over batches
    for rec_seq,gt_seq,action in zip(rec,gt,actions):
        if len(error_dict[action][2])>max_len:
            continue
        rec_seq = revert_output_format(rec_seq, data_mean, data_std, dim_to_ignore)
        gt_seq = revert_output_format(gt_seq, data_mean, data_std, dim_to_ignore)
        # loop over sequence
        gt_seq_euler = []
        rec_seq_euler = []
        for rec_frame,gt_frame in zip(rec_seq,gt_seq):

            for idx in range(3,97,3):
                # bring in euler angles representation
                rec_frame[idx:idx+3] = rotmat2euler(expmap2rotmat(rec_frame[idx:idx+3]))
                gt_frame[idx:idx + 3] = rotmat2euler(
                    expmap2rotmat(gt_frame[idx:idx + 3]))

            gt_seq_euler.append(gt_frame)
            rec_seq_euler.append(rec_frame)

        gt_seq_euler = np.stack(gt_seq_euler,axis=0)
        rec_seq_euler = np.stack(rec_seq_euler, axis=0)

        # set global t and r to 0
        gt_seq_euler[:,:6] = 0
        # rec_seq_euler[:,:6] = 0

        idx_to_use = np.where(np.std(gt_seq_euler, 0) > 1e-4)[0]

        euc_error = np.power(
            gt_seq_euler[:, idx_to_use] - rec_seq_euler[:, idx_to_use], 2)
        for e in error_dict[action]:
            euclidean_per_frame = np.sqrt(np.sum(euc_error[:e], 1))
            mean_euclidean = np.mean(euclidean_per_frame)
            error_dict[action][e].append(mean_euclidean)


# function based on implementation of https://github.com/sbarratt/inception-score-pytorch/blob/master/inception_score.py

def inception_score(imgs, dev, batch_size=32, resize=False, splits=1, debug = False):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    dev -- the device to use
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """

    print(f"Computing Inception Score...")
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size,num_workers=0)

    # Load inception model
    print("Load inception model.")
    inception_model = inception_v3(pretrained=True, transform_input=False)
    inception_model.to(dev)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear').to(torch.float).to(dev)
    softmax = nn.Softmax()
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return softmax(x).detach().cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(tqdm(dataloader,desc="Generating inception features for generated images."), 0):
        if debug and i * dataloader.batch_size >= 40:
            break
        batchv = batch[0].to(dev)
        batch_size_i = batchv.shape[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


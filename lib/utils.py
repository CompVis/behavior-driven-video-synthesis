import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import os.path as osp
import random
from collections import namedtuple
import multiprocessing as mp
from collections import abc
from scipy.spatial import ConvexHull
from tqdm.autonotebook import tqdm
import kornia
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import operator
from functools import reduce
from ignite.metrics import Average
from ignite.engine import Events



JointModel = namedtuple(
    "JointModel",
    "body right_lines left_lines head_lines face rshoulder lshoulder headup kps_to_use right_hand left_hand head_part total_relative_joints kp_to_joint kps_to_change kps_to_change_rel norm_T",
)


def prod(iterable):
    return reduce(operator.mul, iterable, 1)


def n_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_member(model, name):
    if isinstance(model, nn.DataParallel):
        module = model.module
    else:
        module = model

    return getattr(module, name)


def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)


def get_line_colors(n_lines_per_channel):
    line_colors = []
    for channel, nr_lines in enumerate(n_lines_per_channel):

        intervall = int(255 // (nr_lines + 1))
        line_colors_per_channel = []

        for i in range(nr_lines):
            col = [0, 0, 0]
            col[channel] = (i + 1) * intervall
            line_colors_per_channel.append(col)

        line_colors.append(line_colors_per_channel)

    assert len(line_colors) == len(n_lines_per_channel)

    return line_colors


def t5p(kps, jm: JointModel, wh, oh):
    # transform for body, if body consists of 5 points
    part_kps = kps[jm.body, :2]

    #
    neck = part_kps[2]
    # rshoulder - lshoulder depicts direction
    ls_to_rs = part_kps[1] - part_kps[3]
    rh_to_rs = part_kps[1] - part_kps[0]
    lh_to_ls = part_kps[3] - part_kps[-1]

    rhip = part_kps[0]
    lhip = part_kps[-1]

    # computation of upper body points of body quadrangle
    lambda_l = (lhip[1] - neck[1]) * lh_to_ls[0] / (
        ls_to_rs[1] * lh_to_ls[0] - ls_to_rs[0] * lh_to_ls[1]
    ) + (neck[0] - lhip[0]) * lh_to_ls[1] / (
        ls_to_rs[1] * lh_to_ls[0] - ls_to_rs[0] * lh_to_ls[1]
    )
    lambda_r = (rhip[1] - neck[1]) * rh_to_rs[0] / (
        ls_to_rs[1] * rh_to_rs[0] - ls_to_rs[0] * rh_to_rs[1]
    ) + (neck[0] - rhip[0]) * rh_to_rs[1] / (
        ls_to_rs[1] * rh_to_rs[0] - ls_to_rs[0] * rh_to_rs[1]
    )

    p1 = (neck + lambda_r * ls_to_rs).astype(np.float32)
    p2 = (neck + lambda_l * ls_to_rs).astype(np.float32)
    p3 = lhip.astype(np.float32)
    p4 = rhip.astype(np.float32)

    # points = [p1, p2, p3, p4]
    # x = lambda x: x[0]
    # y = lambda x: x[1]
    # sorted_x = list(sorted(points, key=x))
    # max_x = max(points, key=x)
    # max_y = max(points, key=y)
    #
    # # top left and bottom right
    # tl = min(points, key=lambda p: p[0] + max_y[1] - p[1])
    # br = max(points, key=lambda p: max_x[0] - p[0] + p[1])
    # remainder = [
    #     p
    #     for p in points
    #     if np.any(np.not_equal(p, tl)) and np.any(np.not_equal(p, br))
    # ]
    # s = sorted(remainder, key=y, reverse=True)
    # bl = s[0]
    # tr = s[1]

    # points_src = np.float32([tl, tr, br, bl])
    points_src = np.float32([p1, p2, p3, p4])
    points_dst = np.float32(
        np.multiply(
            np.asarray(
                [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                dtype=np.float32,
            ),
            np.asarray(wh, dtype=np.float32),
        )
    )

    T = cv2.getPerspectiveTransform(points_src, points_dst)
    return T


def t4p(kps, jm: JointModel, wh, oh):
    points_src = np.float32(kps[jm.body])
    points_dst = np.multiply(
        np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32
        ),
        np.asarray(wh, dtype=np.float32),
    )

    T = cv2.getPerspectiveTransform(points_src, points_dst)

    return T


def t3p(kps, jm: JointModel, wh, oh):

    if not valid_joints(np.asarray([kps[jm.rshoulder],kps[jm.lshoulder],kps[jm.headup]])):
        bpart_indices = [jm.lshoulder,jm.rshoulder,jm.rshoulder]
        part_src = np.float32(kps[bpart_indices])

        if not valid_joints(part_src):
            return None

        segment = part_src[1] - part_src[0]
        normal = np.array([-segment[1], segment[0]])
        if normal[1] > 0.0:
            normal = -normal

        a = part_src[0] + normal
        b = part_src[0]
        c = part_src[1]
        d = part_src[1] + normal
    else:
        neck = 0.5 * (kps[jm.rshoulder] + kps[jm.lshoulder])
        neck_to_nose = kps[jm.headup] - neck
        part_src = np.float32([neck + 2 * neck_to_nose, neck])

        # segment box
        segment = part_src[1] - part_src[0]
        normal = np.array([-segment[1], segment[0]])
        alpha = 1.0 / 2.0
        a = part_src[0] + alpha * normal
        b = part_src[0] - alpha * normal
        c = part_src[1] - alpha * normal
        d = part_src[1] + alpha * normal
    # part_src = np.float32([a,b,c,d])
    points_src = np.float32([b, c, d, a])

    dst = np.asarray(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]], dtype=np.float32
    )
    wh = np.asarray(wh, dtype=np.float32)
    points_dst = np.multiply(dst, wh)

    T = cv2.getPerspectiveTransform(points_src, points_dst)

    return T


def t2p(kps, ids: tuple, wh, oh, jm=None):

    if np.any(np.all(kps[ids] <= 0.0, axis=1)):
        # leg fallback
        nni = np.nonzero(np.all(kps[ids] > 0.0, axis=1))[0]
        if nni.size == 0:
            return None
        t_id = ids[int(nni)]

        a = kps[t_id]
        b = np.float32([a[0], oh - 1])
        points_src = np.asarray([a, b], dtype=np.float32)
        segment = points_src[1] - points_src[0]
        normal = np.array([-segment[1], segment[0]])
        alpha = 1. / 4.0
        a = points_src[0] + alpha * normal
        b = points_src[0] - alpha * normal
        c = points_src[1] - alpha * normal
        d = points_src[1] + alpha * normal
        points_src = np.float32([a, b, c, d])
    else:
        segment = kps[ids[1]] - kps[ids[0]]
        normal = np.array([-segment[1], segment[0]])
        alpha = 1.0 / 4.0
        a = kps[ids[0]] + alpha * normal
        b = kps[ids[0]] - alpha * normal
        c = kps[ids[1]] - alpha * normal
        d = kps[ids[1]] + alpha * normal
        points_src = np.asarray([a, b, c, d], dtype=np.float32)
    dst = np.asarray(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]], dtype=np.float32
    )
    wh = np.asarray(wh, dtype=np.float32)
    points_dst = np.multiply(dst, wh) - 1.0

    T = cv2.getPerspectiveTransform(points_src, points_dst)

    return T


def get_img_crop(img_batch, target_kps, name, spatial_size, box_factor):
    """
    Crops an image around a set of keypoints or a single keypoint
    """
    # leave out batch dimension
    if name == "head":
        # kepyoints are assumed to be (rshoulder, lshoulder, head)
        assert target_kps.shape[1] == 3
        necks = 0.5 * (target_kps[:, 0] + target_kps[:, 1])
        necks_to_noses = target_kps[:, 2] - necks
        up_heads = necks + 2 * necks_to_noses

        segments = necks - up_heads
        normals = torch.stack([-segments[:, 1], segments[:, 0]], dim=-1)

        alpha = 0.5
        a = up_heads + alpha * normals
        b = up_heads - alpha * normals
        c = necks - alpha * normals
        d = necks + alpha * normals
    elif name == "hand":
        assert target_kps.shape[1] == 2
        # target keypoints are assumed to be (wrist,hand) --> segments point from wrist to hand
        segments = target_kps[:, 1] - target_kps[:, 0]
        # s_norm = torch.norm(segments, dim=1, p=2).unsqueeze(dim=1)
        # # normals are rotated in mathematical positive direction
        normals = torch.stack([-segments[:, 1], segments[:, 0]], dim=-1)
        # n_norm = torch.norm(normals, dim=1, p=2).unsqueeze(dim=1)
        # # bisector of segments and vectors
        # bisectors = torch.mul(normals,s_norm) + torch.mul(segments,n_norm)
        # # should have same norm as normals
        # bisectors = torch.div(bisectors, 2 * s_norm)
        # alpha = 0.5
        # rot_90 = torch.stack([-bisectors[:, 1], bisectors[:, 0]], dim=-1)
        # a = target_kps[:,0] - alpha * bisectors
        # b = target_kps[:,0] + alpha * rot_90
        # c = target_kps[:,1] + alpha * bisectors
        # d = target_kps[:,1] - alpha * rot_90

        alpha = 1.0
        beta = 0.25
        a = target_kps[:, 0] + alpha * normals - beta * segments
        b = target_kps[:, 0] - alpha * normals - beta * segments
        c = target_kps[:, 1] - alpha * normals + beta * segments
        d = target_kps[:, 1] + alpha * normals + beta * segments
    else:
        raise ValueError("Invalid ids or keypoints.")

    src_windows = torch.stack([a, b, c, d], dim=1).to(torch.float)
    dev = src_windows.get_device() if src_windows.get_device() > 0 else "cpu"
    dst = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, spatial_size // (2 ** box_factor) - 1.0],
            [
                spatial_size // (2 ** box_factor) - 1.0,
                spatial_size // (2 ** box_factor) - 1.0,
            ],
            [spatial_size // (2 ** box_factor) - 1.0, 0.0],
        ],
        dtype=torch.float,
        device=dev,
    )
    dst_windows = torch.stack([dst] * src_windows.shape[0], dim=0).to(
        torch.float
    )

    M = kornia.get_perspective_transform(src_windows, dst_windows)
    if dev != "cpu":
        with torch.cuda.device(dev):
            crop = kornia.warp_perspective(
                img_batch,
                M,
                dsize=(
                    spatial_size // (2 ** box_factor),
                    spatial_size // (2 ** box_factor),
                ),
            )
    else:
        crop = kornia.warp_perspective(
            img_batch,
            M,
            dsize=(
                spatial_size // (2 ** box_factor),
                spatial_size // (2 ** box_factor),
            ),
        )
    return crop


def make_joint_img(
    img_shape,
    joints,
    joint_model: JointModel,
    line_colors=None,
    color_channel=None,
    scale_factor=None,
):
    # channels are opencv so g, b, r
    scale_factor = (
        int(img_shape[1] // scale_factor)
        if scale_factor is not None
        else 1
    )
    thickness = scale_factor

    imgs = list()
    for i in range(3):
        imgs.append(np.zeros(img_shape[:2], dtype="uint8"))

    if len(joint_model.body) > 2:
        body_pts = np.array([[joints[part, :] for part in joint_model.body]])
        valid_pts = np.all(np.greater_equal(body_pts, [0.0, 0.0]), axis=-1)
        if np.count_nonzero(valid_pts) > 2:
            body_pts = np.int_([body_pts[valid_pts]])
            if color_channel is None:
                body_color = (0, 127, 255)
                for i, c in enumerate(body_color):
                    cv2.fillPoly(imgs[i], body_pts, c)
            else:
                cv2.fillPoly(imgs[color_channel], body_pts, 255)

    for line_nr, line in enumerate(joint_model.right_lines):
        valid_pts = np.greater_equal(joints[line, :], [0.0, 0.0])
        if np.all(valid_pts):
            a = tuple(np.int_(joints[line[0], :]))
            b = tuple(np.int_(joints[line[1], :]))
            if color_channel is None:
                if line_colors is not None:
                    channel = int(np.nonzero(line_colors[0][line_nr])[0])
                    cv2.line(
                        imgs[channel],
                        a,
                        b,
                        color=line_colors[0][line_nr][channel],
                        thickness=thickness,
                    )
                else:
                    cv2.line(imgs[1], a, b, color=255, thickness=thickness)
            else:
                cv2.line(
                    imgs[color_channel], a, b, color=255, thickness=thickness
                )

    for line_nr, line in enumerate(joint_model.left_lines):

        valid_pts = np.greater_equal(joints[line, :], [0.0, 0.0])
        if np.all(valid_pts):
            a = tuple(np.int_(joints[line[0], :]))
            b = tuple(np.int_(joints[line[1], :]))
            if color_channel is None:
                if line_colors is not None:
                    channel = int(np.nonzero(line_colors[1][line_nr])[0])
                    cv2.line(
                        imgs[channel],
                        a,
                        b,
                        color=line_colors[1][line_nr][channel],
                        thickness=thickness,
                    )
                else:

                    cv2.line(imgs[0], a, b, color=255, thickness=thickness)
            else:
                cv2.line(
                    imgs[color_channel], a, b, color=255, thickness=thickness
                )

    if len(joint_model.head_lines) == 0:
        rs = joints[joint_model.rshoulder, :]
        ls = joints[joint_model.lshoulder, :]
        cn = joints[joint_model.headup, :]
        if np.any(np.less(np.stack([rs, ls], axis=-1), [0.0, 0.0])):
            neck = np.asarray([-1.0, -1.0])
        else:
            neck = 0.5 * (rs + ls)

        pts = np.stack([neck, cn], axis=-1).transpose()

        valid_pts = np.greater_equal(pts, [0.0, 0.0])
        throat_len = np.asarray([0], dtype=np.float)
        if np.all(valid_pts):
            throat_len = np.linalg.norm(pts[0] - pts[1])
            if color_channel is None:
                a = tuple(np.int_(pts[0, :]))
                b = tuple(np.int_(pts[1, :]))
                cv2.line(imgs[0], a, b, color=127, thickness=thickness)
                cv2.line(imgs[1], a, b, color=127, thickness=thickness)
            else:
                cv2.line(
                    imgs[color_channel],
                    tuple(np.int_(pts[0, :])),
                    tuple(np.int_(pts[1, :])),
                    color=255,
                    thickness=thickness,
                )
    else:
        throat_lens = np.zeros(len(joint_model.head_lines), dtype=np.float)
        for line_nr, line in enumerate(joint_model.head_lines):

            valid_pts = np.greater_equal(joints[line, :], [0.0, 0.0])
            if np.all(valid_pts):
                throat_lens[line_nr] = np.linalg.norm(
                    joints[line[0], :] - joints[line[1], :]
                )
                a = tuple(np.int_(joints[line[0], :]))
                b = tuple(np.int_(joints[line[1], :]))
                if color_channel is None:
                    if line_colors is not None:
                        channel = int(np.nonzero(line_colors[2][line_nr])[0])
                        cv2.line(
                            imgs[channel],
                            a,
                            b,
                            color=line_colors[2][line_nr][channel],
                            thickness=thickness,
                        )
                    else:
                        cv2.line(imgs[0], a, b, color=127, thickness=thickness)
                        cv2.line(imgs[1], a, b, color=127, thickness=thickness)
                else:
                    cv2.line(
                        imgs[color_channel],
                        a,
                        b,
                        color=255,
                        thickness=thickness,
                    )

        if throat_lens.size > 0:
            throat_len = np.amax(throat_lens)
        else:
            throat_len = 0
    if len(joint_model.face) > 0:
        for line_nr, line in enumerate(joint_model.face):

            valid_pts = np.greater_equal(joints[line, :], [0.0, 0.0])
            if np.all(valid_pts):
                if (
                    np.linalg.norm(joints[line[0], :] - joints[line[1], :])
                    < throat_len
                ):
                    a = tuple(np.int_(joints[line[0], :]))
                    b = tuple(np.int_(joints[line[1], :]))
                    if color_channel is None:
                        if line_colors is not None:
                            channel = int(
                                np.nonzero(line_colors[2][line_nr])[0]
                            )
                            cv2.line(
                                imgs[channel],
                                a,
                                b,
                                color=line_colors[2][line_nr][channel],
                                thickness=thickness,
                            )
                        else:
                            cv2.line(
                                imgs[0], a, b, color=127, thickness=thickness
                            )
                            cv2.line(
                                imgs[1], a, b, color=127, thickness=thickness
                            )
                    else:
                        cv2.line(
                            imgs[color_channel],
                            a,
                            b,
                            color=255,
                            thickness=thickness,
                        )


    img = np.stack(imgs, axis=-1)
    if img_shape[-1] == 1:
        img = np.mean(img, axis=-1)[:, :, None]
    # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return img


def valid_joints(*joints):
    j = np.stack(joints)
    return (j >= 0).all()


def linear_var(
    act_it, start_it, end_it, start_val, end_val, clip_min, clip_max
):
    act_val = (
        float(end_val - start_val) / (end_it - start_it) * (act_it - start_it)
        + start_val
    )
    return np.clip(act_val, a_min=clip_min, a_max=clip_max)


def bounding_box_batch(kps_batch, img_batch, size_batch, spatial_size):
    """
    Crops the image to the bounding box around the keypoints, per batch, and resizes them to the target size
    :return:
    """
    dev = kps_batch.get_device()

    # target_sizes = spatial_size * torch.ones_like(
    #     size_batch, device=dev if dev >= 0 else "cpu", dtype=torch.float
    # )
    # scales = target_sizes / size_batch.to(torch.float)
    # scales = scales.unsqueeze(dim=-2)

    # kps = kps_batch * scales

    # kps = [
    #     kp.squeeze(dim=0).cpu().numpy()
    #     for kp in torch.split(kps, split_size_or_sections=1, dim=0)
    # ]

    bbox = torch.stack(
        [bb_for_pt_kornia(kp.cpu().numpy(), [spatial_size, spatial_size], dev=dev,relax=0) for kp in kps_batch],
        dim=0,
    )
    with torch.cuda.device(dev):
        crops = kornia.crop_and_resize(
            img_batch, bbox, (spatial_size, spatial_size),
        )
    return crops

def bb_for_pt_kornia(pts, img_shape, dev, relax=0.1):

    pts = pts[np.all(pts>0.,axis=1)]

    x, y, w, h = cv2.boundingRect(pts[:, :2].astype(np.float32))
    # if w >= h:
    #     x_tl = max(0, int(x - relax * w))
    #     x_br = min(img_shape[1], int(x + (1.0 + relax) * w))
    #
    #     y_tl = max(0, int((2 * y + h - (x_br - x_tl)) / 2))
    #     y_br = min(img_shape[0], int((2 * y + h + (x_br - x_tl)) / 2))
    # else:
    #     y_tl = max(0, int(y - relax * h))
    #     y_br = min(img_shape[0], int(y + (1.0 + relax) * h))
    #
    #     x_tl = max(0, int((2 * x + w - (y_br - y_tl)) / 2))
    #     x_br = min(img_shape[1], int((2 * x + w + (y_br - y_tl)) / 2))

    bbox = [x, x+w, y, y+h]
    out = [
        [bbox[0], bbox[2]],
        [bbox[1], bbox[2]],
        [bbox[1], bbox[3]],
        [bbox[0], bbox[3]],
    ]
    return torch.tensor(out, device=dev if dev > 0 else "cpu")


def bb_for_pt(pts, img_shape, dev, relax=0.1):

    pts = pts[np.all(pts>0.,axis=1)]

    x, y, w, h = cv2.boundingRect(pts[:, :2])
    if w >= h:
        x_tl = max(0, int(x - relax * w))
        x_br = min(img_shape[1], int(x + (1.0 + relax) * w))

        y_tl = max(0, int((2 * y + h - (x_br - x_tl)) / 2))
        y_br = min(img_shape[0], int((2 * y + h + (x_br - x_tl)) / 2))
    else:
        y_tl = max(0, int(y - relax * h))
        y_br = min(img_shape[0], int(y + (1.0 + relax) * h))

        x_tl = max(0, int((2 * x + w - (y_br - y_tl)) / 2))
        x_br = min(img_shape[1], int((2 * x + w + (y_br - y_tl)) / 2))

    bbox = [x_tl, x_br, y_tl, y_br]
    out = [
        [bbox[0], bbox[2]],
        [bbox[1], bbox[2]],
        [bbox[1], bbox[3]],
        [bbox[0], bbox[3]],
    ]
    return torch.tensor(out, device=dev if dev > 0 else "cpu")


def get_bounding_box(pts, img_shape, relax=0.1):
    """
    Computes the bounding box of a set of points with same height and width, based on the size of the larger side
    :param pts:
    :param img_shape:
    :param relax:
    :return:
    """
    # format is x,y,w,h
    x, y, w, h = cv2.boundingRect(pts[:, :2])
    if w >= h:
        x_tl = int(x - relax * w)
        x_br = int(x + (1.0 + relax) * w)

        y_tl = int((2 * y + h - (x_br - x_tl)) / 2)
        y_br = int((2 * y + h + (x_br - x_tl)) / 2)
    else:
        y_tl = int(y - relax * h)
        y_br = int(y + (1.0 + relax) * h)

        x_tl = int((2 * x + w - (y_br - y_tl)) / 2)
        x_br = int((2 * x + w + (y_br - y_tl)) / 2)

    bbox = [x_tl, x_br, y_tl, y_br]

    pad_left = abs(min(0, bbox[0]))
    pad_right = abs(max(0, bbox[1] - img_shape[1]))
    pad_top = abs(min(0, bbox[2]))
    pad_bottom = abs(max(0, bbox[-1] - img_shape[0]))
    bbox[0] += pad_left
    bbox[1] += pad_left + pad_right
    bbox[2] += pad_top
    bbox[-1] += pad_top + pad_bottom

    return {
        "bbox": bbox,
        "pads": np.asarray(
            [pad_left, pad_right, pad_top, pad_bottom], dtype=np.int
        ),
    }


def scale_img(x):
    """
    Scale in between 0 and 1
    :param x:
    :return:
    """
    # ma = torch.max(x)
    # mi = torch.min(x)
    out = (x + 1.0) / 2.0
    out = torch.clamp(out, 0.0, 1.0)
    return out


def add_summary_writer(net, path):

    writer = SummaryWriter(log_dir=path)
    # dat = iter(loader)
    # app_img, shape_img, target_img = next(dat)

    try:
        writer.add_graph(net)
    except Exception as ex:
        print("Failed to add graph to generator: {}".format(ex))

    return writer


def save_tensor_as_img(t, path, scale, offset=0, prefix=None):
    if scale:
        imgs = scale_img(t)
    imgs = imgs.permute(0, 3, 2, 1).mul(255.0).to(torch.uint8).unbind(dim=0)
    for i, img in enumerate(imgs):
        img = cv2.cvtColor(img.cpu().numpy(), cv2.COLOR_RGB2BGR)
        if prefix is None:
            cv2.imwrite(osp.join(path, "%0.6d.png" % (offset + i)), img)
        else:
            cv2.imwrite(
                osp.join(path, prefix + "%0.6d.png" % (offset + i)), img
            )


def make_img_grid(img_list):
    """
    Arranges a list of batched images as a numpy image grid
    :return:
    """
    assert isinstance(img_list, list) and torch.is_tensor(img_list[0])
    n_images = img_list[0].shape[0]
    l = len(img_list)
    assert np.all(
        [torch.is_tensor(e) and e.shape[0] == n_images for e in img_list]
    )

    stacked = torch.cat(img_list, dim=0)
    grid = make_grid(
        stacked, nrow=int(stacked.shape[0] // l), padding=10
    ).unsqueeze(dim=0)

    return grid


def set_np_random_seed():
    np.random.seed(0)


def set_pytorch_random_seed():
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_python_random_seed():
    random.seed(1989)


def add_joints_to_img(
    img,
    kps,
    joints,
    color_kps,
    color_joints,
    # thicken=False
):

    # draw joints
    if len(color_joints) == 1:
        color_joints = [color_joints[0] for _ in range(len(joints))]

    for i, jo in enumerate(joints):
        img = cv2.line(img, (int(kps[jo[0], 0]),int(kps[jo[0], 1])),(int(kps[jo[1], 0]),int(kps[jo[1], 1])),color=color_joints[i],thickness=3)


    return img


def get_area_sampling_dist(kps, exp_weight=1.0, kp_subset=None):
    """
    This method ensures that "difficult" poses i.e. these with self-occlusions are sampled more often than common ones
    :return:
    """

    sampling_prob_area = np.asarray([], dtype=np.float)

    for kp in tqdm(
        kps, desc="Computing area sampling distribution for data loading."
    ):
        if kp_subset is None:
            chull = ConvexHull(kp[:, :2])
        else:
            chull = ConvexHull(kp[kp_subset, :2])
        sampling_prob_area = np.append(
            sampling_prob_area, np.power(1 / chull.area, exp_weight)
        )

    return sampling_prob_area


def _do_parallel_data_prefetch(func, Q, data, idx):
    # create dummy dataset instance

    # run prefetching
    res = func(data)
    Q.put([idx, res])
    Q.put("Done")


def parallel_data_prefetch(
    func: callable, data, n_proc, target_data_type="ndarray"
):
    if target_data_type not in ["ndarray", "list"]:
        raise ValueError(
            "Data, which is passed to parallel_data_prefetch has to be either of type list or ndarray."
        )
    if isinstance(data, np.ndarray) and target_data_type == "list":
        raise ValueError("list expected but function got ndarray.")
    elif isinstance(data, abc.Iterable):
        if isinstance(data, dict):
            print(
                f'WARNING:"data" argument passed to parallel_data_prefetch is a dict: Using only its values and disregarding keys.'
            )
            data = list(data.values())
        if target_data_type == "ndarray":
            data = np.asarray(data)
        else:
            data = list(data)
    else:
        raise TypeError(
            f"The data, that shall be processed parallel has to be either an np.ndarray or an Iterable, but is actually {type(data)}."
        )

    Q = mp.Queue(1000)

    # spawn processes
    if target_data_type == "ndarray":
        arguments = [
            [func, Q, part, i]
            for i, part in enumerate(np.array_split(data, n_proc))
        ]
    else:
        step = (
            int(len(data) / n_proc + 1)
            if len(data) % n_proc != 0
            else int(len(data) / n_proc)
        )
        arguments = [
            [func, Q, part, i]
            for i, part in enumerate(
                [data[i : i + step] for i in range(0, len(data), step)]
            )
        ]
    processes = []
    for i in range(n_proc):
        p = mp.Process(target=_do_parallel_data_prefetch, args=arguments[i])
        processes += [p]

    # start processes
    print(f"Start prefetching...")
    import time

    start = time.time()
    gather_res = [[] for _ in range(n_proc)]
    try:
        for p in processes:
            p.start()

        k = 0
        while k < n_proc:
            # get result
            res = Q.get()
            if res == "Done":
                k += 1
            else:
                gather_res[res[0]] = res[1]

    except Exception as e:
        print("Exception: ", e)
        for p in processes:
            p.terminate()

        raise e
    finally:
        for p in processes:
            p.join()
        print(f"Prefetching complete. [{time.time() - start} sec.]")

    if not isinstance(gather_res[0], np.ndarray):
        return np.concatenate([np.asarray(r) for r in gather_res], axis=0)

    # order outputs
    return np.concatenate(gather_res, axis=0)


def fig2data(fig, imsize):
    """

    :param fig: Matplotlib figure
    :param imsize:
    :return:
    """
    canvas = FigureCanvas(fig)

    ax = fig.gca()

    # ax.text(0.0, 0.0, "Test", fontsize=45)
    # ax.axis("off")
    canvas.draw()
    image = np.fromstring(canvas.tostring_rgb(), dtype="uint8")
    width, height = imsize
    image = image.reshape((int(height), int(width), -1))
    return image

def text_to_vid(vid, text, org:tuple, font_size=0.7, font_thickness=2):
    """

    :param vid:
    :param text:
    :param org: the image coordinates, where the text should be put
    :param font_size:
    :param font_thickness:
    :return:
    """

    for k in range(vid.shape[0]):
        vid[k] = cv2.putText(
            vid[k],
            text,
            org,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_size,
            (0, 0, 0),
            font_thickness,
        )

    return vid


def prepare_input(kp, device):
    data = kp[:, :-1].to(torch.float).to(device)
    target = kp[:, 1:].to(torch.float).to(device)
    return data, target


def slerp(val, low, high):
    omega = np.arccos(np.clip(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)), -1, 1))
    so = np.sin(omega)
    if so == 0:
        return (1.0-val) * low + val * high # L'Hopital's rule/LERP
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega) / so * high


class AverageNIterations(Average):

    def compute(self):
        if self.num_examples < 1:
            raise RuntimeError("{} must have at least one example before"
                                     " it can be computed.".format(self.__class__.__name__))

        return self.accumulator / self.num_examples

    def completed(self, engine, name):
        result = self.compute()
        if torch.is_tensor(result) and len(result.shape) == 0:
            result = result.item()
        engine.state.metrics[name] = result

    def update(self, output):
        self._check_output_type(output)

        self.accumulator = self._op(self.accumulator, output)
        if hasattr(output, 'shape'):
            self.num_examples += output.shape[0] if len(output.shape) > 1 else 1
        else:
            self.num_examples += 1

    def attach(self, engine, name, every=300):
        # add funcs to accumulate metrics over iterations
        engine.add_event_handler(Events.STARTED,self.started)
        engine.add_event_handler(Events.ITERATION_COMPLETED(every=every-1), self.completed, name)
        if not engine.has_event_handler(self.started, Events.ITERATION_STARTED):
            engine.add_event_handler(Events.ITERATION_STARTED(every=every), self.started)
        if not engine.has_event_handler(self.iteration_completed, Events.ITERATION_COMPLETED):
            engine.add_event_handler(Events.ITERATION_COMPLETED, self.iteration_completed)
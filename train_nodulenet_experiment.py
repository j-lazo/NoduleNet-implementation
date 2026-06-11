# Cell 2 — Paths

from pathlib import Path
import os, sys, shutil, subprocess, textwrap
import sys

# Cell — imports

import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch

from torch.utils.data import Dataset
from scipy.ndimage import label as cc_label
from tqdm import tqdm

import json
import warnings
from scipy.ndimage import zoom
from scipy.ndimage import rotate
from scipy.ndimage import label as cc_label

from torch.utils.tensorboard import SummaryWriter
from net.nodule_net import NoduleNet
from config import net_config, config

# WORKDIR = Path.cwd()
# REPO_DIR = WORKDIR / "NoduleNet"

# # Change these to your local paths
# DATA_BASE = Path("/path/to/LIDC_or_LUNA_data")

# RAW_MHD_DIR = DATA_BASE / "combined_mhd"          # all LUNA/LIDC .mhd files in one folder
# LIDC_XML_DIR = DATA_BASE / "LIDC-XML-only"        # LIDC XML annotation folder
# LUNG_MASK_DIR = DATA_BASE / "seg-lungs-LUNA16"   # lung masks from LUNA16
# PREPROCESSED_DIR = DATA_BASE / "preprocessed"    # output/input preprocessed data

# RESULTS_DIR = WORKDIR / "results_nodulenet"

# for p in [DATA_BASE, RESULTS_DIR]:
#     p.mkdir(parents=True, exist_ok=True)

# print("WORKDIR:", WORKDIR)
# print("REPO_DIR:", REPO_DIR)
# print("DATA_BASE:", DATA_BASE)


def create_image_mask_pairs(path_volumes, path_masks, path_ids_link_file):
    df_ids_link = pd.read_csv(path_ids_link_file)

    mhd_files = [f for f in os.listdir(path_volumes) if f.endswith(".mhd")]
    only_name_files = [f.replace(".mhd", "") for f in mhd_files]

    list_files_masks = os.listdir(path_masks)
    list_mask_nodules = [
        f for f in list_files_masks
        if "mask" in f
        and "contour" in f
        and "circle" not in f
        and "nodule" in f
        and (f.endswith(".nii.gz") or f.endswith(".nii"))
    ]

    list_number_masks = [int(s.split("_")[0]) for s in list_mask_nodules]

    print(len(list_mask_nodules), "Masks in", path_masks)
    print(len(only_name_files), "CT Volumes in", path_volumes)

    output_dict = {}

    for mhd_file in tqdm(only_name_files):
        sub_df_links = df_ids_link[df_ids_link["SeriesID"] == mhd_file]

        if len(sub_df_links) == 0:
            print("No mask link found for:", mhd_file)
            continue

        mask_id_num = sub_df_links["CID"].tolist()[0]

        if mask_id_num not in list_number_masks:
            print("Mask id not found:", mask_id_num, mhd_file)
            continue

        idx = list_number_masks.index(mask_id_num)

        path_mask = os.path.join(path_masks, list_mask_nodules[idx])
        path_image = os.path.join(path_volumes, mhd_file + ".mhd")

        output_dict[mhd_file] = {
            "path_image": path_image,
            "path_mask": path_mask,
            "cid": mask_id_num,
        }

    return output_dict


# Cell — helper functions

def read_sitk_zyx(path):
    """
    SimpleITK returns array as [z, y, x].
    """
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)
    spacing_xyz = img.GetSpacing()
    spacing_zyx = spacing_xyz[::-1]
    return arr, spacing_zyx, img


def make_instance_mask(binary_mask):
    """
    NoduleNet expects instance masks:
    0 = background
    1,2,3,... = different nodules
    """
    binary_mask = (binary_mask > 0).astype(np.uint8)
    inst_mask, num = cc_label(binary_mask)
    return inst_mask.astype(np.int32), num


def masks_to_bboxes_and_truth_masks(instance_mask, border=8):
    """
    Converts full 3D instance mask to NoduleNet-style boxes.

    Returns:
      bboxes: [N, 6] = [z, y, x, d, h, w]
      labels: [N]
      truth_masks: list of full-volume binary masks, one per nodule
    """
    bboxes = []
    labels = []
    truth_masks = []

    ids = [i for i in np.unique(instance_mask) if i != 0]

    for obj_id in ids:
        obj = instance_mask == obj_id
        zz, yy, xx = np.where(obj)

        if len(zz) == 0:
            continue

        zmin, zmax = zz.min(), zz.max() + 1
        ymin, ymax = yy.min(), yy.max() + 1
        xmin, xmax = xx.min(), xx.max() + 1

        cz = (zmin + zmax) / 2.0
        cy = (ymin + ymax) / 2.0
        cx = (xmin + xmax) / 2.0

        d = (zmax - zmin) + border
        h = (ymax - ymin) + border
        w = (xmax - xmin) + border

        bboxes.append([cz, cy, cx, d, h, w])
        labels.append(1)
        truth_masks.append(obj.astype(np.uint8))

    if len(bboxes) == 0:
        return (
            np.zeros((0, 6), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,) + instance_mask.shape, dtype=np.uint8),
        )

    return (
        np.asarray(bboxes, dtype=np.float32),
        np.asarray(labels, dtype=np.int32),
        np.asarray(truth_masks, dtype=np.uint8),
    )


def center_crop_or_pad_3d(image, mask, crop_size=(128, 128, 128), pad_value=170):
    """
    Simple full-volume center crop/pad.
    Useful for debugging, but not identical to NoduleNet's nodule-centered crop.
    """
    D, H, W = image.shape
    cd, ch, cw = crop_size

    out_img = np.full(crop_size, pad_value, dtype=image.dtype)
    out_msk = np.zeros(crop_size, dtype=mask.dtype)

    src_z0 = max((D - cd) // 2, 0)
    src_y0 = max((H - ch) // 2, 0)
    src_x0 = max((W - cw) // 2, 0)

    src_z1 = min(src_z0 + cd, D)
    src_y1 = min(src_y0 + ch, H)
    src_x1 = min(src_x0 + cw, W)

    dst_z0 = max((cd - D) // 2, 0)
    dst_y0 = max((ch - H) // 2, 0)
    dst_x0 = max((cw - W) // 2, 0)

    dz = src_z1 - src_z0
    dy = src_y1 - src_y0
    dx = src_x1 - src_x0

    out_img[dst_z0:dst_z0+dz, dst_y0:dst_y0+dy, dst_x0:dst_x0+dx] = image[src_z0:src_z1, src_y0:src_y1, src_x0:src_x1]
    out_msk[dst_z0:dst_z0+dz, dst_y0:dst_y0+dy, dst_x0:dst_x0+dx] = mask[src_z0:src_z1, src_y0:src_y1, src_x0:src_x1]

    return out_img, out_msk


class NoduleNetCrop:
    def __init__(self, crop_size=(128, 128, 128), bound_size=12, pad_value=170):
        self.crop_size = np.array(crop_size)
        self.bound_size = bound_size
        self.pad_value = pad_value

    def __call__(self, image, target, mask, do_scale=False, is_random=False):
        """
        image: [1, D, H, W]
        mask:  [D, H, W]
        target: [z, y, x, d] or [z, y, x, d, h, w]
        """
        mask = (mask > 0).astype(np.int32)

        if len(target) >= 6:
            target = np.array([target[0], target[1], target[2], max(target[3:6])], dtype=np.float32)
        else:
            target = np.array(target, dtype=np.float32)

        if do_scale and not is_random:
            radius_lim = [8.0, 120.0]
            scale_lim = [0.75, 1.25]

            scale_range = [
                min(max(radius_lim[0] / target[3], scale_lim[0]), 1),
                max(min(radius_lim[1] / target[3], scale_lim[1]), 1),
            ]
            scale = np.random.rand() * (scale_range[1] - scale_range[0]) + scale_range[0]
            crop_size = (self.crop_size.astype(float) / scale).astype(int)
        else:
            scale = 1.0
            crop_size = self.crop_size.copy()

        start = []

        for i in range(3):
            if not is_random:
                r = target[3] / 2
                s = np.floor(target[i] - r) + 1 - self.bound_size
                e = np.ceil(target[i] + r) + 1 + self.bound_size - crop_size[i]

                if s > e:
                    start_i = np.random.randint(int(e), int(s))
                else:
                    start_i = int(target[i] - crop_size[i] / 2 + np.random.randint(-self.bound_size // 2, self.bound_size // 2))
            else:
                dim = image.shape[i + 1]
                start_i = np.random.randint(0, max(1, dim - crop_size[i]))

            start.append(int(start_i))

        pad = [[0, 0]]
        for i in range(3):
            left_pad = max(0, -start[i])
            right_pad = max(0, start[i] + crop_size[i] - image.shape[i + 1])
            pad.append([left_pad, right_pad])

        z0, y0, x0 = [max(s, 0) for s in start]
        z1 = min(start[0] + crop_size[0], image.shape[1])
        y1 = min(start[1] + crop_size[1], image.shape[2])
        x1 = min(start[2] + crop_size[2], image.shape[3])

        crop_img = image[:, z0:z1, y0:y1, x0:x1]
        crop_msk = mask[z0:z1, y0:y1, x0:x1]

        crop_img = np.pad(crop_img, pad, mode="constant", constant_values=self.pad_value)
        crop_msk = np.pad(crop_msk, pad[1:], mode="constant", constant_values=0)

        if not is_random:
            for i in range(3):
                target[i] -= start[i]

        if do_scale and not is_random:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                crop_img = zoom(crop_img, [1, scale, scale, scale], order=1)
                crop_msk = zoom(crop_msk, [scale, scale, scale], order=0)

            diff = self.crop_size[0] - crop_img.shape[1]

            if diff < 0:
                crop_img = crop_img[:, :self.crop_size[0], :self.crop_size[1], :self.crop_size[2]]
                crop_msk = crop_msk[:self.crop_size[0], :self.crop_size[1], :self.crop_size[2]]
            elif diff > 0:
                pad2_img = [[0, 0], [0, diff], [0, diff], [0, diff]]
                pad2_msk = [[0, diff], [0, diff], [0, diff]]
                crop_img = np.pad(crop_img, pad2_img, mode="constant", constant_values=self.pad_value)
                crop_msk = np.pad(crop_msk, pad2_msk, mode="constant", constant_values=0)

            target[:4] *= scale

        crop_msk, _ = cc_label((crop_msk > 0.5).astype(np.int32))

        return crop_img, target, crop_msk.astype(np.int32)
    

def nodulenet_mask_augment(sample, target, mask, do_flip=True, do_rotate=True):
    """
    sample: [1, D, H, W]
    target: [z, y, x, d]
    mask:   [D, H, W]
    """
    mask = (mask > 0).astype(np.int32)

    if do_rotate:
        valid_rot = False
        counter = 0

        while not valid_rot:
            new_target = target.copy()
            angle = np.random.rand() * 180

            size = np.array(sample.shape[2:4]).astype(float)  # H, W
            rotmat = np.array([
                [np.cos(angle / 180 * np.pi), -np.sin(angle / 180 * np.pi)],
                [np.sin(angle / 180 * np.pi),  np.cos(angle / 180 * np.pi)],
            ])

            new_target[1:3] = np.dot(rotmat, target[1:3] - size / 2) + size / 2

            if np.all(new_target[:3] > target[3]) and np.all(new_target[:3] < np.array(sample.shape[1:4]) - new_target[3]):
                valid_rot = True
                target = new_target
                sample = rotate(sample, angle, axes=(2, 3), reshape=False, order=1)
                mask = rotate(mask, angle, axes=(1, 2), reshape=False, order=0)
            else:
                counter += 1
                if counter == 3:
                    break

    if do_flip:
        # Original NoduleNet does not randomly flip z; only y/x.
        flip_id = np.array([1, np.random.randint(2), np.random.randint(2)]) * 2 - 1

        sample = np.ascontiguousarray(
            sample[:, ::flip_id[0], ::flip_id[1], ::flip_id[2]]
        )
        mask = np.ascontiguousarray(
            mask[::flip_id[0], ::flip_id[1], ::flip_id[2]]
        )

        for ax in range(3):
            if flip_id[ax] == -1:
                target[ax] = sample.shape[ax + 1] - target[ax]

    mask, _ = cc_label((mask > 0.5).astype(np.int32))

    return sample, target, mask.astype(np.int32)


class MHDNiftiMaskNoduleNetDataset(Dataset):
    def __init__(
        self,
        pairs_dict,
        crop_size=(128, 128, 128),
        bbox_border=8,
        bound_size=12,
        pad_value=170,
        mode="train",
        augtype=None,
        r_rand_crop=0.0,
    ):
        self.pairs = pairs_dict
        self.case_ids = list(pairs_dict.keys())
        self.crop_size = crop_size
        self.bbox_border = bbox_border
        self.pad_value = pad_value
        self.mode = mode
        self.r_rand_crop = r_rand_crop

        self.augtype = augtype or {
            "flip": True,
            "rotate": True,
            "scale": True,
            "swap": False,
        }

        self.cropper = NoduleNetCrop(
            crop_size=crop_size,
            bound_size=bound_size,
            pad_value=pad_value,
        )

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx):
        case_id = self.case_ids[idx]
        item = self.pairs[case_id]

        image, _, _ = read_sitk_zyx(item["path_image"])
        mask, _, _ = read_sitk_zyx(item["path_mask"])

        image = image.astype(np.float32)
        mask = (mask > 0).astype(np.uint8)

        if image.shape != mask.shape:
            raise ValueError(f"{case_id}: image {image.shape}, mask {mask.shape}")

        
        #instance_mask, _ = make_instance_mask(mask)
        #instance_mask = mask.astype(np.int32)
        instance_mask = mask
        
        bboxes_full, _, _ = masks_to_bboxes_and_truth_masks(
            instance_mask,
            border=0,
        )

        if len(bboxes_full) > 0:
            # Use random positive crops only while training. Validation/test stay deterministic.
            if self.mode == "train":
                target = bboxes_full[np.random.randint(len(bboxes_full))]
            else:
                target = bboxes_full[0]
            is_random_crop = False
        else:
            target = np.array([np.nan, np.nan, np.nan, np.nan], dtype=np.float32)
            is_random_crop = True

        image = image[np.newaxis]  # [1, D, H, W]

        do_scale = self.augtype["scale"] and self.mode == "train" and not is_random_crop

        image, target, instance_mask = self.cropper(
            image,
            target,
            instance_mask,
            do_scale=do_scale,
            is_random=is_random_crop,
        )

        if self.mode == "train" and not is_random_crop:
            image, target, instance_mask = nodulenet_mask_augment(
                image,
                target,
                instance_mask,
                do_flip=self.augtype["flip"],
                do_rotate=self.augtype["rotate"],
            )

        image = (image.astype(np.float32) - 128.0) / 128.0

        truth_bboxes, truth_labels, truth_masks = masks_to_bboxes_and_truth_masks(
            instance_mask,
            border=self.bbox_border,
        )

        masks = instance_mask[np.newaxis].astype(np.float32)

        return [
            torch.from_numpy(image).float(),
            truth_bboxes.astype(np.float32),
            truth_labels.astype(np.int32),
            truth_masks.astype(np.uint8),
            masks,
            case_id,
        ]
        
        
# Cell — custom collate function

def nodulenet_collate(batch):
    inputs = torch.stack([b[0] for b in batch], dim=0)

    truth_bboxes = [np.asarray(b[1], dtype=np.float32) for b in batch]
    truth_labels = [np.asarray(b[2], dtype=np.int32) for b in batch]
    truth_masks  = [np.asarray(b[3], dtype=np.uint8) for b in batch]
    masks        = [np.asarray(b[4], dtype=np.float32) for b in batch]

    return inputs, truth_bboxes, truth_labels, truth_masks, masks




# -----------------------------
# Experiment script additions
# -----------------------------
import argparse
import datetime as _dt
import random
from typing import Dict, List, Tuple, Any

from torch.utils.data import DataLoader

try:
    import yaml
except ImportError:  # lightweight fallback; install pyyaml for nicer YAML output
    yaml = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_tuple3(value: str) -> Tuple[int, int, int]:
    parts = [int(x.strip()) for x in value.split(',')]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError('Expected three comma-separated integers, e.g. 128,128,128')
    return tuple(parts)


def nodulenet_collate(batch):
    inputs = torch.stack([b[0] for b in batch], dim=0)
    truth_bboxes = [np.asarray(b[1], dtype=np.float32) for b in batch]
    truth_labels = [np.asarray(b[2], dtype=np.int32) for b in batch]
    truth_masks  = [np.asarray(b[3], dtype=np.uint8) for b in batch]
    masks        = [np.asarray(b[4], dtype=np.float32) for b in batch]
    case_ids     = [b[5] if len(b) > 5 else str(i) for i, b in enumerate(batch)]
    return inputs, truth_bboxes, truth_labels, truth_masks, masks, case_ids


def split_pairs(pairs: Dict[str, dict], val_fraction: float, test_fraction: float, seed: int):
    if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1:
        raise ValueError('Require val_fraction >= 0, test_fraction >= 0, and val_fraction + test_fraction < 1')

    case_ids = list(pairs.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(case_ids)

    n_total = len(case_ids)
    n_test = int(round(n_total * test_fraction))
    n_val = int(round(n_total * val_fraction))

    test_ids = case_ids[:n_test]
    val_ids = case_ids[n_test:n_test + n_val]
    train_ids = case_ids[n_test + n_val:]

    return (
        {k: pairs[k] for k in train_ids},
        {k: pairs[k] for k in val_ids},
        {k: pairs[k] for k in test_ids},
        {'train': train_ids, 'val': val_ids, 'test': test_ids},
    )


def make_experiment_id(args) -> str:
    timestamp = _dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    crop = 'x'.join(map(str, args.crop_size))
    return (
        f'nodulenet_lr{args.lr:g}_bs{args.batch_size}_ep{args.epochs}'
        f'_crop{crop}_rcnn{args.epoch_rcnn}_mask{args.epoch_mask}_{timestamp}'
    )


def make_json_safe(obj):
    """Recursively convert objects to JSON/YAML/checkpoint-safe Python types."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(make_json_safe(k)): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, argparse.Namespace):
        return make_json_safe(vars(obj))
    return obj


def save_json(path: Path, data) -> None:
    with open(path, 'w') as f:
        json.dump(make_json_safe(data), f, indent=2)


def save_yaml(path: Path, data: dict) -> None:
    data = make_json_safe(data)
    with open(path, 'w') as f:
        if yaml is not None:
            yaml.safe_dump(data, f, sort_keys=False)
        else:
            # JSON is valid YAML 1.2, so this is still readable as YAML.
            json.dump(data, f, indent=2)


def torch_load_checkpoint(path, map_location='cpu'):
    """Load project checkpoints across PyTorch versions.

    PyTorch 2.6 changed torch.load's default to weights_only=True,
    which rejects metadata objects in normal training checkpoints.
    This script only uses this for checkpoints created by this training run.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # Older PyTorch versions do not have the weights_only argument.
        return torch.load(path, map_location=map_location)


def build_datasets(train_pairs, val_pairs, test_pairs, args):
    train_aug = {'flip': args.flip, 'rotate': args.rotate, 'scale': args.scale, 'swap': False}
    eval_aug = {'flip': False, 'rotate': False, 'scale': False, 'swap': False}

    common = dict(
        crop_size=args.crop_size,
        bbox_border=args.bbox_border,
        bound_size=args.bound_size,
        pad_value=args.pad_value,
    )

    train_dataset = MHDNiftiMaskNoduleNetDataset(train_pairs, mode='train', augtype=train_aug, **common)
    val_dataset = MHDNiftiMaskNoduleNetDataset(val_pairs, mode='val', augtype=eval_aug, **common)
    test_dataset = MHDNiftiMaskNoduleNetDataset(test_pairs, mode='test', augtype=eval_aug, **common)
    return train_dataset, val_dataset, test_dataset


def make_loader(dataset, batch_size, shuffle, num_workers):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=nodulenet_collate,
    )


def set_stage(net, epoch, epoch_rcnn=65, epoch_mask=80):
    net.use_rcnn = epoch >= epoch_rcnn
    net.use_mask = epoch >= epoch_mask


def cleanup_nodulenet_tensors(net):
    attrs = [
        'rpn_proposals', 'detections', 'total_loss', 'rpn_cls_loss', 'rpn_reg_loss',
        'rcnn_cls_loss', 'rcnn_reg_loss', 'mask_loss', 'rpn_logits_flat', 'rpn_deltas_flat',
        'rcnn_logits', 'rcnn_deltas', 'mask_probs', 'mask_targets', 'crop_boxes',
    ]
    for attr in attrs:
        if hasattr(net, attr):
            try:
                delattr(net, attr)
            except Exception:
                pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _safe_loss_attr(net, name):
    value = getattr(net, name, None)
    if value is None:
        return 0.0
    return float(value.detach().cpu().item())


def collect_losses(net, loss):
    return {
        'loss': float(loss.detach().cpu().item()),
        'rpn_cls': _safe_loss_attr(net, 'rpn_cls_loss'),
        'rpn_reg': _safe_loss_attr(net, 'rpn_reg_loss'),
        'rcnn_cls': _safe_loss_attr(net, 'rcnn_cls_loss'),
        'rcnn_reg': _safe_loss_attr(net, 'rcnn_reg_loss'),
        'mask_loss': _safe_loss_attr(net, 'mask_loss'),
    }


def average_logs(logs):
    if not logs:
        return {'loss': float('nan'), 'rpn_cls': float('nan'), 'rpn_reg': float('nan'), 'rcnn_cls': float('nan'), 'rcnn_reg': float('nan'), 'mask_loss': float('nan')}
    keys = logs[0].keys()
    return {k: float(np.nanmean([x[k] for x in logs])) for k in keys}


def train_one_epoch(net, loader, optimizer, epoch, device, writer=None):
    net.set_mode('train')
    logs = []
    pbar = tqdm(loader, desc=f'Train epoch {epoch}')

    for inputs, truth_bboxes, truth_labels, truth_masks, masks, case_ids in pbar:
        inputs = inputs.to(device, non_blocking=True)
        truth_bboxes = [np.asarray(x, dtype=np.float32) for x in truth_bboxes]
        truth_labels = [np.asarray(x, dtype=np.int32) for x in truth_labels]
        truth_masks  = [np.asarray(x, dtype=np.uint8) for x in truth_masks]

        net(inputs, truth_bboxes, truth_labels, truth_masks, masks)
        loss, rpn_stat, rcnn_stat, mask_stat = net.loss()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_log = collect_losses(net, loss)
        logs.append(batch_log)
        pbar.set_postfix({'loss': f"{batch_log['loss']:.4f}"})
        cleanup_nodulenet_tensors(net)

    epoch_log = average_logs(logs)
    if writer is not None:
        for k, v in epoch_log.items():
            writer.add_scalar(k, v, epoch)
    return epoch_log


@torch.no_grad()
def validate_one_epoch(net, loader, epoch, device, writer=None):
    net.set_mode('valid')
    logs = []
    pbar = tqdm(loader, desc=f'Val epoch {epoch}')

    for inputs, truth_bboxes, truth_labels, truth_masks, masks, case_ids in pbar:
        inputs = inputs.to(device, non_blocking=True)
        truth_bboxes = [np.asarray(x, dtype=np.float32) for x in truth_bboxes]
        truth_labels = [np.asarray(x, dtype=np.int32) for x in truth_labels]
        truth_masks  = [np.asarray(x, dtype=np.uint8) for x in truth_masks]

        net(inputs, truth_bboxes, truth_labels, truth_masks, masks)
        loss, rpn_stat, rcnn_stat, mask_stat = net.loss()

        batch_log = collect_losses(net, loss)
        logs.append(batch_log)
        pbar.set_postfix({'loss': f"{batch_log['loss']:.4f}", 'mask': f"{batch_log['mask_loss']:.4f}"})
        cleanup_nodulenet_tensors(net)

    epoch_log = average_logs(logs)
    if writer is not None:
        for k, v in epoch_log.items():
            writer.add_scalar(k, v, epoch)
    return epoch_log


def get_lr(args, epoch):
    if args.lr_schedule == 'step':
        if epoch <= args.epochs * 0.5:
            return args.lr
        if epoch <= args.epochs * 0.8:
            return args.lr * 0.1
        return args.lr * 0.01
    return args.lr


def save_checkpoint(path: Path, net, optimizer, epoch: int, history: List[dict], args, best_metric: float, split_ids: dict):
    state_dict = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    ckpt = {
        'epoch': epoch,
        'state_dict': state_dict,
        'optimizer': optimizer.state_dict(),
        'history': make_json_safe(history),
        'best_metric': float(best_metric),
        'args': make_json_safe(vars(args)),
        'split_ids': make_json_safe(split_ids),
    }
    torch.save(ckpt, path)


def is_improvement(current: float, best: float, mode: str, min_delta: float) -> bool:
    if np.isnan(current):
        return False
    if mode == 'min':
        return current < best - min_delta
    return current > best + min_delta


def dice_iou(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum(dtype=np.float64)
    pred_sum = pred.sum(dtype=np.float64)
    gt_sum = gt.sum(dtype=np.float64)
    union = np.logical_or(pred, gt).sum(dtype=np.float64)
    dsc = (2.0 * inter + eps) / (pred_sum + gt_sum + eps)
    iou = (inter + eps) / (union + eps)
    return float(dsc), float(iou)


def _tensor_to_mask_array(obj):
    if isinstance(obj, (list, tuple)) and len(obj) > 0:
        obj = obj[0]
    if not torch.is_tensor(obj):
        return None
    arr = obj.detach().float().cpu().numpy()
    # Common shapes: [B,C,D,H,W], [B,D,H,W], [C,D,H,W], [D,H,W]
    while arr.ndim > 5:
        arr = arr[0]
    if arr.ndim == 5:
        arr = arr[0]
        arr = arr[1] if arr.shape[0] > 1 else arr[0]
    elif arr.ndim == 4:
        # Ambiguous: if first dim looks like a channel dim, use foreground channel when present.
        arr = arr[1] if arr.shape[0] in (1, 2) and arr.shape[0] > 1 else arr[0]
    if arr.ndim != 3:
        return None
    return arr


def extract_pred_mask(net, threshold: float, fallback_shape: Tuple[int, int, int]):
    # NoduleNet forks expose different attributes during inference; try common mask outputs.
    for attr in ('mask_probs', 'mask_prob', 'mask_logits', 'masks', 'mask_preds'):
        if hasattr(net, attr):
            arr = _tensor_to_mask_array(getattr(net, attr))
            if arr is not None:
                if 'logit' in attr:
                    arr = 1.0 / (1.0 + np.exp(-arr))
                return (arr >= threshold).astype(np.uint8), attr
    return np.zeros(fallback_shape, dtype=np.uint8), 'no_mask_output_found_saved_empty_mask'


def save_mask_like(mask_zyx: np.ndarray, output_path: Path, reference_path: str = None):
    img = sitk.GetImageFromArray(mask_zyx.astype(np.uint8))
    if reference_path is not None:
        try:
            ref = sitk.ReadImage(reference_path)
            # Copy metadata only if the crop shape matches the original image shape.
            if tuple(sitk.GetArrayFromImage(ref).shape) == tuple(mask_zyx.shape):
                img.CopyInformation(ref)
        except Exception:
            pass
    sitk.WriteImage(img, str(output_path))


@torch.no_grad()
def predict_test_set(net, loader, test_pairs, out_dir: Path, device, threshold: float):
    net.set_mode('valid')
    pred_dir = out_dir / 'predictions'
    pred_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for inputs, truth_bboxes, truth_labels, truth_masks, masks, case_ids in tqdm(loader, desc='Predict test'):
        inputs = inputs.to(device, non_blocking=True)
        truth_bboxes = [np.asarray(x, dtype=np.float32) for x in truth_bboxes]
        truth_labels = [np.asarray(x, dtype=np.int32) for x in truth_labels]
        truth_masks  = [np.asarray(x, dtype=np.uint8) for x in truth_masks]

        net(inputs, truth_bboxes, truth_labels, truth_masks, masks)
        # Some implementations populate mask outputs during forward only after loss() has built targets.
        try:
            net.loss()
        except Exception:
            pass

        for b, case_id in enumerate(case_ids):
            gt = np.asarray(masks[b][0] > 0, dtype=np.uint8)
            pred, source = extract_pred_mask(net, threshold=threshold, fallback_shape=gt.shape)
            if pred.shape != gt.shape:
                # Keep metric valid even if a fork returns a ROI-sized mask.
                fixed = np.zeros_like(gt, dtype=np.uint8)
                common = tuple(min(a, b) for a, b in zip(fixed.shape, pred.shape))
                fixed[:common[0], :common[1], :common[2]] = pred[:common[0], :common[1], :common[2]]
                pred = fixed

            dsc, iou = dice_iou(pred, gt)
            out_path = pred_dir / f'{case_id}_pred.nii.gz'
            reference = test_pairs.get(case_id, {}).get('path_image')
            save_mask_like(pred, out_path, reference_path=reference)
            rows.append({
                'case_id': case_id,
                'prediction_file': str(out_path),
                'dsc': dsc,
                'iou': iou,
                'prediction_source': source,
            })
        cleanup_nodulenet_tensors(net)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'test_predictions_metrics.csv', index=False)
    return df


def parse_args():
    parser = argparse.ArgumentParser(description='Train NoduleNet with train/val/test split, experiment tracking, early stopping, and test predictions.')

    parser.add_argument('--working-dir', type=Path, required=True, help='Base working directory. Results go to working_dir/results/experiment_id')
    parser.add_argument('--path-volumes', type=Path, required=True, help='Directory containing .mhd CT volumes')
    parser.add_argument('--path-masks', type=Path, required=True, help='Directory containing nodule mask .nii/.nii.gz files')
    parser.add_argument('--path-ids-link-file', type=Path, required=True, help='CSV linking SeriesID to CID')

    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--lr-schedule', choices=['step', 'constant'], default='step')

    parser.add_argument('--val-fraction', type=float, default=0.2)
    parser.add_argument('--test-fraction', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--crop-size', type=parse_tuple3, default=(128, 128, 128))
    parser.add_argument('--bbox-border', type=int, default=8)
    parser.add_argument('--bound-size', type=int, default=12)
    parser.add_argument('--pad-value', type=int, default=170)

    parser.add_argument('--epoch-rcnn', type=int, default=15)
    parser.add_argument('--epoch-mask', type=int, default=18)
    parser.add_argument('--monitor', type=str, default='val_loss', help='Metric to monitor for best model/early stopping, e.g. val_loss or val_mask_loss')
    parser.add_argument('--monitor-mode', choices=['min', 'max'], default='min')
    parser.add_argument('--min-delta', type=float, default=0.0)
    parser.add_argument('--patience', type=int, default=12, help='Early stop after this many epochs without monitored metric improvement')
    parser.add_argument('--pred-threshold', type=float, default=0.5)

    parser.add_argument('--flip', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--rotate', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--scale', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--experiment-id', type=str, default=None, help='Optional custom experiment id. Default uses hyperparameters + timestamp.')
    parser.add_argument('--resume', type=Path, default=None, help='Optional checkpoint to resume from')

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exp_id = args.experiment_id or make_experiment_id(args)
    out_dir = args.working_dir / 'results' / exp_id
    model_dir = out_dir / 'models'
    tb_dir = out_dir / 'runs'
    model_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    pairs = create_image_mask_pairs(
        path_volumes=str(args.path_volumes),
        path_masks=str(args.path_masks),
        path_ids_link_file=str(args.path_ids_link_file),
    )
    train_pairs, val_pairs, test_pairs, split_ids = split_pairs(pairs, args.val_fraction, args.test_fraction, args.seed)

    train_dataset, val_dataset, test_dataset = build_datasets(train_pairs, val_pairs, test_pairs, args)
    train_loader = make_loader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = make_loader(val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = make_loader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    metadata = {
        'experiment_id': exp_id,
        'created_at': _dt.datetime.now().isoformat(timespec='seconds'),
        'device': str(device),
        'cuda_available': torch.cuda.is_available(),
        'torch_version': torch.__version__,
        'paths': {
            'working_dir': str(args.working_dir),
            'out_dir': str(out_dir),
            'path_volumes': str(args.path_volumes),
            'path_masks': str(args.path_masks),
            'path_ids_link_file': str(args.path_ids_link_file),
        },
        'hyperparameters': vars(args),
        'split_counts': {'train': len(train_dataset), 'val': len(val_dataset), 'test': len(test_dataset)},
        'split_ids': split_ids,
    }
    save_yaml(out_dir / 'config.yaml', metadata)
    save_json(out_dir / 'splits.json', split_ids)

    net = NoduleNet(net_config).to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    start_epoch = 1
    history = []
    initial_best = float('inf') if args.monitor_mode == 'min' else -float('inf')
    best_metric = initial_best
    epochs_without_improvement = 0

    if args.resume is not None:
        ckpt = torch_load_checkpoint(args.resume, map_location='cpu')
        net.load_state_dict(ckpt['state_dict'])
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = int(ckpt.get('epoch', 0)) + 1
        history = ckpt.get('history', [])
        best_metric = float(ckpt.get('best_metric', best_metric))

    train_writer = SummaryWriter(str(tb_dir / 'train'))
    val_writer = SummaryWriter(str(tb_dir / 'val'))

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            set_stage(net, epoch, epoch_rcnn=args.epoch_rcnn, epoch_mask=args.epoch_mask)
            lr = get_lr(args, epoch)
            for group in optimizer.param_groups:
                group['lr'] = lr

            print(f"\nEpoch {epoch}/{args.epochs} | lr={lr:.6g} | use_rcnn={net.use_rcnn} | use_mask={net.use_mask}")
            train_log = train_one_epoch(net, train_loader, optimizer, epoch, device, writer=train_writer)
            val_log = validate_one_epoch(net, val_loader, epoch, device, writer=val_writer)

            row = {
                'epoch': epoch,
                'lr': lr,
                'use_rcnn': bool(net.use_rcnn),
                'use_mask': bool(net.use_mask),
                **{f'train_{k}': v for k, v in train_log.items()},
                **{f'val_{k}': v for k, v in val_log.items()},
            }
            history.append(row)
            save_json(out_dir / 'history.json', history)

            current_metric = row.get(args.monitor)
            if current_metric is None:
                raise KeyError(f"Monitor metric {args.monitor!r} not found. Available keys: {sorted(row.keys())}")

            improved = is_improvement(current_metric, best_metric, args.monitor_mode, args.min_delta)
            if improved:
                best_metric = float(current_metric)
                epochs_without_improvement = 0
                save_checkpoint(model_dir / 'best.ckpt', net, optimizer, epoch, history, args, best_metric, split_ids)
                save_checkpoint(model_dir / f'best_epoch_{epoch:03d}.ckpt', net, optimizer, epoch, history, args, best_metric, split_ids)
                print(f"Saved improved model: {args.monitor}={best_metric:.6f}")
            else:
                epochs_without_improvement += 1
                print(f"No improvement for {epochs_without_improvement}/{args.patience} epochs. Best {args.monitor}={best_metric:.6f}")

            if epochs_without_improvement >= args.patience:
                print('Early stopping triggered.')
                break

        final_epoch = history[-1]['epoch'] if history else 0
        save_checkpoint(model_dir / 'final.ckpt', net, optimizer, final_epoch, history, args, best_metric, split_ids)

    finally:
        train_writer.close()
        val_writer.close()

    # Evaluate/predict with the overall best model if available; otherwise use final state.
    best_path = model_dir / 'best.ckpt'
    if best_path.exists():
        ckpt = torch_load_checkpoint(best_path, map_location='cpu')
        net.load_state_dict(ckpt['state_dict'])
        net.to(device)

    test_df = predict_test_set(net, test_loader, test_pairs, out_dir, device, threshold=args.pred_threshold)
    print(f"Training finished. Results: {out_dir}")
    print(test_df.describe(include='all'))


if __name__ == '__main__':
    main()

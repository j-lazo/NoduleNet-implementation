# Cell 2 — Paths

from pathlib import Path
import os, sys, shutil, subprocess, textwrap
import sys
print("Python:", sys.executable)
# Cell 4 — Check environment

import torch, platform
print("Python:", platform.python_version())
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    
import sys
from matplotlib import pyplot as plt

from config import config
from dataset.mask_reader import MaskReader
from dataset.bbox_reader import BboxReader

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
from scipy.ndimage.interpolation import rotate
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
            target = bboxes_full[np.random.randint(len(bboxes_full))]
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
        ]
        
        
# Cell — custom collate function

def nodulenet_collate(batch):
    inputs = torch.stack([b[0] for b in batch], dim=0)

    truth_bboxes = [np.asarray(b[1], dtype=np.float32) for b in batch]
    truth_labels = [np.asarray(b[2], dtype=np.int32) for b in batch]
    truth_masks  = [np.asarray(b[3], dtype=np.uint8) for b in batch]
    masks        = [np.asarray(b[4], dtype=np.float32) for b in batch]

    return inputs, truth_bboxes, truth_labels, truth_masks, masks


# Cell — create dataset and loader

from torch.utils.data import DataLoader

path_dataset = "/nobackup/proj/disk/naiss2025-6-383/personal/jorgelaz/datasets/LUNA_dataset/"
print(os.path.isdir(path_dataset), path_dataset)
path_vols = os.path.join(path_dataset, "CT_volumes")
print(os.path.isdir(path_vols), path_vols)
path_masks = os.path.join(path_dataset, "masks_nodules/nifti_data")
print(os.path.isdir(path_masks), path_masks)
path_df_links = os.path.join(path_dataset, "LUNA16_metadata_split_offical.csv")
print(os.path.isfile(path_df_links), path_df_links)
BATCH_SIZE = 16
NUM_WORKERS = 8


pairs = create_image_mask_pairs(
    path_volumes=path_vols,
    path_masks=path_masks,
    path_ids_link_file=path_df_links,)


dataset = MHDNiftiMaskNoduleNetDataset(
    pairs_dict=pairs,
    crop_size=(128, 128, 128),
    bbox_border=8,
    bound_size=12,
    pad_value=170,
    mode="train",
    augtype={
        "flip": True,
        "rotate": True,
        "scale": True,
        "swap": False,
    },
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    collate_fn=nodulenet_collate,
)

case_ids = list(pairs.keys())
np.random.seed(42)
np.random.shuffle(case_ids)

val_fraction = 0.2
n_val = int(len(case_ids) * val_fraction)

val_ids = case_ids[:n_val]
train_ids = case_ids[n_val:]

train_pairs = {k: pairs[k] for k in train_ids}
val_pairs = {k: pairs[k] for k in val_ids}

train_dataset = MHDNiftiMaskNoduleNetDataset(
    train_pairs,
    crop_size=(128, 128, 128),
    bbox_border=8,
    bound_size=12,
    pad_value=170,
    mode="train",
    augtype={
        "flip": True,
        "rotate": True,
        "scale": True,
        "swap": False,
    },
)

val_dataset = MHDNiftiMaskNoduleNetDataset(
    val_pairs,
    crop_size=(128, 128, 128),
    bbox_border=8,
    bound_size=12,
    pad_value=170,
    mode="val",
    augtype={
        "flip": False,
        "rotate": False,
        "scale": False,
        "swap": False,
    },
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,        # start with 1 for debugging
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    collate_fn=nodulenet_collate,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    collate_fn=nodulenet_collate,
)

print("Train cases:", len(train_dataset))
print("Val cases:", len(val_dataset))


# Cell — initialize model and optimizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = NoduleNet(net_config)
net = net.to(device)

optimizer = torch.optim.SGD(
    net.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=1e-4,
)

# Cell — helper functions

def set_stage(net, epoch, epoch_rcnn=65, epoch_mask=80):
    """
    Same logic as original train.py:
    - RPN first
    - RCNN after epoch_rcnn
    - mask branch after epoch_mask
    """
    net.use_rcnn = epoch >= epoch_rcnn
    net.use_mask = epoch >= epoch_mask


def cleanup_nodulenet_tensors(net):
    """
    Original train.py manually deletes these to reduce GPU memory.
    Keep try/except because not every branch exists every epoch.
    """
    attrs = [
        "rpn_proposals",
        "detections",
        "total_loss",
        "rpn_cls_loss",
        "rpn_reg_loss",
        "rcnn_cls_loss",
        "rcnn_reg_loss",
        "mask_loss",
        "rpn_logits_flat",
        "rpn_deltas_flat",
        "rcnn_logits",
        "rcnn_deltas",
        "mask_probs",
        "mask_targets",
        "crop_boxes",
    ]

    for attr in attrs:
        if hasattr(net, attr):
            try:
                delattr(net, attr)
            except Exception:
                pass

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def collect_losses(net, loss):
    return {
        "loss": float(loss.detach().cpu().item()),
        "rpn_cls": float(net.rpn_cls_loss.detach().cpu().item()),
        "rpn_reg": float(net.rpn_reg_loss.detach().cpu().item()),
        "rcnn_cls": float(net.rcnn_cls_loss.detach().cpu().item()),
        "rcnn_reg": float(net.rcnn_reg_loss.detach().cpu().item()),
        "mask_loss": float(net.mask_loss.detach().cpu().item()),
    }


def average_logs(logs):
    keys = logs[0].keys()
    return {k: float(np.mean([x[k] for x in logs])) for k in keys}


# Cell — train one epoch

def train_one_epoch(net, loader, optimizer, epoch, writer=None):
    net.set_mode("train")

    logs = []
    pbar = tqdm(loader, desc=f"Train epoch {epoch}")

    for inputs, truth_bboxes, truth_labels, truth_masks, masks in pbar:
        inputs = inputs.to(device, non_blocking=True)

        truth_bboxes = [np.asarray(x, dtype=np.float32) for x in truth_bboxes]
        truth_labels = [np.asarray(x, dtype=np.int32) for x in truth_labels]
        truth_masks  = [np.asarray(x, dtype=np.uint8) for x in truth_masks]

        net(inputs, truth_bboxes, truth_labels, truth_masks, masks)

        loss, rpn_stat, rcnn_stat, mask_stat = net.loss()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_log = collect_losses(net, loss)
        logs.append(batch_log)

        pbar.set_postfix({"loss": f"{batch_log['loss']:.4f}"})

        cleanup_nodulenet_tensors(net)

    return average_logs(logs)


# Cell — validate one epoch

@torch.no_grad()
def validate_one_epoch(net, loader, epoch, writer=None):
    net.set_mode("valid")

    logs = []
    pbar = tqdm(loader, desc=f"Val epoch {epoch}")

    for inputs, truth_bboxes, truth_labels, truth_masks, masks in pbar:
        inputs = inputs.to(device, non_blocking=True)

        truth_bboxes = [np.asarray(x, dtype=np.float32) for x in truth_bboxes]
        truth_labels = [np.asarray(x, dtype=np.int32) for x in truth_labels]
        truth_masks  = [np.asarray(x, dtype=np.uint8) for x in truth_masks]

        net(inputs, truth_bboxes, truth_labels, truth_masks, masks)

        loss, rpn_stat, rcnn_stat, mask_stat = net.loss()

        batch_log = collect_losses(net, loss)
        logs.append(batch_log)

        pbar.set_postfix({
            "loss": f"{batch_log['loss']:.4f}",
            "rpn": f"{batch_log['rpn_cls'] + batch_log['rpn_reg']:.4f}",
            "mask": f"{batch_log['mask_loss']:.4f}",
        })

        cleanup_nodulenet_tensors(net)

    epoch_log = average_logs(logs)

    if writer is not None:
        for k, v in epoch_log.items():
            writer.add_scalar(k, v, epoch)

    return epoch_log



# Cell — initialize model and optimizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = NoduleNet(net_config)
net = net.to(device)

optimizer = torch.optim.SGD(
    net.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=1e-4,
)

OUT_DIR = Path("/nobackup/proj/disk/naiss2025-6-383/personal/jorgelaz/projects/repos/NoduleNet/results/exp2")
MODEL_DIR = OUT_DIR / "model"
TB_DIR = OUT_DIR / "runs"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
TB_DIR.mkdir(parents=True, exist_ok=True)

train_writer = SummaryWriter(str(TB_DIR / "train"))
val_writer = SummaryWriter(str(TB_DIR / "val"))

# Cell — full training loop

EPOCHS = 3
EPOCH_RCNN = 2
EPOCH_MASK = 3
SAVE_EVERY = 1

best_val_loss = float("inf")
history = []

for epoch in range(1, EPOCHS + 1):
    set_stage(net, epoch, epoch_rcnn=EPOCH_RCNN, epoch_mask=EPOCH_MASK)

    # Original repo applies LR schedule only for SGD.
    if isinstance(optimizer, torch.optim.SGD):
        if epoch <= EPOCHS * 0.5:
            lr = 0.01
        elif epoch <= EPOCHS * 0.8:
            lr = 0.001
        else:
            lr = 0.0001

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
    else:
        lr = optimizer.param_groups[0]["lr"]

    print(
        f"\nEpoch {epoch}/{EPOCHS} | "
        f"lr={lr:.6f} | "
        f"use_rcnn={net.use_rcnn} | "
        f"use_mask={net.use_mask}"
    )

    train_log = train_one_epoch(
        net,
        train_loader,
        optimizer,
        epoch,
        writer=train_writer,
    )

    val_log = validate_one_epoch(
        net,
        val_loader,
        epoch,
        writer=val_writer,
    )

    row = {
        "epoch": epoch,
        "lr": lr,
        "use_rcnn": bool(net.use_rcnn),
        "use_mask": bool(net.use_mask),
        **{f"train_{k}": v for k, v in train_log.items()},
        **{f"val_{k}": v for k, v in val_log.items()},
    }

    history.append(row)

    with open(OUT_DIR / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Save last checkpoint
    state_dict = {k: v.cpu() for k, v in net.state_dict().items()}

    last_ckpt = {
        "epoch": epoch,
        "out_dir": str(OUT_DIR),
        "state_dict": state_dict,
        "optimizer": optimizer.state_dict(),
        "history": history,
    }

    torch.save(last_ckpt, MODEL_DIR / "last.ckpt")

    # Save best checkpoint
    if val_log["loss"] < best_val_loss:
        best_val_loss = val_log["loss"]
        torch.save(last_ckpt, MODEL_DIR / "best.ckpt")
        print(f"Saved best checkpoint: val_loss={best_val_loss:.4f}")

    # Save periodic checkpoint
    if epoch % SAVE_EVERY == 0:
        
        torch.save(last_ckpt, MODEL_DIR / f"{epoch:03d}.ckpt")

print("Training finished.")
train_writer.close()
val_writer.close()


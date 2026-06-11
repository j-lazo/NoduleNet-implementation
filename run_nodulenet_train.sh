#!/bin/bash
#SBATCH --job-name=nodulenet
#SBATCH -A naiss2026-3-377-gpu
#SBATCH -p gpu
#SBATCH -t 71:00:00
#SBATCH --gpus=1
#SBATCH --output=.slurm_logs/%x-%j.log
#SBATCH --error=.slurm_logs/%x-%j.log

cd /nobackup/proj/disk/naiss2025-6-383/personal/jorgelaz/projects/repos/NoduleNet

nvidia-smi 
apptainer exec --nv \
  --bind /nobackup/proj/disk/naiss2025-6-383/personal/jorgelaz:/mnt \
  project.sif \
  python train_nodulenet_experiment.py \
  --working-dir /mnt/projects/repos/NoduleNet \
  --path-volumes /mnt/datasets/LUNA_dataset/CT_volumes \
  --path-masks /mnt/datasets/LUNA_dataset/masks_nodules/nifti_data \
  --path-ids-link-file /mnt/datasets/LUNA_dataset/LUNA16_metadata_split_offical.csv \
  --epochs 200 \
  --epoch-rcnn 65 \
  --epoch-mask 80 \
  --batch-size 16 \
  --num-workers 8 \
  --lr 0.01 \
  --val-fraction 0.2 \
  --test-fraction 0.1 \
  --patience 12 \
  --monitor val_loss \
  --monitor-mode min

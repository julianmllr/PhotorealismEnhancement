#!/bin/bash -l
#SBATCH --gres=gpu:a40:1
#SBATCH --time=24:00:00
#SBATCH --job-name=setup_epe

module load python
conda activate epe5

vault=/home/vault/b204dc/b204dc12
dataset=${vault}/datasets/kitti_skipped/
phase=train
real_name=KITTI360

# python ${vault}/scripts/csvlist.py ${dataset}/images data/Pfd/${phase}.csv
# python epe/dataset/generate_fake_gbuffers.py GTA data/Pfd/${phase}.csv --out_dir ${dataset}/fake_gbuffer
python ${vault}/scripts/csvlist.py ${dataset}/images data/${real_name}/files.csv

# python epe/matching/feature_based/collect_crops.py PfD data/Pfd/train.csv 		# creates crop_PfD.csv, crop_Pfd.npz
python epe/matching/feature_based/collect_crops.py ${real_name} data/${real_name}/files.csv	# creates crop_Cityscapes.csv, crop_Cityscapes.npz
python epe/matching/feature_based/find_knn.py crop_PfD.npz crop_${real_name}.npz knn_PfD-${real_name}.npz -k 10
python epe/matching/filter.py knn_PfD-${real_name}.npz crop_PfD.csv crop_${real_name}.csv 1.0 matched_crops_PfD-${real_name}.csv
python epe/matching/compute_weights.py matched_crops_PfD-${real_name}.csv 526 957 crop_weights_PfD-${real_name}.npz
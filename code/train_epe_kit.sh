#!/bin/bash -l
#SBATCH --gres=gpu:a100:1 -C a100_80
#SBATCH --time=24:00:00
#SBATCH --job-name=train_epe_kit_uncropped

module load python
conda activate epe4

vault=/home/vault/b204dc/b204dc12

config=./config/train_pfd2kit.yaml
origin_fake=${vault}/datasets/city_gta_3500/6ch_half/gta/
origin_real=${vault}/datasets/kitti_skipped/uncropped/
dataset_fake=${TMPDIR}/gta
dataset_real=${TMPDIR}/kitti
dataset_val=${TMPDIR}/val

mkdir ${dataset_fake}
mkdir ${dataset_real}
mkdir ${dataset_val}

unzip ${origin_fake}/images.zip -d ${dataset_fake}
unzip ${origin_fake}/labels.zip -d ${dataset_fake}
unzip ${origin_fake}/robust_labels.zip -d ${dataset_fake}
unzip ${origin_fake}/npz_images.zip -d ${dataset_fake}

unzip ${origin_real}/train_uncropped.zip -d ${dataset_real}
mv ${dataset_real}/trainB ${dataset_real}/images
unzip ${origin_real}/robust_labels.zip -d ${dataset_real}

unzip ${vault}/datasets/val/6ch_half/gta/images.zip -d ${dataset_val}
unzip ${vault}/datasets/val/6ch_half/gta/labels.zip -d ${dataset_val}
unzip ${vault}/datasets/val/6ch_half/gta/robust_labels.zip -d ${dataset_val}
unzip ${vault}/datasets/val/6ch_half/gta/npz_images.zip -d ${dataset_val}

real_name=KITTI360
python ${vault}/scripts/csvlist.py ${dataset_fake}/images data/Pfd_kit/train.csv
# python epe/dataset/generate_fake_gbuffers.py GTA data/Pfd/${phase}.csv --out_dir ${dataset}/fake_gbuffer
python ${vault}/scripts/csvlist.py ${dataset_real}/images data/${real_name}/files.csv

python epe/matching/feature_based/collect_crops.py PfD data/Pfd_kit/train.csv 		# creates crop_PfD.csv, crop_Pfd.npz
python epe/matching/feature_based/collect_crops.py ${real_name} data/${real_name}/files.csv	# creates crop_Cityscapes.csv, crop_Cityscapes.npz
python epe/matching/feature_based/find_knn.py crop_PfD.npz crop_${real_name}.npz data/matches/knn_PfD-${real_name}.npz -k 10
python epe/matching/filter.py data/matches/knn_PfD-${real_name}.npz crop_PfD.csv crop_${real_name}.csv 1.0 data/matches/matched_crops_PfD-${real_name}.csv
python epe/matching/compute_weights.py data/matches/matched_crops_PfD-${real_name}.csv 526 957 data/matches/crop_weights_PfD-${real_name}.npz

gbuffer_folder=npz_images
python ${vault}/scripts/txtlist.py ${dataset_fake}/images ${dataset_fake}/robust_labels --gbuffer_folder ${dataset_fake}/${gbuffer_folder} --gt_folder ${dataset_fake}/labels data/Pfd_kit/train.txt
python ${vault}/scripts/txtlist.py ${dataset_real}/images ${dataset_real}/robust_labels data/${real_name}/files.txt
python ${vault}/scripts/txtlist.py ${dataset_val}/images ${dataset_val}/robust_labels --gbuffer_folder ${dataset_val}/${gbuffer_folder} --gt_folder ${dataset_val}/labels data/Pfd_kit/val.txt

python epe/EPEExperiment.py train ${config} --log=info
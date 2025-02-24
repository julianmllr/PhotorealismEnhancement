#!/bin/bash -l
#SBATCH --gres=gpu:a100:1 -C a100_80
#SBATCH --time=24:00:00
#SBATCH --job-name=train_epe

module load python
conda activate epe4

vault=/home/vault/b204dc/b204dc12

config=./config/train_pfd2cs.yaml
dataset=${vault}/datasets/city_gta_3500/6ch_half/
dataset_fake=${TMPDIR}/gta
dataset_real=${TMPDIR}/city
dataset_val=${TMPDIR}/val

mkdir ${dataset_fake}
mkdir ${dataset_real}
mkdir ${dataset_val}

gbuffer_folder=fake_gbuffer
unzip ${dataset}/gta/images.zip -d ${dataset_fake}
unzip ${dataset}/gta/labels.zip -d ${dataset_fake}
unzip ${dataset}/gta/robust_labels.zip -d ${dataset_fake}
unzip ${dataset}/gta/${gbuffer_folder}.zip -d ${dataset_fake} ### AAAAAACHTUUUUUNG!!!

unzip ${dataset}/city/images.zip -d ${dataset_real}
unzip ${dataset}/city/robust_labels.zip -d ${dataset_real}

unzip ${vault}/datasets/val/6ch_half/gta/images.zip -d ${dataset_val}
unzip ${vault}/datasets/val/6ch_half/gta/labels.zip -d ${dataset_val}
unzip ${vault}/datasets/val/6ch_half/gta/robust_labels.zip -d ${dataset_val}
unzip ${vault}/datasets/val/6ch_half/gta/${gbuffer_folder}.zip -d ${dataset_val}

real_name=Cityscapes
python ${vault}/scripts/csvlist.py ${dataset_fake}/images data/Pfd/train.csv
# python epe/dataset/generate_fake_gbuffers.py GTA data/Pfd/${phase}.csv --out_dir ${dataset}/fake_gbuffer
python ${vault}/scripts/csvlist.py ${dataset_real}/images data/${real_name}/files.csv

python epe/matching/feature_based/collect_crops.py PfD data/Pfd/train.csv 		# creates crop_PfD.csv, crop_Pfd.npz
python epe/matching/feature_based/collect_crops.py ${real_name} data/${real_name}/files.csv	# creates crop_Cityscapes.csv, crop_Cityscapes.npz
python epe/matching/feature_based/find_knn.py crop_PfD.npz crop_${real_name}.npz data/matches/knn_PfD-${real_name}.npz -k 10
python epe/matching/filter.py data/matches/knn_PfD-${real_name}.npz crop_PfD.csv crop_${real_name}.csv 1.0 data/matches/matched_crops_PfD-${real_name}.csv
python epe/matching/compute_weights.py data/matches/matched_crops_PfD-${real_name}.csv 526 957 data/matches/crop_weights_PfD-${real_name}.npz

python ${vault}/scripts/txtlist.py ${dataset_fake}/images ${dataset_fake}/robust_labels --gbuffer_folder ${dataset_fake}/${gbuffer_folder} --gt_folder ${dataset_fake}/labels data/Pfd/train.txt
python ${vault}/scripts/txtlist.py ${dataset_real}/images ${dataset_real}/robust_labels data/Cityscapes/files.txt
python ${vault}/scripts/txtlist.py ${dataset_val}/images ${dataset_val}/robust_labels --gbuffer_folder ${dataset_val}/${gbuffer_folder} --gt_folder ${dataset_val}/labels data/Pfd/val.txt

python epe/EPEExperiment.py train ${config} --log=info
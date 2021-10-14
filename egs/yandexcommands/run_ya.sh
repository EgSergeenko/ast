#!/bin/bash
#SBATCH -p gpu
#SBATCH -x sls-titan-[0-2]
##SBATCH -p sm
#SBATCH --gres=gpu:4
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --mem=48000
#SBATCH --job-name="ast-sc"
#SBATCH --output=./log_%j.txt

set -x
# comment this line if not running on sls cluster
#. /data/sls/scratch/share-201907/slstoolchainrc
source /home/es_egor_sergeenko/ast/venvast/bin/activate
export TORCH_HOME=/home/es_egor_sergeenko/ast/pretrained_models

model=ast
dataset=yandexcommands
imagenetpretrain=True
audiosetpretrain=True
bal=none
lr=0.00025
epoch=30
freqm=48
timem=48
mixup=0.0
batch_size=24
fstride=10
tstride=10
tr_data=/home/es_egor_sergeenko/ast/egs/yandexcommands/data/datafiles/train_data.json
val_data=/home/es_egor_sergeenko/ast/egs/yandexcommands/data/datafiles/val_data.json
noise_data=/home/es_egor_sergeenko/ast/egs/yandexcommands/data/datafiles/noise_data.json
#eval_data=./data/datafiles/speechcommand_eval_data.json
exp_dir=/home/es_egor_sergeenko/ast/egs/yandexcommands/exp/test-${dataset}-f$fstride-t$tstride-p$imagenetpretrain-b$batch_size-lr${lr}
label_csv=/home/es_egor_sergeenko/ast/egs/yandexcommands/data/class_labels_indices.csv
n_class=38

if [ -d $exp_dir ]; then
  echo 'exp exist'
  exit
fi
mkdir -p $exp_dir

CUDA_CACHE_DISABLE=1 python -W ignore /home/es_egor_sergeenko/ast/src/run.py --model ${model} --dataset ${dataset} \
--data-train ${tr_data} --data-val ${val_data} --data-noise ${noise_data} --exp-dir $exp_dir \
--label-csv ${label_csv} --n_class ${n_class} \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model True \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--tstride $tstride --fstride $fstride --imagenet_pretrain $imagenetpretrain --audioset_pretrain $audiosetpretrain

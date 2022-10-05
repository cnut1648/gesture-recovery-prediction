#! /bin/bash
#SBATCH --gres=gpu:1080:1
#SBATCH -t 2048
#SBATCH -n 1
#SBATCH --export=ALL,LD_LIBRARY_PATH='/usr/local/cuda-10.1/lib64/'
#SBATCH -o logs/slurm/run::%j.out

/home/jiashu/.conda/envs/meta/bin/python main.py -m \
    module=baseline \
    module/model=AttnModel \
    data.batch_size=8 \
    data=MC \
    data.data_dir="/home/jiashu/seq/processed/USC+Gronau" \
    callback=wandb \
    logger.wandb.project="seq-USC+Gronau" \
    +hsearch=lr-auc-attn
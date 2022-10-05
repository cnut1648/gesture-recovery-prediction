#!/bin/bash
#SBATCH --nodelist=ink-lucy
#SBATCH --gres=gpu:1080:1
#SBATCH -t 2048
#SBATCH -n 1
#SBATCH --export=ALL,LD_LIBRARY_PATH='/usr/local/cuda-10.1/lib64/'
#SBATCH -o logs/slurm/run::%j.out

# /home/jiashu/.conda/envs/meta/bin/python main.py \
#     module=baseline \
#     module/model=AttnModel \
#     data.batch_size=8 \
#     data.column_name=ave_th_2_sides \
#     data.target_as_clf=True \
#     data=DART_2022 \
#     callback=wandb

# /home/jiashu/.conda/envs/meta/bin/python main.py \
#     module=baseline \
#     module/model=AttnModel \
#     data.batch_size=8 \
#     data.column_name=ave_tr_2_sides \
#     data.target_as_clf=True \
#     data=DART_2022 \
#     callback=wandb

/home/jiashu/.conda/envs/meta/bin/python main.py \
    module=baseline \
    module/model=AttnModel \
    data.batch_size=8 \
    data.column_name=ave_e_2_sides \
    data.target_as_clf=True \
    data=DART_2022 \
    callback=wandb
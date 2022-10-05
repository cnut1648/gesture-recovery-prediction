#! /bin/bash

# /home/jiashu/.conda/envs/meta/bin/python main.py \
#     module=baseline_reg \
#     module/model=AttnModel \
#     data.batch_size=8 \
#     data.column_name=ave_th_2_sides \
#     data=DART_2022 \
#     callback=wandb_reg

# /home/jiashu/.conda/envs/meta/bin/python main.py \
#     module=baseline_reg \
#     module/model=AttnModel \
#     data.batch_size=8 \
#     data.column_name=ave_tr_2_sides \
#     data=DART_2022 \
#     callback=wandb_reg

/home/jiashu/.conda/envs/meta/bin/python main.py \
    module=baseline_reg \
    module/model=AttnModel \
    data.batch_size=8 \
    data.column_name=ave_e_2_sides \
    data=DART_2022 \
    callback=wandb_reg
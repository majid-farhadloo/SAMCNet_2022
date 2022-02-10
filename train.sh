#!/bin/bash

# self_att_neighbor_att
# python main.py --dataset diseases \
#                --exp_name self_att_neighbor_att_diseases_2 \
#                --self_neighbor True \
#                --neighbor True \
#                --use_pe False \
#                --num_heads 0 \
#                --batch_size 7 \
#                --lr 0.001 \
#                --momentum 0.9 \
#                --num_points 1024 \
#                --dropout 0.5 \
#                --use_sgd False \
#                --emb_dims 1024 \
#                --epochs 200

# python main.py --dataset margin \
#                --exp_name self_att_neighbor_att_margin_1 \
#                --self_neighbor True \
#                --neighbor True \
#                --use_pe False \
#                --num_heads 0 \
#                --batch_size 7 \
#                --lr 0.001 \
#                --momentum 0.9 \
#                --num_points 1024 \
#                --dropout 0.5 \
#                --use_sgd False \
#                --emb_dims 1024 \
#                --epochs 200

# python main.py --dataset tumor_core \
#                --exp_name self_att_neighbor_att_tumor_core_1 \
#                --self_neighbor True \
#                --neighbor True \
#                --use_pe False \
#                --num_heads 0 \
#                --batch_size 7 \
#                --lr 0.001 \
#                --momentum 0.9 \
#                --num_points 1024 \
#                --dropout 0.5 \
#                --use_sgd False \
#                --emb_dims 1024 \
#                --epochs 200

# PE_neighbor_att

# python main.py --dataset diseases \
#                --exp_name PE_neighbor_atts_diseases_2 \
#                --self_neighbor False \
#                --neighbor True \
#                --use_pe True \
#                --num_heads 0 \
#                --batch_size 7 \
#                --lr 0.001 \
#                --momentum 0.9 \
#                --num_points 1024 \
#                --dropout 0.5 \
#                --use_sgd False \
#                --emb_dims 1024 \
#                --epochs 200

# python main.py --dataset margin \
#                --exp_name PE_neighbor_atts_margin_1 \
#                --self_neighbor False \
#                --neighbor True \
#                --use_pe True \
#                --num_heads 0 \
#                --batch_size 7 \
#                --lr 0.001 \
#                --momentum 0.9 \
#                --num_points 1024 \
#                --dropout 0.5 \
#                --use_sgd False \
#                --emb_dims 1024 \
#                --epochs 200

# python main.py --dataset tumor_core \
#                --exp_name PE_neighbor_atts_tumor_core_1 \
#                --self_neighbor False \
#                --neighbor True \
#                --use_pe True \
#                --num_heads 0 \
#                --batch_size 7 \
#                --lr 0.001 \
#                --momentum 0.9 \
#                --num_points 1024 \
#                --dropout 0.5 \
#                --use_sgd False \
#                --emb_dims 1024 \
#                --epochs 200

# SRNet

CUDA_VISIBLE_DEVICES=0 python main.py --dataset diseases \
               --model srnet \
               --exp_name test \
               --self_neighbor True \
               --neighbor True \
               --use_pe True \
               --num_heads 1 \
               --batch_size 7 \
               --lr 0.001 \
               --momentum 0.9 \
               --num_points 1024 \
               --dropout 0.5 \
               --use_sgd False \
               --emb_dims 1024 \
               --epochs 200

# CUDA_VISIBLE_DEVICES=0 python main.py --dataset margin \
#                --exp_name srnet_margin_2 \
#                --self_neighbor True \
#                --neighbor True \
#                --use_pe True \
#                --num_heads 1 \
#                --batch_size 7 \
#                --lr 0.001 \
#                --momentum 0.9 \
#                --num_points 1024 \
#                --dropout 0.5 \
#                --use_sgd False \
#                --emb_dims 1024 \
#                --epochs 200

# CUDA_VISIBLE_DEVICES=0 python main.py --dataset tumor_core \
#                --exp_name srnet_tumor_core_2 \
#                --self_neighbor True \
#                --neighbor True \
#                --use_pe True \
#                --num_heads 1 \
#                --batch_size 7 \
#                --lr 0.001 \
#                --momentum 0.9 \
#                --num_points 1024 \
#                --dropout 0.5 \
#                --use_sgd False \
#                --emb_dims 1024 \
#                --epochs 200

############
# CUDA_VISIBLE_DEVICES=1 python main.py --dataset diseases \
#                --exp_name srnet_diseases_3 \
#                --self_neighbor True \
#                --neighbor True \
#                --use_pe True \
#                --num_heads 1 \
#                --batch_size 7 \
#                --lr 0.001 \
#                --momentum 0.9 \
#                --num_points 1024 \
#                --dropout 0.5 \
#                --use_sgd False \
#                --emb_dims 1024 \
#                --epochs 200

# CUDA_VISIBLE_DEVICES=1 python main.py --dataset margin \
#                --exp_name srnet_margin_3 \
#                --self_neighbor True \
#                --neighbor True \
#                --use_pe True \
#                --num_heads 1 \
#                --batch_size 7 \
#                --lr 0.001 \
#                --momentum 0.9 \
#                --num_points 1024 \
#                --dropout 0.5 \
#                --use_sgd False \
#                --emb_dims 1024 \
#                --epochs 200

# CUDA_VISIBLE_DEVICES=1 python main.py --dataset tumor_core \
#                --exp_name srnet_tumor_core_3 \
#                --self_neighbor True \
#                --neighbor True \
#                --use_pe True \
#                --num_heads 1 \
#                --batch_size 7 \
#                --lr 0.001 \
#                --momentum 0.9 \
#                --num_points 1024 \
#                --dropout 0.5 \
#                --use_sgd False \
#                --emb_dims 1024 \
#                --epochs 200

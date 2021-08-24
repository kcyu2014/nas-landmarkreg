#!/usr/bin/env bash

/home/yukaiche/anaconda3/envs/pytorch-0.3/bin/python \
 search_main.py --epochs=1000 \
              --num_intermediate_nodes=2 \
              --search_policy=ws_r_batch \
              --gpu=0 \
              --concat=2 \
              --evaluation_seed=1278 \
              --seed_range_start=1270 \
              --seed_range_end=1273 \
              --test_dir=experiments/node2-ws-no-policy \
              --handle_hidden_mode=NORM \
              --clip_hidden_norm 0.25 \
              --evaluate_after_search False


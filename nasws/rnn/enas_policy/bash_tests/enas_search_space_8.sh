#!/usr/bin/env bash

python3.6 search_main.py --epochs=$4 \
                                   --num_intermediate_nodes=8 \
                                   --search_policy=enas \
                                   --gpu=0 \
                                   --evaluation_seed=1278 \
                                   --test_dir=$3 \
                                   --seed_range_start=$1 \
                                   --seed_range_end=$2
                                   --handle_hidden_mode=ACTIVATION
# THis is for reduced search space.
#cd ../.. && python3.6 search_main.py --epochs=1000 \
#                                   --num_intermediate_nodes=2 \
#                                   --search_policy=enas-small \
#                                   --test_dir="enas_policy/node-2" \
#                                   --gpu=0 \
#                                   --seed_range_start=1628 \
#                                   --seed_range_end=1638

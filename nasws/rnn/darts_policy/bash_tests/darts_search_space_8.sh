python3.6 search_main.py --epochs=$4 \
                                      --num_intermediate_nodes=8 \
                                      --search_policy=darts \
                                      --gpu=0 \
                                      --concat=8 \
                                      --evaluation_seed=1278 \
                                      --seed_range_start=$1 \
                                      --seed_range_end=$2 \
                                      --test_dir=$3 \
                                      --handle_hidden_mode=RELOAD
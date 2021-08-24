python3.6 search_main.py --epochs=4000 \
                          --num_intermediate_nodes=8 \
                          --search_policy=darts \
                          --forced_genotype_by_name=DARTS \
                          --gpu=0 \
                          --concat=8 \
                          --dropoute=0.1 \
                          --evaluation_seed=1267 \
                          --seed_range_start=3 \
                          --seed_range_end=4 \
                          --handle_hidden_mode=RELOAD \
                          --continue_train \
                          --test_dir=darts_policy/tests_dir \
                          --main_path=darts_policy/tests_dir/search_seed3-evalseed_1267-Titan-V100 \
                          --resume_path=EXP_search_SEED_3_eval_seed_1267_geno_id_0-V100
#                          --main_path=darts_policy/tests_dir/date_20190502-144408_geno_id_None_search_policy_darts_nodes_8_use_leaf_False_start_seed_3_cuda9 \
#                          --resume_path=EXP_search_SEED_3_eval_seed_1267_geno_id_0_20190502-144408



#                          --continue_train=darts_policy/tests_dir/date_20190502-160224_geno_id_None_search_policy_darts_nodes_8_use_leaf_False_start_seed_3_cuda9

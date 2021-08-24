PYTHON=/home/rranftl/anaconda3/envs/nas-project/bin/python #/home/yukaiche/anaconda3/envs/pytorch-latest/bin/python
PARTITION=q6,q8
GPU=3
DAYS=16
SUBMIT_FILE='bash/helper/slurm_submit_jobs.sh'
DATADIR=/export/share/Datasets/ILSVRC2012
batch_size=1024

if [ "$3" = "cvlab" ]; then
SUBMIT_FILE='bash/helper/kubernetes_submit_jobs.sh'
PYTHON=python
GPU=2
DAYS=3
PARTITION=v100
batch_size=256
# DATADIR=/cvlabsrc1/cvlab/datasets_kyu/ILSVRC2012-debug
DATADIR=/data/kyu/ILSVRC2012
# DATADIR=/cvlabsrc1/cvlab/datasets_kyu/ILSVRC2012
# DATADIR=/cvlabsrc1/cvlab/datasets_kyu/ILSVRC2015/Data/CLS-LOC/
fi


# different aspects to check.
EXPERIMENT="cvpr-imagenet-fix-node"

if [ $1 = 'debug' ]; then
python cnn_search_main.py               --epochs=200               --epochs_lr=200               --init_channels 16               --batch_size 512               --evaluate_batch_size 512               --save_every_epoch 30               --dataset imagenet               --num_archs_subspace -1               --dataset_dir /cvlabsrc1/cvlab/datasets_kyu/ILSVRC2012-debug   --learning_rate_min 1e-5            --num_intermediate_nodes 4               --layers 8               --extensive_save False               --gpus 2               --n_worker 32               --learning_rate 0.1               --learning_rate_scheduler cosine               --weight_decay 5e-5               --supernet_cell_type op_on_edge_fix               --search_space darts_nds               --nasbenchnet_vertex_type mixedvertex               --label_smooth 0               --train_portion 0.15               --valid_portion 0.05               --test_dir experiments/cvpr-imagenet-fix-node/debug/search-imagenet-spos-pretrain/darts_nds/lr0.1-spos_rankloss-channels16-layers8epochs200-smooth0               --seed_range_start 1270               --seed_range_end 1271               --search_policy spos               --supernet_train_method spos_rankloss               --supernet_warmup_epoch 140               --supernet_warmup_method oneshot               --bn_type=bn               --wsbn_track_stat=False               --wsbn_affine=False               --resume True               --resume_path experiments/cvpr-imagenet-fix-node/search-imagenet-pretrained/darts_nds/lr0.1-spos_rankloss-channels16-layers8epochs150-smooth0/SEED1270_1271-cuda9Tesla-V100-SXM2-32GB/spos_rankloss_SEED_1270/checkpoint.pt.120               --softoneshot_alpha 0.01               --landmark_loss_procedure random_pairwise_loss               --landmark_loss_fn mae_relu               --landmark_use_valid True               --landmark_loss_coef 10               --landmark_loss_coef_scheduler cosine_increase               --landmark_warmup_epoch 10               --landmark_forward_mode oneshot               --landmark_loss_adjacent True               --landmark_loss_weighted default               --landmark_num_archs  30               --landmark_sample_method fixed               --landmark_loss_random_pairs 1               --landmark_sample_method random               --evaluate_after_search               --evaluate_step_budget 50               --evaluate_sampler evolutionary   --debug            2>&1                   | tee logs/cvpr-imagenet-fix-node/search-imagenet-spos-pretrain_darts_nds-lr0.1-spos_rankloss-channels16-layers8epochs200-smooth0-1270.log ;
# bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-nasbench-supernet-train-darts_nds_subsample-subspace10-epochs150-lr0.1-1269-2.sh
# bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-nasbench-supernet-train-darts_nds_subsample-subspace20-epochs150-lr0.1-1269-2.sh
# bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-nasbench-supernet-train-darts_nds_subsample-subspace30-epochs150-lr0.1-1269-2.sh 

fi 


if [[ $1 = 'submit' || $1 = 'evaluate' ]]; then
    if [ "$2" -eq 0 ]; then
      SUB_EXP='search-imagenet'
      declare -a arr3=('spos' 'spos_rankloss')
    elif [ "$2" -eq 1 ]; then
      SUB_EXP='search-imagenet-subspace'
      # SUB_EXP='search-imagenet-subspace'
      # declare -a arr3=('120')
      declare -a arr3=('-1')
    elif [ "$2" -eq 2 ]; then
      SUB_EXP='search-imagenet-lr'
      declare -a arr3=('0.5' '0.25' '0.05')
      # declare -a arr3=('spos' 'spos_rankloss')
      # conclusion: 0.25 as LR
    elif [ "$2" -eq 3 ]; then
      SUB_EXP='search-imagenet-epochs'
      declare -a arr3=('100')
      # declare -a arr3=('200' '150' '100')
    elif [ "$2" -eq 10 ]; then
      SUB_EXP='search-imagenet-pretrained'
      declare -a arr3=('150')
    elif [ "$2" -eq 50 ]; then
      SUB_EXP='search-imagenet-nao-pretrain-lr'
      # declare -a arr3=('spos_rankloss')
      declare -a arr3=('spos' 'spos_rankloss')
    elif [ "$2" -eq 51 ]; then
      SUB_EXP='search-imagenet-pcdarts-pretrain-lr'
      declare -a arr3=('darts' 'darts_rankloss')
    elif [ "$2" -eq 52 ]; then
      SUB_EXP='search-imagenet-gdas-pretrain-lr'
      declare -a arr3=('darts')
      # declare -a arr3=('darts' 'darts_rankloss')
    elif [ "$2" -eq 53 ]; then
      SUB_EXP='search-imagenet-spos-pretrain-lr'
      # declare -a arr3=('spos_rankloss')
      declare -a arr3=('spos' 'spos_rankloss')
    else
        exit
    fi
    
    declare -a arr1=('darts_nds')
    # declare -a arr2=("1269")
    declare -a arr2=("1268" "1269")
    for ((i=0;i<${#arr1[@]};i=i+1)); do
    for ((k=0;k<${#arr3[@]};k=k+1)); do
        cmdALL=''
    for ((j=0;j<${#arr2[@]};j=j+1)); do
    # for ((l=0;l<${#arr4[@]};l=l+1)); do
        space=${arr1[i]}
        start=${arr2[j]}
        resume_path=''
        policy=spos
        portion=0.15
        bn_type=bn
        bn_affine=True
        bn_track=False
        arch=-1
        plot_step=51
        vertex_type=mixedvertex
        supernet_warmup_epoch=40
        supernet_warmup_method=oneshot
        # landmark best setting
        coef=10 
        pair=1
        vld='True'
        adjacent='True'
        landmark_weighted=default
        arch=30
        loss_fn=mae_relu
        landmark_loss_procedure=random_pairwise_loss
        coef_scheduler='cosine_increase'
        supernet_warmup_method=oneshot
        landmark_forward_mode=oneshot
        supernet_eval_method=oneshot
        softoneshot_alpha=0.01
        # default arguments.
          # for NDS only, no other search configs
        epochs=50
        epochs_lr=50
        lr_scheduler='cosine'
        weight_decay='5e-5'
        optimizer='sgd'
        init_channels=16
        # init_channels=48
        # batch_size=256 
        batch_size=1024
        node=4
        label_smooth=0
        lr=0.1
        # lr=0.5
        layers=8
        GPU=$GPU
        bn_affine=False
        # for NASbench 102
        landmark_warmup_epoch=10
        method='spos_rankloss'
        PARTITION=gpu
        subspace=-1
        supernet_cell_type='op_on_edge_fix'
        start=$(( ${start} +1 ))

        if [ "$2" -eq 0 ]; then
          method=${arr3[k]}
          EXPNAME="${space}/${method}-epochs${epochs}-lr${lr}"
        elif [ "$2" -eq 1 ]; then
          subspace=${arr3[k]}
          method='spos'
          EXPNAME="${space}/subspace${subspace}-${method}-channels${init_channels}-layers${layers}epochs${epochs}-lr${lr}-smooth${label_smooth}"
        elif [ "$2" -eq 2 ]; then
          # subspace=${arr3[k]}
          method='spos'
          lr=${arr3[k]}
          EXPNAME="${space}/lr${lr}-${method}-channels${init_channels}-layers${layers}epochs${epochs}-smooth${label_smooth}"
        elif [ "$2" -eq 3 ]; then
          # subspace=${arr3[k]}
          method='spos_rankloss'
          lr=0.1
          # lr is not true and at sweat spot ...
          supernet_warmup_epoch=50
          epochs=${arr3[k]}
          epochs_lr=${arr3[k]}
          EXPNAME="${space}/lr${lr}-${method}-channels${init_channels}-layers${layers}epochs${epochs}-smooth${label_smooth}"
        elif [ "$2" -eq 10 ]; then
          # subspace=${arr3[k]}
          method='spos_rankloss'
          lr=0.1
          portion=0.9
          epochs=${arr3[k]}
          epochs_lr=150
          supernet_warmup_epoch=140
          EXPNAME="${space}/lr${lr}-${method}-channels${init_channels}-layers${layers}epochs${epochs}-smooth${label_smooth}"        
        elif [ "$2" -eq 50 ]; then
          # subspace=${arr3[k]}
          method=${arr3[k]}
          lr=0.1
          epochs=200
          epochs_lr=200
          learning_rate_min=1e-5
          tmp_end=$(( ${start} +1 ))
          resume_path="experiments/cvpr-imagenet-fix-node/search-imagenet-pretrained/darts_nds/lr0.1-spos_rankloss-channels16-layers8epochs150-smooth0/SEED${start}_${tmp_end}-cuda9Tesla-V100-SXM2-32GB/spos_rankloss_SEED_${start}/checkpoint.pt.120"
          supernet_warmup_epoch=140
          batch_size=512
          policy=nao
          EXPNAME="${space}/lr${lr}-${method}-channels${init_channels}-layers${layers}epochs${epochs}-smooth${label_smooth}"  
        elif [ "$2" -eq 51 ]; then
          # subspace=${arr3[k]}
          method=${arr3[k]}
          lr=0.1
          epochs=80
          epochs_lr=130
          learning_rate_min=1e-5
          supernet_warmup_epoch=40
          supernet_warmup_method=softoneshot
          landmark_forward_mode=softoneshot
          supernet_eval_method=softoneshot
          softoneshot_alpha=0.01
          policy=pcdarts
          EXPNAME="${space}/lr${lr}-${method}-channels${init_channels}-layers${layers}epochs${epochs}-smooth${label_smooth}"
        elif [ "$2" -eq 52 ]; then
          # subspace=${arr3[k]}
          method=${arr3[k]}
          lr=0.1
          epochs=200
          epochs_lr=200
          learning_rate_min=1e-5
          tmp_end=$(( ${start} +1 ))
          resume_path="experiments/cvpr-imagenet-fix-node/search-imagenet-pretrained/darts_nds/lr0.1-spos_rankloss-channels16-layers8epochs150-smooth0/SEED${start}_${tmp_end}-cuda9Tesla-V100-SXM2-32GB/spos_rankloss_SEED_${start}/checkpoint.pt.120"
          supernet_warmup_epoch=140
          batch_size=512
          policy=gdas
          # load the pre-trained weights...
          EXPNAME="${space}/lr${lr}-${method}-channels${init_channels}-layers${layers}epochs${epochs}-smooth${label_smooth}"        
        elif [ "$2" -eq 53 ]; then
          # subspace=${arr3[k]}
          method=${arr3[k]}
          lr=0.1
          learning_rate_min=1e-5
          epochs=200
          epochs_lr=200
          tmp_end=$(( ${start} +1 ))
          resume_path=None
          # resume_path="experiments/cvpr-imagenet-fix-node/search-imagenet-pretrained/darts_nds/lr0.1-spos_rankloss-channels16-layers8epochs150-smooth0/SEED${start}_${tmp_end}-cuda9Tesla-V100-SXM2-32GB/spos_rankloss_SEED_${start}/checkpoint.pt.120"
          supernet_warmup_epoch=140
          batch_size=512
          policy=spos
          EXPNAME="${space}/lr${lr}-${method}-channels${init_channels}-layers${layers}epochs${epochs}-smooth${label_smooth}"        
        else
            exit
        fi

        EXPNAME="${EXPNAME}"
        LOGNAME="${EXPNAME////-}-${start}"
        if [ $1 = 'submit' ]; then
        cmd=" mkdir -p logs/${EXPERIMENT}/ && \
        $PYTHON cnn_search_main.py \
              --epochs=${epochs} \
              --epochs_lr=${epochs_lr} \
              --init_channels ${init_channels} \
              --batch_size ${batch_size} \
              --evaluate_batch_size ${batch_size} \
              --save_every_epoch 30 \
              --dataset imagenet \
              --num_archs_subspace $subspace \
              --dataset_dir $DATADIR \
              --num_intermediate_nodes ${node} \
              --layers ${layers} \
              --extensive_save False \
              --gpus $GPU \
              --n_worker 32 \
              --learning_rate ${lr} \
              --learning_rate_scheduler ${lr_scheduler} \
              --learning_rate_min ${learning_rate_min} \
              --weight_decay ${weight_decay} \
              --supernet_cell_type ${supernet_cell_type} \
              --search_space ${space} \
              --nasbenchnet_vertex_type ${vertex_type} \
              --label_smooth ${label_smooth} \
              --train_portion ${portion} \
              --valid_portion 0.05 \
              --test_dir experiments/${EXPERIMENT}/${SUB_EXP}/${EXPNAME} \
              --seed_range_start ${start} \
              --seed_range_end $(($start + 1)) \
              --search_policy ${policy} \
              --supernet_train_method ${method} \
              --supernet_warmup_epoch $supernet_warmup_epoch \
              --supernet_warmup_method ${supernet_warmup_method} \
              --bn_type=${bn_type} \
              --wsbn_track_stat=${bn_track} \
              --wsbn_affine=${bn_affine} \
              --resume True \
              --resume_path ${resume_path} \
              --softoneshot_alpha ${softoneshot_alpha} \
              --landmark_loss_procedure ${landmark_loss_procedure} \
              --landmark_loss_fn ${loss_fn} \
              --landmark_use_valid ${vld} \
              --landmark_loss_coef ${coef} \
              --landmark_loss_coef_scheduler ${coef_scheduler} \
              --landmark_warmup_epoch ${landmark_warmup_epoch} \
              --landmark_forward_mode ${landmark_forward_mode} \
              --landmark_loss_adjacent ${adjacent} \
              --landmark_loss_weighted ${landmark_weighted} \
              --landmark_num_archs  ${arch} \
              --landmark_sample_method fixed \
              --landmark_loss_random_pairs ${pair} \
              --landmark_sample_method random \
              --evaluate_after_search \
              --evaluate_step_budget 150 \
              --evaluate_evolutionary_population_size 20 \
              --evaluate_evolutionary_tournament_size 10 \
              --evaluate_sampler evolutionary \
              2>&1 \
                  | tee logs/${EXPERIMENT}/${SUB_EXP}_$LOGNAME.log "
        else
        echo 'wrong!'
        exit
        fi
        cmdALL="$cmd; ${cmdALL}"
      #   cmdALL="mkdir -p logs/${EXPERIMENT}/ && echo '$cmd' && $cmd"
      # KBSNAME=imagenet-$SUB_EXP-$EXPNAME-${start}
      # bash $SUBMIT_FILE "$cmdALL" $(($i))  $GPU $PARTITION $DAYS $KBSNAME '' k8s/imagenet.yaml $4 $3 
    done
      cmdALL="mkdir -p logs/${EXPERIMENT}/; echo '$cmdALL';$cmdALL"
      KBSNAME=imagenet-$SUB_EXP-$EXPNAME-multiseed
      bash $SUBMIT_FILE "$cmdALL" $(($i)) $GPU $PARTITION $DAYS $KBSNAME '' k8s/imagenet.yaml $4 $3 

    done
    done
    # done

fi


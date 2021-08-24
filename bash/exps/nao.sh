# Generate the search space
# NASBench 102 type of search 
# hpyerparameters checking here.
# we should first validate the hyper-params with multi-seed as entry course.

PYTHON=/home/rranftl/anaconda3/envs/nas-project/bin/python #/home/yukaiche/anaconda3/envs/pytorch-latest/bin/python
PARTITION=gpu
# PARTITION=q8 # for rtx 8000,  q6 for RTX 6000
GPU=1
DAYS=2
SUBMIT_FILE='bash/helper/slurm_submit_jobs.sh'
CONFIG=configs/cluster.json

if [ "$3" = "cvlab" ]; then
SUBMIT_FILE='bash/helper/kubernetes_submit_jobs.sh'
PYTHON=python
GPU=1
DAYS=3
CONFIG=configs/cvlab.json
fi

# Validating some hyperparameters during search.
# all other aspects should keep to a minimum.


# different aspects to check.
EXPERIMENT="cvpr-landmark-loss"

if [ $1 = 'debug' ]; then

python cnn_search_main.py --epochs=30               --init_channels 64               --batch_size 64               --save_every_epoch 30               --num_intermediate_nodes 4               --layers 3               --extensive_save False               --gpus 2               --learning_rate 0.025               --learning_rate_scheduler cosine               --weight_decay 1e-4               --search_space nasbench201               --nasbenchnet_vertex_type mixedvertex               --train_portion 0.5               --test_dir experiments/cvpr-landmark-loss/debug-nao/nasbench201/node4-spos-epochs30-lr0.025               --seed_range_start 1268               --seed_range_end 1269               --search_policy nao               --supernet_train_method spos               --supernet_warmup_epoch 3               --supernet_warmup_method oneshot               --bn_type=bn               --wsbn_track_stat=False               --wsbn_affine=True               --resume True               --landmark_loss_procedure random_pairwise_loss               --landmark_loss_fn mae_relu               --landmark_use_valid True               --landmark_loss_coef 10               --landmark_loss_adjacent True               --landmark_loss_weighted default               --landmark_num_archs  30               --landmark_sample_method fixed               --landmark_loss_random_pairs 1               --landmark_sample_method random               --evaluate_after_search               --evaluate_step_budget 1000               --evaluate_sampler evolutionary      --debug         2>&1                   | tee logs/cvpr-landmark-loss/debug-nao_nasbench101-spos-epochs150-lr0.025-1268.log

# bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-nasbench-supernet-train-darts_nds_subsample-subspace10-epochs150-lr0.1-1269-2.sh
# bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-nasbench-supernet-train-darts_nds_subsample-subspace20-epochs150-lr0.1-1269-2.sh
# bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-nasbench-supernet-train-darts_nds_subsample-subspace30-epochs150-lr0.1-1269-2.sh 

fi 


if [[ $1 = 'submit' || $1 = 'evaluate' ]]; then
    if [ "$2" -eq 0 ]; then
      SUB_EXP='nao-baseline'
      declare -a arr3=('spos')
      # declare -a arr3=('spos' 'spos_rankloss')
      # do not change any configuration just run three times.
    elif [ "$2" -eq 1 ]; then
      SUB_EXP='nao-rankloss-warmup-epochs'
      declare -a arr3=('50' '150' '250')
    elif [ "$2" -eq 2 ]; then
      SUB_EXP='nao-rankloss-lr'
      declare -a arr3=('0.075' '0.01' '0.025')
      # stopped because the results are inversed. try with inverse loss fn ...
    elif [ "$2" -eq 10 ]; then
      SUB_EXP='nao-rankloss-lossfn'
      declare -a arr3=('mae_softplus_beta5' 'mae_relu')
      # declare -a arr3=('mae_relu_inverse')
    else
        exit
    fi
    
    # declare -a arr1=('darts_nds')
    declare -a arr1=('nasbench101')
    # declare -a arr1=('nasbench101' 'darts_nds')
    # declare -a arr1=('darts_nds' 'nasbench101')
    # declare -a arr2=("1268" "1269")
    declare -a arr2=("1268" "1269" "1270")

    for ((i=0;i<${#arr1[@]};i=i+1)); do
    for ((k=0;k<${#arr3[@]};k=k+1)); do
        cmdALL=''
    for ((j=0;j<${#arr2[@]};j=j+1)); do
    # for ((l=0;l<${#arr4[@]};l=l+1)); do
        space=${arr1[i]}
        start=${arr2[j]}
        policy=nao
        portion=0.5
        bn_type=bn
        bn_affine=True
        bn_track=False
        arch=-1
        plot_step=51
        vertex_type=mixedvertex
        supernet_warmup_epoch=150
        coef_scheduler='cosine_increase'
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
        landmark_warmup_epoch=10
        # default arguments.
        if [[ "$space" =~ .*"nasbench101".* ]]; then
          # for nasbench default arguments.
              supernet_warmup_epoch=150
              epochs=400
              epochs_lr=400
              method='spos_rankloss'
              lr_scheduler='cosine'
              optimizer='sgd'
              init_channels=64 
              batch_size=128
              node=5
              layers=3 # not important, not used anyway.
              GPU=2
              lr=0.025
              PARTITION=gpu
              weight_decay='1e-4'
        elif [[ "$space" =~ .*"darts_nds".* ]]; then
          # for NDS
              supernet_warmup_epoch=150
              epochs=400
              epochs_lr=400
              method='spos_rankloss'
              lr_scheduler='cosine'
              weight_decay='5e-4'
              optimizer='sgd'
              init_channels=36
              batch_size=64
              node=4
              lr=0.05
              layers=12
              arch=100
              GPU=1
              PARTITION=gpu
              bn_affine=False
              start=$(( ${start} +1 ))
          elif [[ "$space" =~ .*"nasbench201".* ]]; then
              epochs=400
              epochs_lr=400
              supernet_warmup_epoch=100
              landmark_warmup_epoch=10
              method='spos_rankloss'
              lr_scheduler='cosine'
              weight_decay='5e-4'
              batch_size=256
              optimizer='sgd'
              node=4
              lr=0.025
              init_channels=16
              layers=5
              GPU=1
              PARTITION=gpu
        else
          exit
        fi

        if [ "$2" -eq 0 ]; then
          method=${arr3[k]}
          EXPNAME="${space}/${method}-epochs${epochs}-lr${lr}"
        elif [ "$2" -eq 1 ]; then
          supernet_warmup_epoch=${arr3[k]}
          EXPNAME="${space}/epochs${epochs}-lr${lr}-warmup-${supernet_warmup_epoch}"
        elif [ "$2" -eq 2 ]; then
          lr=${arr3[k]}
          EXPNAME="${space}/epochs${epochs}-lr${lr}-warmup-${supernet_warmup_epoch}"
        elif [ "$2" -eq 10 ]; then
          loss_fn=${arr3[k]}
          EXPNAME="${space}/epochs${epochs}-lr${lr}-lossfn-${loss_fn}"
        else
            exit
        fi

        EXPNAME="${EXPNAME}"
        LOGNAME="${EXPNAME////-}-${start}"
        if [ $1 = 'submit' ]; then
        cmd="$PYTHON cnn_search_main.py --epochs=${epochs} \
              --epochs_lr ${epochs_lr} \
              --init_channels ${init_channels} \
              --batch_size ${batch_size} \
              --save_every_epoch 30 \
              --num_intermediate_nodes ${node} \
              --layers ${layers} \
              --extensive_save False \
              --gpus $GPU \
              --learning_rate ${lr} \
              --learning_rate_scheduler ${lr_scheduler} \
              --weight_decay ${weight_decay} \
              --search_space ${space} \
              --nasbenchnet_vertex_type ${vertex_type} \
              --train_portion ${portion} \
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
              --landmark_loss_procedure ${landmark_loss_procedure} \
              --landmark_loss_fn ${loss_fn} \
              --landmark_use_valid ${vld} \
              --landmark_loss_coef ${coef} \
              --landmark_loss_coef_scheduler ${coef_scheduler} \
              --landmark_warmup_epoch ${landmark_warmup_epoch} \
              --landmark_loss_adjacent ${adjacent} \
              --landmark_loss_weighted ${landmark_weighted} \
              --landmark_num_archs  ${arch} \
              --landmark_sample_method fixed \
              --landmark_loss_random_pairs ${pair} \
              --landmark_sample_method random \
              --evaluate_after_search \
              --evaluate_step_budget 1000 \
              --evaluate_sampler evolutionary \
              2>&1 \
                  | tee logs/${EXPERIMENT}/${SUB_EXP}_$LOGNAME.log "
        else
        echo 'wrong!'
        exit
        fi
        cmdALL="mkdir -p logs/${EXPERIMENT}/ && echo '$cmd' && $cmd"
        KBSNAME=nao-$SUB_EXP-$EXPNAME-seed${start}
        bash $SUBMIT_FILE "$cmdALL" $(($i)) $GPU $PARTITION $DAYS $KBSNAME '' k8s/cifar.yaml $4 $3 
    done
      # KBSNAME=nao-$SUB_EXP-$EXPNAME-multiseeds
      # cmdALL="mkdir -p logs/${EXPERIMENT}/ && echo '$cmd' && $cmd; ${cmdALL}"
      # bash $SUBMIT_FILE "$cmdALL" $(($i)) $GPU $PARTITION $DAYS $KBSNAME '' k8s/cifar.yaml $4 $3 
    done
    done
    # done

fi


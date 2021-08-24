
#############
# Running
#############
#  kubectl exec kyu-imagenet-long-7-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-ldmk-iteration-nasbench201-epochs200-lr0.025-arch10-multiseed-0.sh
#  kubectl exec kyu-imagenet-long-7-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-ldmk-iteration-nasbench201-epochs200-lr0.025-arch20-multiseed-0.sh
#  kubectl exec kyu-imagenet-long-7-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-ldmk-iteration-nasbench201-epochs200-lr0.025-arch200-multiseed-0.sh
#  kubectl exec kyu-imagenet-long-7-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-ldmk-iteration-nasbench201-epochs200-lr0.025-arch1-multiseed-0.sh
#  kubectl exec kyu-imagenet-long-7-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-ldmk-iteration-nasbench201-epochs200-lr0.025-arch2-multiseed-0.sh
#  kubectl exec kyu-imagenet-long-7-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-ldmk-iteration-nasbench201-epochs200-lr0.025-arch3-multiseed-0.sh

# ablation study: a better landmark sampling algorithm
#  kubectl exec kyu-imagenet-long-7-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-ldmk-sampling-distances-v2-nasbench201-epochs200-lr0.025-distance-5-multiseed-0.sh
#  kubectl exec kyu-imagenet-long-7-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-ldmk-sampling-distances-v2-nasbench201-epochs200-lr0.025-distance-4-multiseed-0.sh
#  kubectl exec kyu-imagenet-long-7-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-ldmk-sampling-distances-v2-nasbench201-epochs200-lr0.025-distance-3-multiseed-0.sh
#  kubectl exec kyu-imagenet-long-7-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-ldmk-sampling-distances-v2-nasbench201-epochs200-lr0.025-distance-2-multiseed-0.sh
#  kubectl exec kyu-imagenet-long-7-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-ldmk-sampling-distances-v2-nasbench201-epochs200-lr0.025-distance-1-multiseed-0.sh

# ImageNet experiment again.
# 50 epochs one
# kubectl exec kyu-imagenet-long-7-interactive -- bash  /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-cifar-fromscratch-cifar10-published-arch-1-epoch120-1.sh

# Longer training, full imagenet training approach...
#  1 3 4 5 9 total 5 exps...
kubectl exec kyu-imagenet-long-9-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-fromscratch-imagenet-cvpr-imagenet-fromscratch-imagenet-fix-spos_rankloss-4-epoch250-0.sh
kubectl exec kyu-imagenet-long-5-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-fromscratch-imagenet-cvpr-imagenet-fromscratch-imagenet-fix-nao_rankloss-0-epoch300-0.sh
kubectl exec kyu-imagenet-long-4-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-fromscratch-imagenet-cvpr-imagenet-fromscratch-imagenet-fix-gdas-0-epoch250-0.sh
kubectl exec kyu-imagenet-long-3-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-fromscratch-imagenet-cvpr-imagenet-fromscratch-imagenet-fix-nao-2-epoch250-0.sh
kubectl exec kyu-imagenet-long-1-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-fromscratch-imagenet-cvpr-imagenet-fromscratch-imagenet-fix-gdas_rankloss-1-epoch250-0.sh


# GDAS baseline

# kubectl exec kyu-imagenet-long-9-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-fromscratch-imagenet-cvpr-imagenet-fromscratch-imagenet-fix-gdas-0-epoch60-0.sh
# kubectl exec kyu-imagenet-long-7-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-fromscratch-imagenet-cvpr-imagenet-fromscratch-imagenet-fix-gdas-1-epoch60-1.sh
# kubectl exec kyu-imagenet-long-1-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-fromscratch-imagenet-cvpr-imagenet-fromscratch-imagenet-fix-gdas-2-epoch60-2.sh
# kubectl exec kyu-imagenet-long-7-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-fromscratch-imagenet-cvpr-imagenet-fromscratch-imagenet-fix-gdas-3-epoch60-3.sh

# kubectl exec kyu-imagenet-long-9-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-fromscratch-imagenet-cvpr-imagenet-fromscratch-imagenet-fix-gdas-4-epoch60-4.sh
# kubectl exec kyu-imagenet-long-4-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-fromscratch-imagenet-nips-landmark-fromscratch-imagenet-fix-enas-DARTS_V2-0-epoch60-1.sh
# kubectl exec kyu-imagenet-long-3-interactive -- bash  /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-fromscratch-imagenet-nips-landmark-fromscratch-imagenet-fix-published-PC_DARTS_image-0-epoch60-0.sh


# kubectl exec kyu-imagenet-long-7-interactive -- bash  /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-cifar-fromscratch-cifar10-published-arch-0-epoch120-0.sh
# kubectl exec kyu-imagenet-long-7-interactive -- bash  /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-cifar-fromscratch-cifar10-published-arch-2-epoch120-2.sh


# GDAS Rankloss train from scratch ...
# 5,7

kubectl exec kyu-imagenet-long-7-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-fromscratch-imagenet-cvpr-imagenet-fromscratch-imagenet-fix-gdas_rankloss-0-epoch60-0.sh
kubectl exec kyu-imagenet-long-3-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-fromscratch-imagenet-cvpr-imagenet-fromscratch-imagenet-fix-gdas_rankloss-1-epoch60-1.sh
kubectl exec kyu-imagenet-long-2-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-fromscratch-imagenet-cvpr-imagenet-fromscratch-imagenet-fix-gdas_rankloss-2-epoch60-2.sh
kubectl exec kyu-imagenet-long-0-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-fromscratch-imagenet-cvpr-imagenet-fromscratch-imagenet-fix-gdas_rankloss-3-epoch60-3.sh
kubectl exec kyu-imagenet-long-5-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-fromscratch-imagenet-cvpr-imagenet-fromscratch-imagenet-fix-gdas_rankloss-4-epoch60-4.sh


# SPOS rankloss again
kubectl exec kyu-imagenet-long-5-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-fromscratch-imagenet-cvpr-imagenet-fromscratch-imagenet-fix-spos_rankloss-3-epoch60-3.sh
kubectl exec kyu-imagenet-long-4-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-fromscratch-imagenet-cvpr-imagenet-fromscratch-imagenet-fix-spos_rankloss-4-epoch60-4.sh
kubectl exec kyu-imagenet-long-1-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-fromscratch-imagenet-cvpr-imagenet-fromscratch-imagenet-fix-spos_rankloss-2-epoch60-2.sh
kubectl exec kyu-imagenet-long-8-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-fromscratch-imagenet-cvpr-imagenet-fromscratch-imagenet-fix-spos_rankloss-0-epoch60-0.sh
kubectl exec kyu-imagenet-long-6-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-fromscratch-imagenet-cvpr-imagenet-fromscratch-imagenet-fix-spos_rankloss-1-epoch60-1.sh

# SPOS imagenet post search 
# kubectl exec kyu-imagenet-long-8-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-imagenet-search-imagenet-spos-pretrain-lr-darts_nds-lr0.1-spos-channels16-layers8epochs200-smooth0-multiseed-0.sh
# kubectl exec kyu-imagenet-long-6-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-imagenet-search-imagenet-spos-pretrain-lr-darts_nds-lr0.1-spos_rankloss-channels16-layers8epochs200-smooth0-multiseed-0.sh

# fix the results... on 1 card, batch size = 384...
kubectl exec kyu-imagenet-long-3-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-fromscratch-imagenet-cvpr-imagenet-fromscratch-imagenet-fix-spos-1-epoch60-1.sh
kubectl exec kyu-imagenet-long-4-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-fromscratch-imagenet-cvpr-imagenet-fromscratch-imagenet-fix-spos-0-epoch60-0.sh
kubectl exec kyu-imagenet-long-9-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-fromscratch-imagenet-cvpr-imagenet-fromscratch-imagenet-fix-spos-2-epoch60-2.sh

kubectl exec kyu-imagenet-long-7-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-fromscratch-imagenet-cvpr-imagenet-fromscratch-imagenet-fix-nao-2-epoch60-2.sh
kubectl exec kyu-imagenet-long-1-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-fromscratch-imagenet-cvpr-imagenet-fromscratch-imagenet-fix-nao-1-epoch60-1.sh
kubectl exec kyu-imagenet-long-0-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-fromscratch-imagenet-cvpr-imagenet-fromscratch-imagenet-fix-nao-0-epoch60-0.sh

# SUper slow training now ... !fixed!
kubectl exec kyu-imagenet-long-8-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-fromscratch-imagenet-cvpr-imagenet-fromscratch-imagenet-fix-nao_rankloss-0-epoch60-0.sh

# completely swap the order.... Now we only train from scratch for ImageNet on our cluster...
kubectl exec kyu-imagenet-long-2-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-fromscratch-imagenet-cvpr-imagenet-fromscratch-imagenet-fix-nao_rankloss-1-epoch60-1.sh

kubectl exec kyu-imagenet-long-6-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-fromscratch-imagenet-cvpr-imagenet-fromscratch-imagenet-fix-nao_rankloss-4-epoch60-4.sh

monodepth-long-1-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-fromscratch-imagenet-cvpr-imagenet-fromscratch-imagenet-fix-nao_rankloss-3-epoch60-3.sh

monodepth-long-2-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-fromscratch-imagenet-cvpr-imagenet-fromscratch-imagenet-fix-nao_rankloss-2-epoch60-2.sh


## Rankloss first then baseline...
kubectl exec kyu-imagenet-long-2-interactive -- bash  /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-imagenet-search-imagenet-spos-pretrain-lr-darts_nds-lr0.1-spos_rankloss-channels16-layers8epochs200-smooth0-multiseed-0.sh
kubectl exec kyu-imagenet-long-2-interactive -- bash  /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-imagenet-search-imagenet-spos-pretrain-lr-darts_nds-lr0.1-spos-channels16-layers8epochs200-smooth0-multiseed-0.sh


kubectl exec kyu-imagenet-long-1-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-imagenet-search-imagenet-gdas-pretrain-lr-darts_nds-lr0.1-darts_rankloss-channels16-layers8epochs200-smooth0-multiseed-0.sh
kubectl exec kyu-imagenet-long-1-interactive -- bash  /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-imagenet-search-imagenet-gdas-pretrain-lr-darts_nds-lr0.1-darts-channels16-layers8epochs200-smooth0-multiseed-0.sh


# TODO remind myself here...
kubectl exec kyu-imagenet-long-3-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-imagenet-search-imagenet-nao-pretrain-lr-darts_nds-lr0.1-spos_rankloss-channels16-layers8epochs200-smooth0-multiseed-0.sh

kubectl exec kyu-imagenet-long-3-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-imagenet-search-imagenet-nao-pretrain-lr-darts_nds-lr0.1-spos-channels16-layers8epochs200-smooth0-multiseed-0.sh

## 




# gdas experiments on ImageNet...

# squeeze some experiments for sampling distance results...


########## OLD #############



# ImageNet experimetn baseline
kubectl exec kyu-imagenet-long-5-interactive -- bash  /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-imagenet-search-imagenet-nao-pretrain-darts_nds-lr0.1-spos-channels16-layers8epochs150-smooth0-1270-0.sh

# ImageNet exp results...
# kubectl exec kyu-imagenet-long-1-interactive -- bash  /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-imagenet-search-imagenet-nao-pretrain-darts_nds-lr0.1-spos_rankloss-channels16-layers8epochs200-smooth0-multiseed-0.sh
kubectl exec kyu-imagenet-long-1-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-imagenet-search-imagenet-gdas-pretrain-darts_nds-lr0.1-darts_rankloss-channels16-layers8epochs200-smooth0-multiseed-0.sh

kubectl exec kyu-imagenet-long-6-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-imagenet-search-imagenet-nao-pretrain-darts_nds-lr0.1-spos-channels16-layers8epochs200-smooth0-multiseed-0.sh
kubectl exec kyu-imagenet-long-4-interactive -- bash  /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-imagenet-search-imagenet-nao-pretrain-darts_nds-lr0.1-spos_rankloss-channels16-layers8epochs150-smooth0-1269-0.sh
kubectl exec kyu-imagenet-long-4-interactive -- bash  /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-imagenet-search-imagenet-gdas-pretrain-darts_nds-lr0.1-darts-channels16-layers8epochs200-smooth0-multiseed-0.sh


kubectl exec kyu-imagenet-long-3-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-pcdarts-pcdarts-rankloss-lr-darts_nds-alpha0.01-epochs400-lr0.01-multiseed-0.sh
kubectl exec kyu-imagenet-long-3-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-pcdarts-pcdarts-rankloss-lr-darts_nds-alpha0.01-epochs400-lr0.01-multiseed-0.sh
kubectl exec kyu-imagenet-long-3-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-pcdarts-pcdarts-rankloss-arch-darts_nds-numarch200-epochs400-lr0.01-multiseed-0.sh
kubectl exec kyu-imagenet-long-3-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-pcdarts-pcdarts-rankloss-arch-darts_nds-numarch100-epochs400-lr0.01-multiseed-0.sh



# pretrain imagenet with a new setting (fix node)
kubectl exec kyu-imagenet-long-3-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-imagenet-search-imagenet-pretrained-darts_nds-lr0.1-spos_rankloss-channels16-layers8epochs150-smooth0-1269-0.sh
kubectl exec kyu-imagenet-long-2-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-imagenet-search-imagenet-pretrained-darts_nds-lr0.1-spos_rankloss-channels16-layers8epochs150-smooth0-1268-0.sh

kubectl exec kyu-imagenet-long-5-interactive -- /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-gdas-gdas-rankloss-landmark_arch-darts_nds-archs100-epochs400-lr0.1-multiseed-0.sh
kubectl exec kyu-imagenet-long-5-interactive --  bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-gdas-gdas-rankloss-landmark_arch-darts_nds-archs200-epochs400-lr0.1-multiseed-0.sh
kubectl exec kyu-imagenet-long-4-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-gdas-gdas-comparison-v3-darts_nds-darts_rankloss-epochs410-lr0.1-seed-1270-0.sh
kubectl exec kyu-imagenet-long-4-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-gdas-gdas-comparison-v3-darts_nds-darts_rankloss-epochs410-lr0.1-seed-1271-0.sh
# kubectl exec kyu-imagenet-long-5-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-gdas-gdas-comparison-v3-darts_nds-darts_rankloss-epochs410-lr0.1-seed-1269-0.sh
kubectl exec kyu-imagenet-long-1-interactive -- bash  /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-imagenet-search-imagenet-nao-pretrain-darts_nds-lr0.1-spos_rankloss-channels16-layers8epochs150-smooth0-1270-0.sh
kubectl exec kyu-imagenet-long-3-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-imagenet-search-imagenet-gdas-darts_nds-lr0.1-darts_rankloss-channels16-layers8epochs80-smooth0-1269-0.sh
kubectl exec kyu-imagenet-long-4-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-imagenet-search-imagenet-nao-darts_nds-lr0.1-spos-channels16-layers8epochs80-smooth0-1270-0.sh
kubectl exec kyu-imagenet-long-5-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-imagenet-search-imagenet-nao-darts_nds-lr0.1-spos_rankloss-channels16-layers8epochs80-smooth0-1270-0.sh
kubectl exec kyu-imagenet-long-1-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-ldmk-sampling-distances-nasbench201-epochs200-lr0.025-distance-5-multiseed-0.sh
kubectl exec kyu-imagenet-long-2-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-ldmk-sampling-distances-nasbench201-epochs200-lr0.025-distance-4-multiseed-0.sh
kubectl exec kyu-imagenet-long-3-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-ldmk-sampling-distances-nasbench201-epochs200-lr0.025-distance-3-multiseed-0.sh
kubectl exec kyu-imagenet-long-4-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-ldmk-sampling-distances-nasbench201-epochs200-lr0.025-distance-2-multiseed-0.sh
kubectl exec kyu-imagenet-long-5-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-ldmk-sampling-distances-nasbench201-epochs200-lr0.025-distance-1-multiseed-0.sh
kubectl exec kyu-imagenet-long-5-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-imagenet-search-imagenet-gdas-darts_nds-lr0.1-darts-channels16-layers8epochs80-smooth0-1269-0.sh

kubectl exec kyu-imagenet-long-5-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-imagenet-search-imagenet-darts_nds-spos_rankloss-epochs50-lr0.1-1269-0.sh

kubectl exec kyu-imagenet-long-2-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-pcdarts-baseline-original-darts-darts_nds-methoddarts_rankloss-epochs150-lr0.1-channel16-layers8-multiseed-0.sh
# one pcdarts running ere...


# GDAS test, this is very close to spos, should work.
kubectl exec kyu-imagenet-long-4-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-gdas-gdas-rankloss-lr-nasbench201-epochs400-lr0.025-multiseed-0.sh 
# only that...
kubectl exec kyu-imagenet-long-4-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-gdas-gdas-rankloss-lr-nasbench201-epochs400-lr0.01-multiseed-0.sh ; bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-gdas-gdas-rankloss-lr-nasbench201-epochs400-lr0.05-multiseed-0.sh 
kubectl exec kyu-imagenet-long-4-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-gdas-gdas-comparison-v1-nasbench201-darts-epochs410-lr0.025-multiseed-0.sh


############
# Queued
############


############
# finished
############
kubectl exec kyu-imagenet-long-5-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-imagenet-search-imagenet-pcdarts-darts_nds-lr0.1-spos-channels16-layers8epochs80-smooth0-1269-0.sh
kubectl exec -it kyu-imagenet-long-5-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-imagenet-search-imagenet-pretrained-darts_nds-lr0.1-spos_rankloss-channels16-layers8epochs150-smooth0-1269-0.sh

kubectl exec -it kyu-imagenet-long-5-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-imagenet-search-imagenet-pretrained-darts_nds-lr0.1-spos_rankloss-channels16-layers8epochs150-smooth0-1268-0.sh
kubectl exec kyu-imagenet-long-3-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-ldmk-sampling-distances-nasbench201-epochs200-lr0.025-distance-4-1270-0.sh &
kubectl exec kyu-imagenet-long-3-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-ldmk-sampling-distances-nasbench201-epochs200-lr0.025-distance-3-1270-0.sh & 
kubectl exec kyu-imagenet-long-3-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-ldmk-sampling-distances-nasbench201-epochs200-lr0.025-distance-2-1270-0.sh &
kubectl exec kyu-imagenet-long-3-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-ldmk-sampling-distances-nasbench201-epochs200-lr0.025-distance-1-1270-0.sh &


kubectl exec -it kyu-imagenet-long-2-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-imagenet-search-imagenet-epochs-darts_nds-lr0.1-spos_rankloss-channels16-layers8epochs100-smooth0-1269-0.sh
kubectl exec kyu-imagenet-long-2-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-ldmk-sampling-distances-nasbench201-epochs200-lr0.025-distance-4-1269-0.sh &
kubectl exec kyu-imagenet-long-2-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-ldmk-sampling-distances-nasbench201-epochs200-lr0.025-distance-3-1269-0.sh &
kubectl exec kyu-imagenet-long-2-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-ldmk-sampling-distances-nasbench201-epochs200-lr0.025-distance-2-1269-0.sh &
kubectl exec kyu-imagenet-long-2-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-ldmk-sampling-distances-nasbench201-epochs200-lr0.025-distance-1-1269-0.sh &

# seed 1269, pre-trained
kubectl exec kyu-imagenet-long-2-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-ldmk-sampling-distances-nasbench201-epochs200-lr0.025-distance-5-1269-0.sh

# seed 1270, pre-trained, first run this, until 150 epochs, then launch the second batch
kubectl exec kyu-imagenet-long-2-interactive -- bash /cvlabdata2/home/kyu/k8s_submit/test/kyu_k8s_job-ldmk-sampling-distances-nasbench201-epochs200-lr0.025-distance-5-1270-0.sh







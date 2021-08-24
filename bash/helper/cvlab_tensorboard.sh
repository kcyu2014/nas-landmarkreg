
HOST=cv27

# Environment things.
CONDA_ENV=tensorflow-cpu
WORK_DIR=/home/kyu/pycharm/automl
SUBMISSION_DIR=/home/kyu/pycharm/automl/slurm-submissions

PORT=9000
DAYS=1
CPUID=2

# Overwrite Configs accordingly.
[[ ! -z "$2" ]] && export PORT=$2

echo "Tensorboard running at cv27 http://10.90.43.16:${PORT}"

FILENAME=tensorboard-${1}-${PORT}
FILENAME=${FILENAME////-} 
CMD="/cvlabdata2/home/kyu/miniconda3/envs/tensorboard/bin/tensorboard --logdir /home/kyu/pycharm/automl/experiments/$1 --port ${PORT}  2>&1  | tee ${SUBMISSION_DIR}/${FILENAME}.out & disown"
# Disown the process is crucial to the command.

ssh kyu@${HOST} <<EOF
cd ${WORK_DIR}
echo "Working dir`pwd`"
echo "Running tensorboard for $1 at $PORT"
echo "#!/bin/bash" > /tmp/${FILENAME}.sh
printf "timeout ${DAYS}d bash -c '$CMD'" >> /tmp/${FILENAME}.sh
bash /tmp/${FILENAME}.sh
EOF

exit
#  Tracked results 
bash bash/helper/cvlab_tensorboard.sh cvpr-landmark-loss/ 9053
bash bash/helper/cvlab_tensorboard.sh cvpr-landmark-loss-ablation/ 9054
bash bash/helper/cvlab_tensorboard.sh darts-fromscratch/ 9055

bash bash/helper/cvlab_tensorboard.sh cvpr-landmark-loss-ablation/sampling-distances 9095
bash bash/helper/cvlab_tensorboard.sh cvpr-imagenet-fix-node/ 9100 &
bash bash/helper/cvlab_tensorboard.sh cvpr-gdas/ 9101 &
bash bash/helper/cvlab_tensorboard.sh cvpr-pcdarts/ 9102 &

bash bash/helper/cvlab_tensorboard.sh cvpr-imagenet/ 9949 &
bash bash/helper/cvlab_tensorboard.sh nips-landmark/ 9950
bash bash/helper/cvlab_tensorboard.sh nips-landmark/fromscratch-imagenet 9810 &
bash bash/helper/cvlab_tensorboard.sh nips-landmark/validate-coef 9951 &
bash bash/helper/cvlab_tensorboard.sh nips-landmark/node2-lossfn-infinite 9952 &
bash bash/helper/cvlab_tensorboard.sh nips-landmark/validate-lossfn-infinite 9955 &
bash bash/helper/cvlab_tensorboard.sh nips-landmark/validate-numarch 9956 &
bash bash/helper/cvlab_tensorboard.sh nips-landmark/validate-softplus-beta 9957 &
bash bash/helper/cvlab_tensorboard.sh nips-landmark/validate-lossfn 9958 &
bash bash/helper/cvlab_tensorboard.sh nips-landmark/baseline 9959 &
bash bash/helper/cvlab_tensorboard.sh nips-landmark/nasbench101-fix-channels 9960
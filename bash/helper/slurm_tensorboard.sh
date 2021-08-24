#!/usr/bin/env bash

# Try to submit job to SLURM cluster.
# From local computer.


###############################
## PLEASE put your info here
HOST=isl-iam1.rr.intel.com
USERNAME=rranftl
WORK_DIR=automl
TENSORBOARD_PATH=tensorboard
###############################

PORT=9000
DAYS=1
CPUID=2
SUBMISSION_DIR=$WORK_DIR/slurm-submissions

# Overwrite Configs accordingly.
[[ ! -z "$2" ]] && export PORT=$2


CMD="ifconfig && cd $WORK_DIR/experiments/$1 && $TENSORBOARD_PATH --logdir . --port ${PORT}"
FILENAME=tensorboard-${RUN_ID}

echo "Tensorboard running at isl-cpu${CPUID} http://10.14.219.15${CPUID}:${PORT}"

ssh ${USERNAME}@${HOST} <<EOF
cd ${WORK_DIR}
echo "#!/bin/bash" > /tmp/${FILENAME}.sh
printf "$CMD" >> /tmp/${FILENAME}.sh
echo "submit tensorboard job ... "
sbatch -p cpu -w isl-cpu${CPUID} -c 1 -t ${DAYS}-0 -o ${SUBMISSION_DIR}/tensorboard-%N.%j.out  /tmp/${FILENAME}.sh
EOF

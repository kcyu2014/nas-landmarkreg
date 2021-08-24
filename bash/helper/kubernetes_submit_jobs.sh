#!/usr/bin/env bash

# Try to submit job to SLURM cluster.
# From local computer.

HOST=cv15

# Environment things.
WORK_DIR=/home/kyu/pycharm/automl
RUN_ID=0
GPU=1
PARTITION=gpu
DAYS=6
TEMPLATE=None

if [[ `hostname` == *"MacBook-Pro"* ]]; then
PYTHON=/Users/kyu/anaconda/envs/web/bin/python
else
PYTHON=python
fi

# Overwrite Configs accordingly.
[[ ! -z "$2" ]] && export RUN_ID=$2
[[ ! -z "$3" ]] && export GPU=$3
[[ ! -z "$4" ]] && export PARTITION=$4
[[ ! -z "$5" ]] && export DAYS=$5
[[ ! -z "$6" ]] && export KBSNAME=$6
[[ ! -z "$7" ]] && export WORK_DIR=$7
[[ ! -z "$8" ]] && export TEMPLATE=$8


SUBMISSION_DIR='/cvlabdata2/home/kyu/k8s_submit/test/'
file=kyu_k8s_job-${KBSNAME}-${RUN_ID}.sh
file=${SUBMISSION_DIR}${file////-}
# echo $file
# echo $CMD

# Wrote the command to a remote location that is accessible

ssh kyu@${HOST} <<EOF
cd ${WORK_DIR}
echo "Working dir \`pwd\`"
echo "Result save to ${SUBMISSION_DIR}, running scripts ${file}."
echo "#!/bin/bash" > ${file}
printf "$1" >> ${file}
EOF

[[ $9 = 'delete' ]] && file='delete'
[[ $9 = 'nohup' ]] && file='nohup'
[[ ${10} = 'debug' ]] || [[ ${10} = 'debug' ]] && PARTITION='interactive'

$PYTHON k8s/submit_kbs_jobs.py --script ${file} --time $DAYS --gpu $GPU --job_name $KBSNAME --partition $PARTITION --runtype=job --job_template $TEMPLATE

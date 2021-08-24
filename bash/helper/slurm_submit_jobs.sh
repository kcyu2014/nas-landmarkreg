#!/usr/bin/env bash


###############################
## PLEASE put your info here
HOST=isl-iam1.rr.intel.com
USERNAME=rranftl
WORK_DIR=/home/rranftl/automl
###############################
# ###############################
# ## PLEASE put your info here
# HOST=cv15
# USERNAME=kyu
# WORK_DIR=/home/kyu/pycharm/automl
# ###############################

SUBMISSION_DIR=$WORK_DIR/slurm-submissions
RUN_ID=0
GPU=1
PARTITION=gpu
DAYS=3

# Overwrite Configs accordingly.
[[ ! -z "$2" ]] && export RUN_ID=$2
[[ ! -z "$3" ]] && export GPU=$3
[[ ! -z "$4" ]] && export PARTITION=$4
[[ ! -z "$5" ]] && export DAYS=$5
[[ ! -z "$6" ]] && export K8SNAME=$6
[[ ! -z "$7" ]] && echo "ignore 3 arg"
[[ ! -z "$8" ]] && echo "ignore 4 arg"
[[ ! -z "$9" ]] && export EXPERIMENT=$9

NUMCPU=$(( 10 * $GPU ))
echo "using partation: $PARTITION with $GPU GPUs and $NUMCPU CPUs"
K8SNAME=${K8SNAME////-}
file=/tmp/kyu_${K8SNAME}-${RUN_ID}.sh
# if [ "$4" = "quadro" ]; then
    # CMD="sbatch --parsable -p ${PARTITION} --gres=gpu:${GPU} -c ${NUMCPU} -t ${DAYS}-0 -o ${SUBMISSION_DIR}/slurm-%N.%j.out -e ${SUBMISSION_DIR}/slurm-%N.%j.err ${file}"
# else

if [ $DAYS -gt 2 ]; then
REPEAT=$((DAYS/2 - 1))
DAYS=2

CMD="sbatch --parsable -p ${PARTITION} --gres=gpu:${GPU} -c ${NUMCPU} -t ${DAYS}-0 -o ${SUBMISSION_DIR}/slurm-%N.%j.out -e ${SUBMISSION_DIR}/slurm-%N.%j.err "
else
REPEAT=0
CMD="sbatch --parsable -p ${PARTITION} --gres=gpu:${GPU} -c ${NUMCPU} -t ${DAYS}-0 -o ${SUBMISSION_DIR}/slurm-%N.%j.out -e ${SUBMISSION_DIR}/slurm-%N.%j.err ${file}"
fi

if [ $REPEAT -gt 0 ]; then
ssh ${USERNAME}@${HOST} <<EOF
cd ${WORK_DIR}
echo "Working dir`pwd`"
echo "Result save to ${SUBMISSION_DIR}"
echo "cmd is ${CMD}"
echo "Creating bash file for slurm ..."
echo "#!/bin/bash" > ${file}
printf "$1" >> ${file}
echo "Submit slurm jobs for ${REPEAT} times ..."
jid=\$(${CMD} ${file})
echo "\$jid is submitted" 
for ((n=0;n<${REPEAT};n++))
do 
    jid=\$(${CMD} --dependency=afterany:\$jid ${file})
    echo "\$jid is appending to the queue."
done
EOF

else
ssh ${USERNAME}@${HOST} <<EOF
cd ${WORK_DIR}
echo "Working dir`pwd`"
echo "Result save to ${SUBMISSION_DIR}"
echo "cmd is ${CMD}"
echo "Creating bash file for slurm ..."
echo "#!/bin/bash" > ${file}
printf "$1" >> ${file}
echo "Submit slurm jobs ..."
echo "${CMD}"
${CMD}
EOF
fi
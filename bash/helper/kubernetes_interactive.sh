# Run a interactive pod.

GPU=1
PARTITION=interactive
SCRIPT=debug

[[ ! -z "$2" ]] && export GPU=$2
[[ ! -z "$3" ]] && export SCRIPT=$3
# [[ ! -z "$4" ]] && [[ "$4" = 'delete' ]] && export 

python k8s/submit_kbs_jobs.py --script ${SCRIPT} --time 1 --gpu $GPU --job_name $1 --partition $PARTITION --runtype=pod --job_template $4 --force-delete


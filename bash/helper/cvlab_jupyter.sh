#!/usr/bin/env bash

# Try to submit job to SLURM cluster.
# From local computer.


HOST=cv27

# Environment things.
CONDA_ENV=pytorch-test
WORK_DIR=/home/kyu/pycharm
PORT=6123

# Overwrite Configs accordingly.
[[ ! -z "$1" ]] && export PORT=$1
[[ ! -z "$2" ]] && export GPU=$2 && echo "Using GPU ${2}"

if [[ ${GPU} -ge 1 ]];then
# using jupyter k8s. todo later.
echo "Jupyter-Notebook running at GPU node"
HOST=cv132
ssh kyu@${HOST} <<EOF
ifconfig
cd ${WORK_DIR}
echo "Working dir`pwd`"
NVIDIA_VISIBLE_DEVICES=3 /cvlabdata2/home/kyu/miniconda3/bin/jupyter notebook --port ${PORT}  --no-browser
EOF
echo "Jupyter-Notebook running at ${HOST} http://10.90.45.4:${PORT}/?token=6fc3a5b6a7eb129b7be201cc670edbb46d3fa821a11d01b0"
else
HOST=cv16
# CPU jupyter on cv16.
ssh kyu@${HOST} <<EOF
ifconfig
cd ${WORK_DIR}
echo "Working dir`pwd`"
echo "Running JupyterNotebook at ${PORT} "
/cvlabdata2/home/kyu/miniconda3/bin/jupyter notebook --port ${PORT}  --no-browser
EOF
echo "Jupyter-Notebook running at cv16 http://http://10.90.43.5:${PORT}/?token=6fc3a5b6a7eb129b7be201cc670edbb46d3fa821a11d01b0"
fi

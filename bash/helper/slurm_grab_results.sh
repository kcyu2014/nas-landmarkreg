#!/usr/bin/env bash

# Try to submit job to SLURM cluster.
# From local computer.

###############################
## PLEASE put your info here
HOST=isl-iam1.rr.intel.com
USERNAME=rranftl
WORK_DIR=/home/rranftl/automl
###############################

# Overwrite Configs accordingly.
[[ ! -z "$1" ]] && export EPATH=$1
ZIPNAME=${EPATH////-}

echo "Zip file under $EPATH to $ZIPNAME"

ssh ${USERNAME}@${HOST} <<EOF
cd ${WORK_DIR}
tar --exclude-from=exclude-pattern.txt -zcvf zip-experiments/${ZIPNAME}.tar ${EPATH}
EOF

echo "Copy the results back ..."
scp ${USERNAME}@${HOST}:$WORK_DIR/zip-experiments/$ZIPNAME.tar zip-experiments/$ZIPNAME.tar

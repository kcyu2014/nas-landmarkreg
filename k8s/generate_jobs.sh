#!/usr/bin/env bash

#if [ -d "multijobs" ]; then
#    rm -r ./multijobs
#fi
if [ ! -d "multijobs" ]; then
    mkdir ./multijobs
fi

if [ ! -z "$1" ];
then
    FILENAME=$1
else
    FILENAME='jinja_search_space_8.yaml'
fi

# delete the job
if [ -z "$2" ]; then
    echo "Creating the job from $FILENAME."
    cat $FILENAME | python -c "from jinja2 import Template; import sys; print(Template(sys.stdin.read()).render());" > ./multijobs/$FILENAME
    kubectl create -f ./multijobs/$FILENAME
else
    echo "Deleting job $FILENAME"
    kubectl delete -f ./multijobs/$FILENAME
fi



## Common command
## Create the job from file
#kubectl create -f $PATH_TO_YOUR_YAML
#
## Kill job by name
#kubectl delete pod $YOUR_POD_NAME
## Kill job by file
#kubectl delete -f $PATH_TO_YOUR_YAML
#
## Get the running jobs status
#kubectl get pods
#
## login your pod via bash
#kubectl exec -it $YOUR_POD_NAME -- bash

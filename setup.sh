!# /usr/bin/bash
# prepare the enviornment.

echo "Creating the enviornment"
conda env create -f icml-nasws-requirements.yml

echo "prepare the directory for aux"
mkdir logs
mkdir zip-experiments
mkdir slurm-submissions

echo "Link the data folder from my previous code env"
rm -r data
ln -s /home/yukaiche/pycharm/automl/data .

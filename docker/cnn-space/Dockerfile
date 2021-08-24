FROM ic-registry.epfl.ch/cvlab-k8s-master/pytorch:19.08

USER root
# Install Requirements

RUN pip install networkx tensorflow==1.12.0
RUN cd /tmp && git clone https://github.com/google-research/nasbench && cd nasbench && pip install -e .
# install the pytorch-1.2 use cuda 10.0
RUN conda install -y pytorch torchvision cudatoolkit=10.0 -c pytorch \
    && conda clean -ya

USER kyu

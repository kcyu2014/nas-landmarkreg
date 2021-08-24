FROM ic-registry.epfl.ch/cvlab-k8s-master/nvidia-pytorch-base:19.06-py3
MAINTAINER Kaicheng Yu <kaicheng.yu@epfl.ch>

RUN "sh" "-c" "echo nameserver 8.8.8.8 >> /etc/resolv.conf"

##########################
#####  Docmuemts #########
##########################

# Build a a light environment with the correct user access to cvlabdata3
# No root access originally, can sudo to get it
# Create to support k8s deployment and pod commmand
# Hope it can work!

RUN "sh" "-c" "echo nameserver 8.8.8.8 >> /etc/resolv.conf"

## user: root permission
USER root

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
 && rm -rf /var/lib/apt/lists/*

# Install python packages via conda.
ENV CONDA_AUTO_UPDATE_CONDA=false
RUN conda install -y "conda>=4.4.11" && conda clean -ya
#################################################
# ADAPT this for your own project!!!
RUN conda install -y \
        h5py \
        matplotlib \
        numpy \
        pillow \
        graphviz \
        pyyaml \
        protobuf \
        scipy \
        tqdm \
        && \
    conda clean -ya
RUN pip install \
        tensorboardX

# Create user with UID=NB_UID and in the 'NB_GID' group
# and make sure these dirs are writable by the `NB_GID` group.
# Configure user and group

#################################################
# CHANGE THIS PART ACCORDINGLY !!!
ENV SHELL=/bin/bash \
    NB_USER=kyu \
    NB_UID=168151 \
    NB_GROUP=CVLAB-unit \
    NB_GID=11166
#################################################

ENV HOME=/home/$NB_USER

RUN groupadd $NB_GROUP -g $NB_GID
RUN useradd -m -s /bin/bash -N -u $NB_UID -g $NB_GID $NB_USER && \
    echo "${NB_USER}:${NB_USER}" | chpasswd && \
    usermod -aG sudo,adm,root ${NB_USER}
RUN chown -R ${NB_USER}:${NB_GROUP} ${HOME}
#RUN chown $NB_USER:$NB_GID $CONDA_DIR
#################################################

# The user gets passwordless sudo
RUN echo "${NB_USER}   ALL = NOPASSWD: ALL" > /etc/sudoers


## user: user permission
USER $NB_USER

FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

# Install Requirements
RUN apt-get update && apt-get install -y build-essential \
                                         checkinstall \
                                         libreadline-gplv2-dev \
                                         libncursesw5-dev \
                                         libssl-dev \
                                         libsqlite3-dev \
                                         tk-dev \
                                         libgdbm-dev \
                                         libc6-dev \
                                         libbz2-dev \
                                         zlib1g-dev \
                                         openssl \
                                         libffi-dev \
                                         python3-dev \
                                         python3-setuptools \
                                         wget \
                                         git \
                                         default-jre \
                                         ipython \
                                         screen \
                                         ipython-notebook

# Prepare to build
RUN mkdir /tmp/Python36
WORKDIR /tmp/Python36

# Pull down Python 3.6, build, and install
RUN wget https://www.python.org/ftp/python/3.6.7/Python-3.6.7.tar.xz
RUN tar xvf Python-3.6.7.tar.xz
WORKDIR /tmp/Python36/Python-3.6.7
RUN ./configure
RUN make altinstall

WORKDIR /tmp
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python3.6 get-pip.py
RUN pip3 install --upgrade pip

WORKDIR /
RUN mkdir -p /persistent_volume
WORKDIR /

# Install packages
RUN pip3 install torch==0.3.1
RUN pip3 install tensorboardX
RUN pip3 install pandas==0.23.4
RUN pip3 install scikit-learn==0.20.0
RUN pip3 install ipdb
RUN pip3 install jupyter
RUN pip3 install plotly
RUN pip3 install --upgrade msgpack==0.5.6


ENV PYTHONPATH /code
ENV DATA /data

WORKDIR  /code

ENV TZ=Europe/Zurich
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# FIX local problem
RUN apt-get install -y locales
RUN locale-gen "en_US.UTF-8"
ENV LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 LANGUAGE=en_US.UTF-8

ENTRYPOINT ["/bin/bash"]

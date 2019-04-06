FROM nvidia/cuda
MAINTAINER Nicholas Bardy (nicholasbardy@gmail.com)
### Begin Setion 

# Install dependencies
RUN apt-get update
ENV DEBIAN_FRONTEND=non-interactive
RUN apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev \
libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
xz-utils tk-dev libffi-dev liblzma-dev

### End Seciton 

### End Seciton -- Install Conda

RUN apt-get install git -y

 UN apt-get install libsparsehash-dev
RUN apt-get install unzip

RUN apt-get clean

### Start Section --- Install Conda 
ENV PATH /opt/conda/bin:$PATH
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda2-5.3.0-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

### Add conda deps
RUN conda update conda
RUN conda install python=3.6
RUN conda install pytorch-nightly -c pytorch # See https://pytorch.org/get-started/locally/
RUN conda install -c anaconda pillow scipy future

### End - Add conda deps

# Clone Repo
WORKDIR ~
RUN git clone https://github.com/nbardy/SparseConvNet.git
WORKDIR SparseConvNet/

RUN rm -rf build/ dist/ sparseconvnet.egg-info sparseconvnet_SCN*.so
RUN python setup.py develop

FROM continuumio/miniconda3
MAINTAINER Nicholas Bardy (nicholasbardy@gmail.com)

#### Begin Setion Install Python3.6 from source

# Install dependencies
RUN apt-get update
RUN apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev \
libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
xz-utils tk-dev libffi-dev liblzma-dev

# Get From internet
RUN wget https://www.python.org/ftp/python/3.6.3/Python-3.6.3.tgz
RUN tar xvf Python-3.6.3.tgz
WORKDIR Python-3.6.3

# Run build Steps
RUN ./configure --enable-optimizations
RUN make -j8
RUN make altinstall

# Link as system python
RUN ln -s /usr/bin/python python3.6

### End Seciton --- Install Python3.6 from source


# Clone Repo
WORKDIR ~
RUN git clone https://github.com/facebookresearch/SparseConvNet.git
WORKDIR SparseConvNet/

RUN conda install pytorch-nightly -c pytorch # See https://pytorch.org/get-started/locally/
RUN conda install google-sparsehash -c bioconda
RUN apt-get install libsparsehash-dev unrar
RUN conda install -c anaconda pillow
RUN rm -rf build/ dist/ sparseconvnet.egg-info sparseconvnet_SCN*.so
RUN python setup.py develop

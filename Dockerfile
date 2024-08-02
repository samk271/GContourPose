# Use nvidia/cuda image
FROM nvidia/cuda:11.8.0-base-ubuntu20.04

# set bash as current shell
RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive 

# Install basic packages
RUN apt update && \
    apt install -y --no-install-recommends \
    build-essential wget 

    # Install miniconda
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
     /bin/bash ~/miniconda.sh -b -p $CONDA_DIR
ENV PATH=$CONDA_DIR/bin:$PATH

#Install dependencies and create environment
RUN conda clean -ya && conda update conda -y && conda install -n base conda-libmamba-solver -y && conda config --set solver libmamba
RUN conda config --set channel_priority flexible
RUN conda create -n GContourPose python=3.10
RUN pip install --upgrade pip
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install numpy matplotlib scipy pillow opencv-python
RUN apt-get update && apt-get install libgl1 ffmpeg libsm6 libxext6  -y

WORKDIR /gcontourpose

ENTRYPOINT ["python3", "main.py"]
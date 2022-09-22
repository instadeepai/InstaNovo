FROM nvcr.io/nvidia/tensorflow:21.06-tf2-py3

ARG VERSION
ARG LAST_COMMIT

ENV USER=app
ENV UID=42000
ENV GID=42001
ENV HOME_DIRECTORY=/app

# Ensure ARG are sets
RUN test -n "$VERSION" && test -n "$LAST_COMMIT"

# Update and upgrade your base image
RUN apt-get update && \
        apt-get upgrade -y

# Install required system dependencies and clear cache
RUN DEBIAN_FRONTEND=noninteractive apt-get install git \
        libcusolver10 \
        ca-certificates -y && \
        apt-get clean

# Create home directory
RUN mkdir -p ${HOME_DIRECTORY}

# Create group and user
RUN groupadd --force --gid $GID $USER && \
        useradd -M --home $HOME_DIRECTORY --base-dir $HOME_DIRECTORY \
        --uid $UID --gid $GID --shell "/bin/bash" $USER

# Copy files into HOME_DIRECTORY
COPY . $HOME_DIRECTORY

# Makes HOME_DIRECTORY files owned by USER
RUN chown -R ${USER}:${USER} ${HOME_DIRECTORY}

# Set HOME_DIRECTORY as default
WORKDIR $HOME_DIRECTORY

# Default user
USER $USER

# Add Python bin to PATH
ENV PATH=$PATH:$HOME_DIRECTORY/.local/bin

# Install python requirements
RUN pip install --upgrade --quiet pip setuptools && \
        pip install --no-cache-dir -r ./requirements.txt

# Set different env variables
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV VERSION=${VERSION}
ENV LAST_COMMIT=${LAST_COMMIT}

# Append the current directory to your python path
ENV PYTHONPATH=$PWD:$PYTHONPATH

# Install our pkg
RUN pip install --prefix ~/.local -e .

# Default Tensorboard port
EXPOSE 6006

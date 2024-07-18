FROM nvcr.io/nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# When your experiment writes result files into a bind mount (directories mirrored using -v or --volume),
# your results will (for most containers) be read only, because docker containers run as root by default.
# This means, to clear up your result files, you need to be a root user.
# To avoid this, you can modify your Dockerfile to use a non-root user.
ARG USER=app
ARG UID=1001
ARG GID=1000
ARG HOME_DIRECTORY=/app
ARG RUNS_DIRECTORY=/runs
ARG VENV_DIRECTORY=/opt/venv

ARG VERSION=latest
ARG LAST_COMMIT=latest

# Ensure ARGs are sets
RUN test -n "$USER" && \
        test -n "$UID" && \
        test -n "$GID" && \
        test -n "$HOME_DIRECTORY" && \
        test -n "$RUNS_DIRECTORY" && \
        test -n "$VERSION" && \
        test -n "$LAST_COMMIT"

# avoid unnecessary writes to disk
ENV LANG=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# set CUDA device IDS to be ordered by PCR bus IDs
# see https://stackoverflow.com/a/43131539
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID

# force JAX and TensorFlow to not pre-allocate all GPU memory
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false TF_FORCE_GPU_ALLOW_GROWTH=true

# prevents the "The CUDA linking API did not work" error
ENV XLA_FLAGS=--xla_gpu_force_compilation_parallelism=1

# enable terminal 256 colors
ENV TERM=xterm-256color

# set application working directory
ENV PYTHONPATH=$HOME_DIRECTORY:$PYTHONPATH
WORKDIR $HOME_DIRECTORY

# install packages
#   - python3 (3.10.12 for ubuntu 22.04)
#   - python3-pip
#   - git (log git info via neptune and install submodules, requires ssh)
#   - google-cloud-cli (requires ssh curl wget)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libaio-dev \
        curl \
        git \
        make \
        python3 \
        python3-pip \
        python3-venv \
        ssh \
        wget \
        && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends google-cloud-cli && \
    apt-get clean && \
    rm -rf /var/lib/{apt,dpkg,cache,log}


# Create group and user
RUN groupadd --force --gid $GID $USER && \
        useradd -M --home $HOME_DIRECTORY --base-dir $HOME_DIRECTORY \
        --uid $UID --gid $GID --shell "/bin/bash" $USER

# Create runs directory
RUN mkdir -p ${RUNS_DIRECTORY}
RUN mkdir -p ${VENV_DIRECTORY}

# Copy files into HOME_DIRECTORY
COPY . $HOME_DIRECTORY

# Makes HOME_DIRECTORY and RUNS_DIRECTORY files owned by USER
RUN chown -R ${USER}:${GID} ${HOME_DIRECTORY}
RUN chown -R ${USER}:${GID} ${RUNS_DIRECTORY}
RUN chown -R ${USER}:${GID} ${VENV_DIRECTORY}

# Set HOME_DIRECTORY as default
WORKDIR $HOME_DIRECTORY

# Default user
USER $USER

# Create virtual environment
ENV VIRTUAL_ENV=${VENV_DIRECTORY}
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install python requirements
RUN pip install uv && \
    UV_HTTP_TIMEOUT=600 uv pip install --python $(which python) \
                   --no-cache \
                   -r requirements/requirements.txt \
                   -r requirements/requirements-dev.txt

# Install InstaNovo
RUN uv pip install --python $(which python) -e .

# Set different env variables
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV VERSION=${VERSION}
ENV LAST_COMMIT=${LAST_COMMIT}

# Append the current directory to your python path
ENV PYTHONPATH=$PWD:$PYTHONPATH

# Default Tensorboard port
EXPOSE 6006

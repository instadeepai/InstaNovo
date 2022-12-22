FROM deepspeed/deepspeed:v072_torch112_cu117

# When your experiment writes result files into a bind mount (directories mirrored using -v or --volume),
# your results will (for most containers) be read only, because docker containers run as root by default.
# This means, to clear up your result files, you need to be a root user.
# To avoid this, you can modify your Dockerfile to use a non-root user.
ARG USER=app
ARG UID=42000
ARG GID=42001
ARG HOME_DIRECTORY=/app
ARG RUNS_DIRECTORY=/runs

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

# Update and upgrade your base image
RUN apt-get update && \
        apt-get upgrade -y

# Install required system dependencies and clear cache
RUN DEBIAN_FRONTEND=noninteractive apt-get install git \
        ca-certificates -y && \
        apt-get clean

# Downloading gcloud package
RUN curl https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz > /tmp/google-cloud-sdk.tar.gz

# Installing the gcloud package
RUN mkdir -p /usr/local/gcloud \
  && tar -C /usr/local/gcloud -xvf /tmp/google-cloud-sdk.tar.gz \
  && /usr/local/gcloud/google-cloud-sdk/install.sh

# Adding the gcloud mapackage path to local
ENV PATH $PATH:/usr/local/gcloud/google-cloud-sdk/bin

# Create group and user
RUN groupadd --force --gid $GID $USER && \
        useradd -M --home $HOME_DIRECTORY --base-dir $HOME_DIRECTORY \
        --uid $UID --gid $GID --shell "/bin/bash" $USER

# Create runs directory
RUN mkdir -p ${RUNS_DIRECTORY}

# Copy files into HOME_DIRECTORY
COPY . $HOME_DIRECTORY

# Makes HOME_DIRECTORY and RUNS_DIRECTORY files owned by USER
RUN chown -R ${USER}:${USER} ${HOME_DIRECTORY}
RUN chown -R ${USER}:${USER} ${RUNS_DIRECTORY}

# Set HOME_DIRECTORY as default
WORKDIR $HOME_DIRECTORY

# Default user
USER $USER

# Add Python and conda bin locations to PATH
ENV PATH=$PATH:$HOME_DIRECTORY/.local/bin:/opt/conda/bin

# Install python requirements
RUN /opt/conda/bin/pip install --upgrade --quiet pip setuptools && \
        /opt/conda/bin/pip install --no-cache-dir -r ./requirements.txt

# Set different env variables
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV VERSION=${VERSION}
ENV LAST_COMMIT=${LAST_COMMIT}

# Append the current directory to your python path
ENV PYTHONPATH=$PWD:$PYTHONPATH

# Print the DeepSpeed environment report to validate the DeepSpeed install and
# see which extensions/ops this machine is compatible with.
RUN ds_report

# Install our pkg
RUN pip install --prefix ~/.local -e .

# Default Tensorboard port
EXPOSE 6006

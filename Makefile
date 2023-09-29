# This Makefile provides shortcut commands to facilitate local development.

# Common variables
PACKAGE_NAME = dtu_denovo_sequencing

# Train variables
NUM_NODES = 1
BATCH_SIZE = 12
NUM_GPUS:= $(shell python -m dtu_denovo_sequencing.utils.parse_nr_gpus)


# LAST_COMMIT returns the current HEAD commit
LAST_COMMIT = $(shell git rev-parse --short HEAD)

# VERSION represents a clear statement of which tag based version of the repository you're actually running.
# If you run a tag based version, it returns the according HEAD tag, otherwise it returns:
# * `LAST_COMMIT-staging` if no tags exist
# * `BASED_TAG-SHORT_SHA_COMMIT-staging` if a previous tag exist
VERSION := $(shell git describe --always --exact-match --abbrev=0 --tags $(LAST_COMMIT) 2> /dev/null)
ifndef VERSION
	BASED_VERSION := $(shell git describe --always --abbrev=3 --tags $(git rev-list --tags --max-count=1))
	ifndef BASED_VERSION
	VERSION = $(LAST_COMMIT)-staging
	else
	VERSION = $(BASED_VERSION)-staging
	endif
endif

# Docker variables
DOCKER_HOME_DIRECTORY = "/app"
DOCKER_RUNS_DIRECTORY = "/runs"

DOCKERFILE := Dockerfile
DOCKERFILE_DEV := Dockerfile.dev
DOCKERFILE_CI := Dockerfile.ci

DOCKER_IMAGE_NAME = registry.gitlab.com/instadeep/dtu-denovo-sequencing
DOCKER_IMAGE_TAG = $(VERSION)
DOCKER_IMAGE = $(DOCKER_IMAGE_NAME):$(DOCKER_IMAGE_TAG)
DOCKER_IMAGE_DEV = $(DOCKER_IMAGE_NAME):$(DOCKER_IMAGE_TAG)-dev
DOCKER_IMAGE_CI = $(DOCKER_IMAGE_NAME):$(DOCKER_IMAGE_TAG)
DOCKER_IMAGE_CI_DEV = $(DOCKER_IMAGE_NAME):$(DOCKER_IMAGE_TAG)-dev

DOCKER_RUN_FLAGS = --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864
DOCKER_RUN_FLAGS_VOLUME_MOUNT_HOME = $(DOCKER_RUN_FLAGS) --volume $(PWD):$(DOCKER_HOME_DIRECTORY)
DOCKER_RUN_FLAGS_VOLUME_MOUNT_RUNS = $(DOCKER_RUN_FLAGS) --volume $(PWD)/runs:$(DOCKER_RUNS_DIRECTORY)
DOCKER_RUN = docker run $(DOCKER_RUN_FLAGS) $(IMAGE_NAME)



# Build commands

.PHONY: build build-arm build-dev build-dev-arm build-ci build-ci-dev

define docker_buildx_dev_template
	docker buildx build --platform=$(1) --progress=plain . \
		-f $(DOCKERFILE_DEV) -t $(2) --build-arg GID=$(shell id -g) \
		--build-arg UID=$(shell id -u) --build-arg LAST_COMMIT=$(LAST_COMMIT) \
		--build-arg VERSION=$(VERSION) --build-arg HOME_DIRECTORY=$(DOCKER_HOME_DIRECTORY) \
		--build-arg RUNS_DIRECTORY=$(DOCKER_RUNS_DIRECTORY)
endef

define docker_buildx_template
	docker buildx build --platform=$(1) --progress=plain . \
		-f $(DOCKERFILE) -t $(2) --build-arg GID=$(shell id -g) \
		--build-arg UID=$(shell id -u)  --build-arg LAST_COMMIT=$(LAST_COMMIT) \
		--build-arg VERSION=$(VERSION) --build-arg HOME_DIRECTORY=$(DOCKER_HOME_DIRECTORY) \
		--build-arg RUNS_DIRECTORY=$(DOCKER_RUNS_DIRECTORY)
endef

define docker_build_ci_template
	docker build --progress=plain . -f $(1) -t $(2) \
		--build-arg LAST_COMMIT=$(LAST_COMMIT) --build-arg VERSION=$(VERSION)
endef

build:
	$(call docker_buildx_template,linux/amd64,$(DOCKER_IMAGE))

build-arm:
	$(call docker_buildx_template,linux/arm64,$(DOCKER_IMAGE))

build-dev:
	$(call docker_buildx_dev_template,linux/amd64,$(DOCKER_IMAGE_DEV))

build-dev-arm:
	$(call docker_buildx_dev_template,linux/arm64,$(DOCKER_IMAGE_DEV))

build-ci:
	$(call docker_build_ci_template,$(DOCKERFILE),$(DOCKER_IMAGE_CI))
	docker tag $(DOCKER_IMAGE_CI) $(DOCKER_IMAGE)

build-ci-dev:
	$(call docker_build_ci_template,$(DOCKERFILE_DEV),$(DOCKER_IMAGE_CI_DEV))
	docker tag $(DOCKER_IMAGE_CI_DEV) $(DOCKER_IMAGE_DEV)

# Push commands

.PHONY: push-ci push-ci-dev

push-ci:
	docker push $(DOCKER_IMAGE)
	docker push $(DOCKER_IMAGE_CI)

push-ci-dev:
	docker push $(DOCKER_IMAGE_DEV)
	docker push $(DOCKER_IMAGE_CI_DEV)

push-ci-gitlab:
	docker push $(DOCKER_IMAGE_CI)

# Dev commands

.PHONY: test bash-dev docs-build

test: build-dev
	docker run --rm $(DOCKER_IMAGE_DEV) pytest --verbose $(PACKAGE_NAME)

bash:
	docker run -it $(DOCKER_RUN_FLAGS) $(DOCKER_IMAGE) /bin/bash

bash-dev: build-dev
	docker run -it $(DOCKER_RUN_FLAGS) $(DOCKER_IMAGE_DEV) /bin/bash

docs: build-dev
	docker run $(DOCKER_RUN_FLAGS) -p 8000:8000 $(DOCKER_IMAGE) mkdocs serve

set-gcp-credentials:
	python3 -m dtu_denovo_sequencing.utils.set_gcp_credentials
	gcloud auth activate-service-account dtu-denovo-sa@ext-dtu-denovo-sequencing-gcp.iam.gserviceaccount.com --key-file=ext-dtu-denovo-sequencing-gcp.json --project=ext-dtu-denovo-sequencing-gcp

# Train commands

.PHONY: train train-dev

train:
	deepspeed \
		--num_nodes=$(NUM_NODES) \
		--num_gpus=$(NUM_GPUS) \
		./dtu_denovo_sequencing/transnovo/train.py \
		train_data_path=./data/denovo_dataset_v1/ \
		batch_size=$(BATCH_SIZE) \
		distributed.n_gpus_per_node=$(NUM_GPUS) \
		--deepspeed \
		--deepspeed_config=deepspeed_cfg.json



train-docker:
	time docker run -it $(DOCKER_RUN_FLAGS_VOLUME_MOUNT_RUNS) $(DOCKER_IMAGE) \
		deepspeed \
			--num_nodes=$(NUM_NODES) \
			--num_gpus=$(NUM_GPUS) \
			./dtu_denovo_sequencing/train.py \
			train_data_path=./data/denovo_dataset_v1/ \
			batch_size=$(BATCH_SIZE) \
			distributed.n_gpus_per_node=$(NUM_GPUS) \
			--deepspeed \
			--deepspeed_config=deepspeed_cfg.json

train-docker-dev:
	time docker run -it $(DOCKER_RUN_FLAGS_VOLUME_MOUNT_RUNS) $(DOCKER_IMAGE_DEV) \
		deepspeed \
		--num_nodes=$(NUM_NODES) \
		./dtu_denovo_sequencing/train.py \
		checkpoint_path=$(DOCKER_RUNS_DIRECTORY) \
		train_data_path=./data/denovo_dataset_v1/ \
		batch_size=$(BATCH_SIZE) \
		--deepspeed \
		--deepspeed_config=deepspeed_cfg.json

# Download dataset commands

.PHONY: download_dataset_v1 download_dataset_v2

download_dataset_v1:
	mkdir -p ./data/denovo_dataset_v1
	gsutil -m cp -R gs://denovo_dataset_v1/ ./data

download_dataset_v2:
	mkdir -p ./data/denovo_dataset_v2
	gsutil -m cp -R gs://denovo_dataset_v2/ ./data


# MLFlow commands

MLFLOW_IMAGE_NAME=mlflow-gcp
GCP_PROJECT=ext-dtu-denovo-sequencing-gcp
ARTIFACT_REGISTRY=europe-west6-docker.pkg.dev
VERSION=latest


mlflow-auth:
	gcloud auth login && gcloud config set project ${GCP_PROJECT} && gcloud auth configure-docker ${ARTIFACT_REGISTRY}

mlflow-build:
	docker build -t "${MLFLOW_IMAGE_NAME}" --file Dockerfile.mlflow .

mlflow-tag:
	docker tag "${MLFLOW_IMAGE_NAME}" "${ARTIFACT_REGISTRY}/${GCP_PROJECT}/mlflow/${MLFLOW_IMAGE_NAME}:${VERSION}"

mlflow-push:
	docker push "${ARTIFACT_REGISTRY}/${GCP_PROJECT}/mlflow/${MLFLOW_IMAGE_NAME}:${VERSION}"

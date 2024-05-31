# This Makefile provides shortcut commands to facilitate local development.

# Common variables
PACKAGE_NAME = instanovo

# Train variables
NUM_NODES = 1
BATCH_SIZE = 12
NUM_GPUS:= $(shell python -m instanovo.utils.parse_nr_gpus)


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

DOCKER_RUN_FLAGS = --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --shm-size='1gb'
DOCKER_RUN_FLAGS_VOLUME_MOUNT_HOME = $(DOCKER_RUN_FLAGS) --volume $(PWD):$(DOCKER_HOME_DIRECTORY)
DOCKER_RUN_FLAGS_VOLUME_MOUNT_RUNS = $(DOCKER_RUN_FLAGS) --volume $(PWD)/runs:$(DOCKER_RUNS_DIRECTORY)
DOCKER_RUN = docker run $(DOCKER_RUN_FLAGS) $(IMAGE_NAME)

PYTEST = pytest --alluredir=allure_results --cov-report=html --cov --cov-config=.coveragerc --random-order --verbose .
COVERAGE = coverage report -m

#################################################################################
## Docker build commands																#
#################################################################################

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

## Build Docker image for InstaNovo on AMD64
build:
	$(call docker_buildx_template,linux/amd64,$(DOCKER_IMAGE))

## Build Docker image for InstaNovo on ARM64
build-arm:
	$(call docker_buildx_template,linux/arm64,$(DOCKER_IMAGE))

## Build development Docker image for InstaNovo on AMD64
build-dev:
	$(call docker_buildx_dev_template,linux/amd64,$(DOCKER_IMAGE_DEV))

## Build development Docker image for InstaNovo on ARM64
build-dev-arm:
	$(call docker_buildx_dev_template,linux/arm64,$(DOCKER_IMAGE_DEV))

## Build continuous integration Docker image for InstaNovo
build-ci:
	$(call docker_build_ci_template,$(DOCKERFILE),$(DOCKER_IMAGE_CI))
	docker tag $(DOCKER_IMAGE_CI) $(DOCKER_IMAGE)

## Build development continuous integration Docker image for InstaNovo
build-ci-dev:
	$(call docker_build_ci_template,$(DOCKERFILE_DEV),$(DOCKER_IMAGE_CI_DEV))
	docker tag $(DOCKER_IMAGE_CI_DEV) $(DOCKER_IMAGE_DEV)

#################################################################################
## Docker push commands																#
#################################################################################

.PHONY: push-ci push-ci-dev

## Push default and continuous integration Docker images for InstaNovo to GitLab registry
push-ci:
	docker push $(DOCKER_IMAGE)
	docker push $(DOCKER_IMAGE_CI)

## Push development and continuous integration development Docker images for InstaNovo to GitLab registry
push-ci-dev:
	docker push $(DOCKER_IMAGE_DEV)
	docker push $(DOCKER_IMAGE_CI_DEV)

#################################################################################
## Install packages commands																 	#
#################################################################################

.PHONY: compile install install-dev install-all

## Compile all the pinned requirements*.txt files from the unpinned requirements*.in files
compile:
	pip install --upgrade uv
	rm -f requirements/*.txt
	uv pip compile requirements/requirements.in --emit-index-url  --output-file=requirements/requirements.txt --extra-index-url https://pypi.cs.uni-tuebingen.de/simple/pyopenms/
	uv pip compile requirements/requirements-dev.in --output-file=requirements/requirements-dev.txt
	uv pip compile requirements/requirements-docs.in --output-file=requirements/requirements-docs.txt
	uv pip compile requirements/requirements-mlflow.in --output-file=requirements/requirements-mlflow.txt

## Install required packages
install:
	pip install --upgrade uv
	uv pip install -r requirements/requirements.txt

## Install required and development packages
install-dev:
	pip install --upgrade uv
	uv pip install -r requirements/requirements.txt \
	               -r requirements/requirements-dev.txt

## Install required, development, documentation and MLFlow packages
install-all:
	pip install --upgrade uv
	uv pip install -r requirements/requirements.txt \
	               -r requirements/requirements-dev.txt \
				   -r requirements/requirements-docs.txt \
				   -r requirements/requirements-mlflow.txt


##  Sync pinned dependencies with your virtual environment
sync:
	pip install --upgrade uv
	uv pip sync requirements/requirements.txt


#################################################################################
## Development commands																 	#
#################################################################################

.PHONY: tests coverage test-docker coverage-docker bash bash-dev docs set-gcp-credentials

## Run all tests
tests:
	$(PYTEST)

## Calculate the code coverage
coverage:
	$(COVERAGE)

## Run all tests in the development Docker Image
test-docker:
	docker run $(DOCKER_RUN_FLAGS) $(DOCKER_IMAGE_DEV) nvidia-smi && $(PYTEST)

## Calculate the code coverage in the development Docker image
coverage-docker:
	docker run $(DOCKER_RUN_FLAGS) $(DOCKER_IMAGE_DEV) nvidia-smi && $(PYTEST) && $(COVERAGE)

## Open a bash shell in the default Docker image
bash:
	docker run -it $(DOCKER_RUN_FLAGS) $(DOCKER_IMAGE) /bin/bash

## Open a bash shell in the development Docker image
bash-dev:
	docker run -it $(DOCKER_RUN_FLAGS) $(DOCKER_IMAGE_DEV) /bin/bash

## Serve the documentation site locally
docs:
	pip install --upgrade uv
	uv pip install -r requirements/requirements-docs.txt
	git config --global --add safe.directory "$(dirname "$(pwd)")"
	rm -rf docs/reference
	python ./docs/gen_ref_nav.py
	mkdocs build --verbose --site-dir docs_public
	mkdocs serve

## Set the GCP credentials
set-gcp-credentials:
	python -m instanovo.utils.set_gcp_credentials
	gcloud auth activate-service-account dtu-denovo-sa@ext-dtu-denovo-sequencing-gcp.iam.gserviceaccount.com --key-file=ext-dtu-denovo-sequencing-gcp.json --project=ext-dtu-denovo-sequencing-gcp

#################################################################################
## Train commands																#
#################################################################################

.PHONY: train_acpt train_extended finetune_on_hcpt finetune_on_phospho eval_abhi

## Train InstaNovo on AC-PT
train_acpt:
	mkdir -p ./data/ac_pt_shards
	gsutil -m cp -R gs://ac_pt_shards/ ./data
	python -m instanovo.transformer.train \
		--config-name instanovo

## Train InstaNovo on AC-PT, Phospho and PRIDE data
train_extended:
	mkdir -p ./data/denovo_formatted_ipc
	gsutil -m cp -R gs://denovo_formatted_ipc/ ./data
	mkdir -p ./data/extended
	cp -R ./data/denovo_formatted_ipc/ac_pt_shards/*.ipc ./data/extended
	python ./scripts/move_shards.py ./data/denovo_formatted_ipc/pride_extended/ ./data/extended/ 100
	python ./scripts/move_shards.py ./data/denovo_formatted_ipc/phospho/ ./data/extended/ 200
	python -m instanovo.transformer.train \
		--config-name instanovo_extended

## Finetune InstaNovo on HC-PT data
finetune_on_hcpt:
	mkdir -p ./data/denovo_dataset_v1_ipc
	gsutil -m cp -R gs://denovo_dataset_v1_ipc/ ./data
	mkdir -p ./checkpoints/acpt_ba75cf85
	gsutil cp "gs://denovo_checkpoints/epoch=2-step=2000000.ckpt" ./checkpoints/acpt_ba75cf85/
	python -m instanovo.transformer.train \
		--config-name instanovo_finetune_hcpt

## Finetune InstaNovo on phospho data
finetune_on_phospho:
	mkdir -p ./data/denovo_phospho
	gsutil -m cp -R gs://denovo_phospho/ ./data
#	mkdir -p ./checkpoints/acpt_ba75cf85
#	gsutil cp "gs://denovo_checkpoints/acpt_ba75cf85/epoch=9-step=8750000.ckpt" ./checkpoints/acpt_ba75cf85/
	mkdir -p ./checkpoints
	aws s3 cp s3://dtu-denovo-s-2e6da747d6d34f62-outputs/output/1358365a-9225-4a99-9354-9a0738d23eaa/checkpoints/instanovo-base/epoch=4-step=3400000.ckpt ./checkpoints/model.ckpt --endpoint-url https://storage.googleapis.com
	python -m instanovo.transformer.train \
		--config-name instanovo_phospho

## Evaluate InstaNovo on data from Abhi
eval_abhi:
	mkdir -p ./data/instanovo_hlaI_pred
	gsutil -m cp -R gs://biondeep-data/instanovo/instanovo_hlaI_pred/ ./data
	python -m instanovo.utils.convert_to_ipc \
		./data/instanovo_hlaI_pred/ \
		./data/instanovo_hlaI_pred.ipc \
		--source_type mzxml --verbose
	gsutil cp gs://denovo_checkpoints/acpt_d093a745/epoch\=8-step\=6300000.ckpt ./checkpoints/acpt_d093a745/
	python -m instanovo.transformer.predict \
		./data/instanovo_hlaI_pred.ipc \
		~/Coding/dtu-denovo-sequencing/checkpoints/acpt_d093a745/epoch\=8-step\=6300000.ckpt \
		-o instanovo_hlaI_pred.csv \
		-n -w 8 -b
	gsutil cp instanovo_hlaI_pred.csv gs://denovo_checkpoints/


#################################################################################
## Download dataset commands													#
#################################################################################

.PHONY: download_dataset_v1 download_dataset_v2

## Download dataset v1
download_dataset_v1:
	mkdir -p ./data/denovo_dataset_v1
	gsutil -m cp -R gs://denovo_dataset_v1/ ./data

## Download dataset v2
download_dataset_v2:
	mkdir -p ./data/denovo_dataset_v2
	gsutil -m cp -R gs://denovo_dataset_v2/ ./data



#################################################################################
## MLFlow commands																#
#################################################################################

.PHONY: mlflow-auth mlflow-build mlflow-tag mlflow-push

MLFLOW_IMAGE_NAME=mlflow-gcp
GCP_PROJECT=ext-dtu-denovo-sequencing-gcp
ARTIFACT_REGISTRY=europe-west6-docker.pkg.dev
VERSION=latest

## Authentication for MLFlow
mlflow-auth:
	gcloud auth login && gcloud config set project ${GCP_PROJECT} && gcloud auth configure-docker ${ARTIFACT_REGISTRY}

## Build Docker image for MLFlow
mlflow-build:
	docker build -t "${MLFLOW_IMAGE_NAME}" --file Dockerfile.mlflow .

## Tag Docker image for MLFlow
mlflow-tag:
	docker tag "${MLFLOW_IMAGE_NAME}" "${ARTIFACT_REGISTRY}/${GCP_PROJECT}/mlflow/${MLFLOW_IMAGE_NAME}:${VERSION}"

## Push Docker image for MLFlow
mlflow-push:
	docker push "${ARTIFACT_REGISTRY}/${GCP_PROJECT}/mlflow/${MLFLOW_IMAGE_NAME}:${VERSION}"

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')

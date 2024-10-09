# This Makefile provides shortcut commands to facilitate local development.

# Common variables
PACKAGE_NAME = instanovo

# Train variables
NUM_NODES = 1
BATCH_SIZE = 12
NUM_GPUS:= 1


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

## Install required packages
install:
	pip install --upgrade uv
	uv pip install -r requirements/requirements.txt

## Install required and development packages
install-dev:
	pip install --upgrade uv
	uv pip install -r requirements/requirements.txt \
	               -r requirements/requirements-dev.txt

## Install required, development and documentation packages
install-all:
	pip install --upgrade uv
	uv pip install -r requirements/requirements.txt \
	               -r requirements/requirements-dev.txt \
				   -r requirements/requirements-docs.txt \


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
	python -m instanovo.scripts.get_zenodo_record
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

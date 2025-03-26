# This Makefile provides shortcut commands to facilitate local development.

# Common variables
PACKAGE_NAME = instanovo

# Train variables
NUM_NODES = 1
BATCH_SIZE = 12
NUM_GPUS:= $(shell uv run --quiet --with PyYAML scripts/parse_nr_gpus.py)


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

PYTEST = python -m pytest --alluredir=allure_results --cov-report=html --cov --random-order --verbose .
COVERAGE = python -m coverage report -m

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

PHONY: install install-all upgrade

## Install required and development packages
install:
	uv sync --extra cu124
	uv run pre-commit install

## Install required, development and documentation packages
install-all:
	uv sync --extra cu124 --group docs

# Upgrade all packages
upgrade:
	uv lock --upgrade
	uv sync --extra cu124

#################################################################################
## Development commands															#
#################################################################################

.PHONY: tests coverage test-docker coverage-docker bash bash-dev docs set-gcp-credentials add-kyber-pvc-data

## Run all tests
tests:
	uv run instanovo/scripts/get_zenodo_record.py
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
	uv sync --extra cu124 --group docs
	git config --global --add safe.directory "$(dirname "$(pwd)")"
	rm -rf docs/reference
	python ./docs/gen_ref_nav.py
	mkdocs build --verbose --site-dir docs_public
	mkdocs serve

## Set the GCP credentials
set-gcp-credentials:
	uv run instanovo/scripts/set_gcp_credentials.py
	gcloud auth activate-service-account dtu-denovo-sa@ext-dtu-denovo-sequencing-gcp.iam.gserviceaccount.com --key-file=ext-dtu-denovo-sequencing-gcp.json --project=ext-dtu-denovo-sequencing-gcp

## Add data to Kyber PVC
add-kyber-pvc-data:
	mkdir -p /mnt/instanovo-data-kyber/identity_splits_parquet/acpt
	mkdir -p /mnt/instanovo-data-kyber/identity_splits_parquet/massivekb
	mkdir -p /mnt/instanovo-data-kyber/identity_splits_parquet/phospho
	mkdir -p /mnt/instanovo-data-kyber/identity_splits_parquet/pride

	gsutil -m cp -R gs://denovo_formatted_ipc/identity_splits_parquet/acpt/*.parquet /mnt/instanovo-data-kyber/identity_splits_parquet/acpt
	gsutil -m cp -R gs://denovo_formatted_ipc/identity_splits_parquet/massivekb/*.parquet /mnt/instanovo-data-kyber/identity_splits_parquet/massivekb
	gsutil -m cp -R gs://denovo_formatted_ipc/identity_splits_parquet/phospho/*.parquet /mnt/instanovo-data-kyber/identity_splits_parquet/phospho
	gsutil -m cp -R gs://denovo_formatted_ipc/identity_splits_parquet/pride/*.parquet /mnt/instanovo-data-kyber/identity_splits_parquet/pride

	ls -lR /mnt/instanovo-data-kyber/identity_splits_parquet


#################################################################################
## Train commands																#
#################################################################################

.PHONY: train_acpt train_extended train_extended_massive train_nine_species_v1 train_nine_species_v2 train_instanovoplus_acpt train_instanovoplus_extended_massive finetune_on_hcpt finetune_on_phospho finetune_on_nine_species_v2 ft_eval_nine_species_v2 zs_eval_nine_species_v2 eval_abhi eval_species_zero_shot eval_instanovoplus_acpt

## Train InstaNovo+ on AC-PT
train_instanovoplus_acpt:
	instanovo diffusion train --config-name instanovoplus

## Train InstaNovo+ on Extended + Massive-KB
train_instanovoplus_extended_massive:
	instanovo diffusion train --config-name instanovoplus_extended_massive

## Train InstaNovo on AC-PT
train_acpt:
	mkdir -p ./data/ac_pt_shards
	instanovo transformer train --config-name instanovo

## Train InstaNovo on AC-PT, Phospho and PRIDE data
train_extended:
	instanovo transformer train --config-name instanovo_extended

train_extended_massive:
	instanovo transformer train --config-name instanovo_extended_massive

## Train InstaNovo on nine species v1 data
train_nine_species_v1:
	mkdir -p ./data/new_schema
	gsutil -m cp -R gs://nine_species_dataset/species/exc_yeast_ipc/new_schema/*.ipc ./data/new_schema
	instanovo transformer train --config-name instanovo_nine_species_v1

## Train InstaNovo on nine species v2 data
train_nine_species_v2:
	mkdir -p ./data/species_formatted_ipc
	gsutil -m cp -R gs://nine_species_dataset_v2/species_formatted_ipc/*.ipc ./data/species_formatted_ipc

	gsutil -m cp "gs://nine_species_dataset_v2/species_formatted_ipc/species_split.csv" ./data/species_formatted_ipc

	uv run scripts/splits_and_shards.py \
		./data/species_formatted_ipc \
		--holdout_file_path "./data/species_formatted_ipc/saccharomyces_cerevisiae.ipc"
		--split_csv_path "./data/species_formatted_ipc/species_split.csv" \
		--check_split True

	instanovo transformer train --config-name instanovo_nine_species_v2

## Finetune InstaNovo on HC-PT data
finetune_on_hcpt:
	mkdir -p ./data/denovo_dataset_v1_ipc
	gsutil -m cp -R gs://denovo_dataset_v1_ipc/ ./data
	mkdir -p ./checkpoints/acpt_ba75cf85
	gsutil cp "gs://denovo_checkpoints/epoch=2-step=2000000.ckpt" ./checkpoints/acpt_ba75cf85/
	instanovo transformer train --config-name instanovo_finetune_hcpt

## Finetune InstaNovo on phospho data
finetune_on_phospho:
	mkdir -p ./data/denovo_phospho
	gsutil -m cp -R gs://denovo_phospho/ ./data
#	mkdir -p ./checkpoints/acpt_ba75cf85
#	gsutil cp "gs://denovo_checkpoints/acpt_ba75cf85/epoch=9-step=8750000.ckpt" ./checkpoints/acpt_ba75cf85/
	mkdir -p ./checkpoints

# 	aws s3 cp s3://dtu-denovo-s-2e6da747d6d34f62-outputs/output/1358365a-9225-4a99-9354-9a0738d23eaa/checkpoints/instanovo-base/epoch=4-step=3400000.ckpt ./checkpoints/model.ckpt --endpoint-url https://storage.googleapis.com

	gsutil cp gs://denovo_checkpoints/acpt_d093a745/epoch\=8-step\=6300000.ckpt ./checkpoints/model.ckpt

	instanovo transformer train --config-name instanovo_phospho

## Finetune InstaNovo on nine species v2 data
finetune_on_nine_species_v2:
	mkdir -p ./data/species_formatted_ipc
	gsutil -m cp -R gs://nine_species_dataset_v2/species_formatted_ipc/*.ipc ./data/species_formatted_ipc

	gsutil -m cp "gs://nine_species_dataset_v2/species_formatted_ipc/species_split.csv" ./data/species_formatted_ipc

	mkdir -p ./checkpoints
	gsutil cp gs://denovo_checkpoints/extended_38aa4b76/epoch=3-step=800000.ckpt ./checkpoints/model.ckpt

	uv run scripts/splits_and_shards.py \
		./data/species_formatted_ipc \
		--holdout_file_path "./data/species_formatted_ipc/saccharomyces_cerevisiae.ipc" \
		--split_csv_path "./data/species_formatted_ipc/species_split.csv" \
		--check_split True

	instanovo transformer train --config-name instanovo_nine_species_v2

## Evaluate InstaNovo+ on AC-PT
eval_instanovoplus_acpt:
	mkdir -p ./checkpoints
	gsutil -m cp -R gs://denovo_checkpoints/acpt_diffusion_25e04470 ./checkpoints
	ls -la ./checkpoints/acpt_diffusion_25e04470
	instanovo diffusion predict --config-name instanovoplus \
		--instanovo-plus-model=./checkpoints/acpt_diffusion_25e04470


## Evaluate InstaNovo on nine species v2 data from fine-tuned checkpoint
ft_eval_nine_species_v2:
	mkdir -p ./data/species_formatted_ipc
	gsutil -m cp -R gs://nine_species_dataset_v2/species_formatted_ipc/*.ipc ./data/species_formatted_ipc/

	mkdir -p ./checkpoints
	gsutil cp gs://denovo_checkpoints/ft_v2_yeast_3cab501c/epoch-0-step-2000.ckpt ./checkpoints/model.ckpt

	instanovo transformer predict \
		--data-path=./data/species_formatted_ipc/saccharomyces_cerevisiae.ipc \
		--instanovo-model=./checkpoints/model.ckpt \
		--output-path=saccharomyces_cerevisiae.csv \
		subset=0.02

## Evaluate InstaNovo on nine species v2 data with zero-shot learning
zs_eval_nine_species_v2:
	mkdir -p ./data/species_formatted_ipc
	gsutil -m cp -R gs://nine_species_dataset_v2/species_formatted_ipc/*.ipc ./data/species_formatted_ipc/

	mkdir -p ./checkpoints
	gsutil cp gs://denovo_checkpoints/extended_38aa4b76/epoch=3-step=800000.ckpt ./checkpoints/model.ckpt

	instanovo transformer predict \
		-d ./data/species_formatted_ipc/apis_mellifera.ipc \
		-i ./checkpoints/model.ckpt \
		-o apis_mellifera.csv \
		subset=0.1

	instanovo transformer predict \
		-d ./data/species_formatted_ipc/bacillus_subtilis.ipc \
		-i ./checkpoints/model.ckpt \
		-o bacillus_subtilis.csv \
		subset=0.0125

	instanovo transformer predict \
		-d ./data/species_formatted_ipc/candidatus_endoloripes.ipc \
		-i ./checkpoints/model.ckpt \
		-o candidatus_endoloripes.csv \
		subset=0.5

	instanovo transformer predict \
		-d ./data/species_formatted_ipc/h_sapiens.ipc \
		-i ./checkpoints/model.ckpt \
		-o h_sapiens.csv \
		subset=0.75

	instanovo transformer predict \
		-d ./data/species_formatted_ipc/methanosarcina_mazei.ipc \
		-i ./checkpoints/model.ckpt \
		-o methanosarcina_mazei.csv \
		subset=0.08

	instanovo transformer predict \
		-d ./data/species_formatted_ipc/mus_musculus.ipc \
		-i ./checkpoints/model.ckpt \
		-o mus_musculus.csv \
		subset=1.0

	instanovo transformer predict \
		-d ./data/species_formatted_ipc/saccharomyces_cerevisiae.ipc \
		-i ./checkpoints/model.ckpt \
		-o saccharomyces_cerevisiae.csv \
		subset=0.02

	instanovo transformer predict \
		-d ./data/species_formatted_ipc/solanum_lycopersicum.ipc \
		-i ./checkpoints/model.ckpt \
		-o solanum_lycopersicum.csv \
		subset=0.2

	instanovo transformer predict \
		-d ./data/species_formatted_ipc/vigna_mungo.ipc \
		-i ./checkpoints/model.ckpt \
		-o vigna_mungo.csv \
		subset=0.15

## Evaluate InstaNovo on data from Abhi
eval_abhi:
	mkdir -p ./data/instanovo_hlaI_pred
	gsutil -m cp -R gs://biondeep-data/instanovo/instanovo_hlaI_pred/ ./data
	uv run instanovo/utils/convert_to_ipc.py \
		./data/instanovo_hlaI_pred/ \
		./data/instanovo_hlaI_pred.ipc \
		--source_type mzxml --verbose
	gsutil cp gs://denovo_checkpoints/acpt_d093a745/epoch\=8-step\=6300000.ckpt ./checkpoints/acpt_d093a745/
	instanovo transformer predict \
		-d ./data/instanovo_hlaI_pred.ipc \
		-i ~/Coding/dtu-denovo-sequencing/checkpoints/acpt_d093a745/epoch\=8-step\=6300000.ckpt \
		-o instanovo_hlaI_pred.csv \
		-n -w 8 -b
	gsutil cp instanovo_hlaI_pred.csv gs://denovo_checkpoints/

## Evaluate InstaNovo on 9 species data zero-shot
eval_species_zero_shot:
	mkdir -p ./checkpoints/extended_38aa4b76/
	gsutil -m cp gs://denovo_checkpoints/extended_38aa4b76/epoch=3-step=800000.ckpt ./checkpoints/extended_38aa4b76/
	mkdir -p ./data/species/
	gsutil -m cp -R gs://nine_species_dataset_v2/species_formatted_ipc/*.ipc ./data/species/
	instanovo transformer predict \
		-d /data/species/apis_mellifera.ipc \
		-i "./checkpoints/extended_38aa4b76/epoch\=3-step\=800000.ckpt" \
		-o apis_mellifera.csv \
		subset=1.0 \
		batch_size=256


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

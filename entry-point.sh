#!/usr/bin/env bash

set -e


export GCP_PROJECT="ext-dtu-denovo-sequencing-gcp"

# Verify that all required variables are set
if [[ -z "${MLFLOW_TRACKING_USERNAME}" ]]; then
    echo "Error: MLFLOW_TRACKING_USERNAME not set"
    exit 1
fi

if [[ -z "${MLFLOW_TRACKING_PASSWORD}" ]]; then
    echo "Error: MLFLOW_TRACKING_PASSWORD not set"
    exit 1
fi

if [[ -z "${MLFLOW_ARTIFACT_URL}" ]]; then
    echo "Error: MLFLOW_ARTIFACT_URL not set"
    exit 1
fi

if [[ -z "${MLFLOW_DATABASE_URL}" ]]; then
    echo "Error: MLFLOW_DATABASE_URL not set"
    exit 1
fi

if [[ -z "${PORT}" ]]; then
    export PORT=8080
fi

if [[ -z "${HOST}" ]]; then
    export HOST=0.0.0.0
fi

export WSGI_AUTH_CREDENTIALS="${MLFLOW_TRACKING_USERNAME}:${MLFLOW_TRACKING_PASSWORD}"
export _MLFLOW_SERVER_ARTIFACT_ROOT="${MLFLOW_ARTIFACT_URL}"
export _MLFLOW_SERVER_FILE_STORE="${MLFLOW_DATABASE_URL}"

# Start MLflow and ngingx using supervisor
exec gunicorn -b "${HOST}:${PORT}" -w 4 --log-level debug --access-logfile=- --error-logfile=- --log-level=debug mlflow_auth:app

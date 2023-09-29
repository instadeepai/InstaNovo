from __future__ import annotations

from mlflow.server import app as mlflow_app
from wsgi_basic_auth import BasicAuth

app = BasicAuth(mlflow_app)

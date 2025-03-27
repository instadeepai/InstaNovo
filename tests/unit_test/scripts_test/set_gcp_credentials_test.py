import base64
import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import mock_open, patch

import pytest

from instanovo.scripts.set_gcp_credentials import (
    set_credentials,
)


@patch("builtins.open", new_callable=mock_open)
def test_set_credentials(mock_file: Any) -> None:
    """Tests setting of GCP credentials."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(
            OSError,
            match="To use GCP buckets you should set 'GOOGLE_APPLICATION_CREDENTIALS' env variable."
            " It corresponds to the path to the json file with the credentials.",
        ):
            set_credentials()

    with patch.dict(
        os.environ,
        {"GOOGLE_APPLICATION_CREDENTIALS": "path/example_credentials.json"},
        clear=True,
    ):
        with pytest.raises(
            OSError,
            match="If the json file 'GOOGLE_APPLICATION_CREDENTIALS' does not exist, "
            "you must set 'GS_CREDENTIALS_ENCODED' as the base64 encoded json file.",
        ):
            set_credentials()

    with patch.dict(
        os.environ,
        {
            "GOOGLE_APPLICATION_CREDENTIALS": "path/example_credentials.json",
            "GS_CREDENTIALS_ENCODED": base64.b64encode(
                json.dumps({"credential_key": "credential_value"}).encode("ascii")
            ).decode("ascii"),
        },
        clear=True,
    ):
        set_credentials()
        mock_file.assert_called_once_with(Path("path/example_credentials.json"), "w")
        writer = mock_file()
        written_content = "".join(call[0][0] for call in writer.write.call_args_list)
        assert json.loads(written_content) == {"credential_key": "credential_value"}

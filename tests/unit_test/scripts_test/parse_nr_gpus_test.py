from unittest.mock import mock_open, patch

from scripts.parse_nr_gpus import parse_nr_gpus


def test_parse_nr_gpus() -> None:
    """Test function that returns the number of GPUs specified in the manifest.yaml file."""
    fake_manifest_content = """
    spec:
        types:
            Worker:
                resources:
                    accelerators:
                        gpu:
                            count: 2
    """

    with patch("builtins.open", mock_open(read_data=fake_manifest_content)):
        assert parse_nr_gpus() == 2

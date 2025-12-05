from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import DictConfig

from instanovo.utils.device_handler import (
    apply_device_config,
    check_device,
    detect_device,
    get_device_capabilities,
    get_device_config_updates,
    validate_and_configure_device,
)


class TestGetDeviceCapabilities:
    """Test the get_device_capabilities function."""

    @patch("torch.cuda.is_available", autospec=True)
    @patch("torch.backends.mps.is_available", autospec=True)
    def test_cuda_available(self, mock_mps: MagicMock, mock_cuda: MagicMock) -> None:
        """Test when CUDA is available."""
        mock_cuda.return_value = True
        mock_mps.return_value = False

        capabilities = get_device_capabilities()

        assert capabilities == {"cuda": True, "mps": False}
        mock_cuda.assert_called_once()
        mock_mps.assert_called_once()

    @patch("torch.cuda.is_available", autospec=True)
    @patch("torch.backends.mps.is_available", autospec=True)
    def test_mps_available(self, mock_mps: MagicMock, mock_cuda: MagicMock) -> None:
        """Test when MPS is available."""
        mock_cuda.return_value = False
        mock_mps.return_value = True

        capabilities = get_device_capabilities()

        assert capabilities == {"cuda": False, "mps": True}

    @patch("torch.cuda.is_available", autospec=True)
    @patch("torch.backends.mps.is_available", autospec=True)
    def test_both_available(self, mock_mps: MagicMock, mock_cuda: MagicMock) -> None:
        """Test when both CUDA and MPS are available."""
        mock_cuda.return_value = True
        mock_mps.return_value = True

        capabilities = get_device_capabilities()

        assert capabilities == {"cuda": True, "mps": True}

    @patch("torch.cuda.is_available", autospec=True)
    @patch("torch.backends.mps.is_available", autospec=True)
    def test_neither_available(self, mock_mps: MagicMock, mock_cuda: MagicMock) -> None:
        """Test when neither CUDA nor MPS are available."""
        mock_cuda.return_value = False
        mock_mps.return_value = False

        capabilities = get_device_capabilities()

        assert capabilities == {"cuda": False, "mps": False}


class TestDetectDevice:
    """Test the detect_device function."""

    @patch("instanovo.utils.device_handler.get_device_capabilities", autospec=True)
    def test_detect_cuda(self, mock_capabilities: MagicMock) -> None:
        """Test device detection when CUDA is available."""
        mock_capabilities.return_value = {"cuda": True, "mps": False}

        device = detect_device()

        assert device == "cuda"

    @patch("instanovo.utils.device_handler.get_device_capabilities", autospec=True)
    def test_detect_mps(self, mock_capabilities: MagicMock) -> None:
        """Test device detection when only MPS is available."""
        mock_capabilities.return_value = {"cuda": False, "mps": True}

        device = detect_device()

        assert device == "mps"

    @patch("instanovo.utils.device_handler.get_device_capabilities", autospec=True)
    def test_detect_cpu(self, mock_capabilities: MagicMock) -> None:
        """Test device detection when neither CUDA nor MPS are available."""
        mock_capabilities.return_value = {"cuda": False, "mps": False}

        device = detect_device()

        assert device == "cpu"

    @patch("instanovo.utils.device_handler.get_device_capabilities", autospec=True)
    def test_cuda_priority_over_mps(self, mock_capabilities: MagicMock) -> None:
        """Test that CUDA is preferred over MPS when both are available."""
        mock_capabilities.return_value = {"cuda": True, "mps": True}

        device = detect_device()

        assert device == "cuda"


class TestGetDeviceConfigUpdates:
    """Test the get_device_config_updates function."""

    def test_cuda_config_updates(self) -> None:
        """Test configuration updates for CUDA device."""
        updates = get_device_config_updates("cuda")

        expected = {
            "mps": False,
            "force_cpu": False,
        }
        assert updates == expected

    def test_mps_config_updates(self) -> None:
        """Test configuration updates for MPS device."""
        updates = get_device_config_updates("mps")

        expected = {"mps": True, "force_fp32": True, "force_cpu": False, "model": {"peak_embedding_dtype": "float32"}}
        assert updates == expected

    def test_cpu_config_updates(self) -> None:
        """Test configuration updates for CPU device."""
        updates = get_device_config_updates("cpu")

        expected = {"force_cpu": True}
        assert updates == expected

    def test_unknown_device(self) -> None:
        """Test configuration updates for unknown device."""
        with pytest.raises(ValueError):
            get_device_config_updates("unknown")


class TestApplyDeviceConfig:
    """Test the apply_device_config function."""

    @patch("instanovo.utils.device_handler.detect_device", autospec=True)
    def test_apply_config_auto_detect(self, mock_detect: MagicMock) -> None:
        """Test applying config with auto-detected device."""
        mock_detect.return_value = "cuda"
        config = DictConfig({})

        device = apply_device_config(config)

        assert device == "cuda"
        assert not config["mps"]
        assert not config["force_cpu"]

    def test_apply_config_specified_device(self) -> None:
        """Test applying config with specified device."""
        config = DictConfig({})

        device = apply_device_config(config, "mps")

        assert device == "mps"
        assert config["mps"]
        assert config["force_fp32"]
        assert not config["force_cpu"]

    def test_apply_config_existing_model_section(self) -> None:
        """Test applying config when model section already exists."""
        config = DictConfig({"model": {"peak_embedding_dtype": "float64"}})

        device = apply_device_config(config, "mps")

        assert device == "mps"
        assert config["model"]["peak_embedding_dtype"] == "float32"

    def test_apply_config_cpu_device(self) -> None:
        """Test applying config for CPU device."""
        config = DictConfig({})

        device = apply_device_config(config, "cpu")

        assert device == "cpu"
        assert config["force_cpu"]


class TestValidateAndConfigureDevice:
    """Test the validate_and_configure_device function."""

    @patch("instanovo.utils.device_handler.get_device_capabilities", autospec=True)
    def test_mps_available_and_requested(self, mock_capabilities: MagicMock, caplog: Any) -> None:
        """Test when MPS is available and requested."""
        mock_capabilities.return_value = {"cuda": False, "mps": True}
        config = DictConfig({"mps": True, "force_cpu": False})

        validate_and_configure_device(config)

        assert config["force_fp32"]
        assert not config["force_cpu"]
        assert "MPS is set to True, forcing fp32. Note that performance on MPS may differ to performance on CUDA." in caplog.text

    @patch("instanovo.utils.device_handler.get_device_capabilities", autospec=True)
    def test_mps_not_available_but_requested(self, mock_capabilities: MagicMock, caplog: Any) -> None:
        """Test when MPS is requested but not available."""
        mock_capabilities.return_value = {"cuda": False, "mps": False}
        config = DictConfig({"mps": True, "force_cpu": False})

        validate_and_configure_device(config)

        assert not config["mps"]
        assert "MPS is not available, setting mps to False." in caplog.text

    @patch("instanovo.utils.device_handler.get_device_capabilities", autospec=True)
    def test_mps_requested_but_force_cpu_set(self, mock_capabilities: MagicMock, caplog: Any) -> None:
        """Test when MPS is requested but force_cpu is True."""
        mock_capabilities.return_value = {"cuda": False, "mps": True}
        config = DictConfig({"mps": True, "force_cpu": True})

        validate_and_configure_device(config)

        assert not config["mps"]
        assert "Force CPU is set to True, setting mps to False." in caplog.text

    @patch("instanovo.utils.device_handler.get_device_capabilities", autospec=True)
    def test_cuda_not_available_no_force_cpu(self, mock_capabilities: MagicMock, caplog: Any) -> None:
        """Test when CUDA is not available and force_cpu is False."""
        mock_capabilities.return_value = {"cuda": False, "mps": False}
        config = DictConfig({"mps": False, "force_cpu": False})

        validate_and_configure_device(config)

        assert config["force_cpu"]
        assert "CUDA is not available, setting force_cpu to True." in caplog.text


class TestCheckDeviceBackwardCompatibility:
    """Test the check_device function for backward compatibility."""

    @patch("instanovo.utils.device_handler.detect_device", autospec=True)
    def test_check_device_no_config(self, mock_detect: MagicMock) -> None:
        """Test check_device without config parameter."""
        mock_detect.return_value = "cuda"

        device = check_device()

        assert device == "cuda"
        mock_detect.assert_called_once()

    @patch("instanovo.utils.device_handler.apply_device_config", autospec=True)
    def test_check_device_with_config(self, mock_apply: MagicMock) -> None:
        """Test check_device with config parameter."""
        mock_apply.return_value = "mps"
        config = DictConfig({})

        device = check_device(config)

        assert device == "mps"
        mock_apply.assert_called_once_with(config)

    def test_check_device_none_config(self) -> None:
        """Test check_device with None config."""
        with patch("instanovo.utils.device_handler.detect_device", autospec=True) as mock_detect:
            mock_detect.return_value = "cpu"

            device = check_device(None)

            assert device == "cpu"


class TestIntegration:
    """Integration tests for device handler functions."""

    @patch("torch.cuda.is_available", autospec=True)
    @patch("torch.backends.mps.is_available", autospec=True)
    def test_full_workflow_cuda_system(self, mock_mps: MagicMock, mock_cuda: MagicMock) -> None:
        """Test full workflow on a system with CUDA."""
        mock_cuda.return_value = True
        mock_mps.return_value = False

        # Test device detection
        device = detect_device()
        assert device == "cuda"

        # Test config application
        config = DictConfig({})
        applied_device = apply_device_config(config, device)
        assert applied_device == "cuda"
        assert not config["mps"]
        assert not config["force_cpu"]

    @patch("torch.cuda.is_available", autospec=True)
    @patch("torch.backends.mps.is_available", autospec=True)
    def test_full_workflow_mps_system(self, mock_mps: MagicMock, mock_cuda: MagicMock) -> None:
        """Test full workflow on a system with MPS."""
        mock_cuda.return_value = False
        mock_mps.return_value = True

        # Test device detection
        device = detect_device()
        assert device == "mps"

        # Test config application
        config = DictConfig({})
        applied_device = apply_device_config(config, device)
        assert applied_device == "mps"
        assert config["mps"]
        assert config["force_fp32"]
        assert not config["force_cpu"]

    @patch("torch.cuda.is_available", autospec=True)
    @patch("torch.backends.mps.is_available", autospec=True)
    def test_full_workflow_cpu_only_system(self, mock_mps: MagicMock, mock_cuda: MagicMock) -> None:
        """Test full workflow on a CPU-only system."""
        mock_cuda.return_value = False
        mock_mps.return_value = False

        # Test device detection
        device = detect_device()
        assert device == "cpu"

        # Test config application
        config = DictConfig({})
        applied_device = apply_device_config(config, device)
        assert applied_device == "cpu"
        assert config["force_cpu"]


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_config(self) -> None:
        """Test with empty configuration."""
        config = DictConfig({})
        device = apply_device_config(config, "cuda")

        assert device == "cuda"
        assert "mps" in config
        assert "force_cpu" in config

    def test_config_with_existing_values(self) -> None:
        """Test with configuration that has existing values."""
        config = DictConfig({"mps": True, "force_cpu": True, "existing_param": "value"})

        device = apply_device_config(config, "cuda")

        assert device == "cuda"
        assert not config["mps"]  # Should be overwritten
        assert not config["force_cpu"]  # Should be overwritten
        assert config["existing_param"] == "value"  # Should be preserved

    def test_model_config_skip_when_missing(self) -> None:
        """Test that model config is skipped when model section doesn't exist."""
        config = DictConfig({})

        device = apply_device_config(config, "mps")

        assert device == "mps"
        assert config["mps"]
        assert config["force_fp32"]
        assert not config["force_cpu"]

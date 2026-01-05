import pytest
import torch

from tunalab.testing import get_available_devices, get_test_dtypes


@pytest.fixture(params=get_available_devices())
def device(request):
    """Fixture that parametrizes tests over available devices."""
    return request.param


@pytest.fixture(params=get_test_dtypes())
def dtype(request):
    """Fixture that parametrizes tests over standard dtypes."""
    return request.param

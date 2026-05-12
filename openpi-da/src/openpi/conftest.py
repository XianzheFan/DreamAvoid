import os
import pynvml
import pytest

def set_jax_cpu_backend_if_no_gpu() -> None:
    try:
        pynvml.nvmlInit()
        pynvml.nvmlShutdown()
    except pynvml.NVMLError:
        os.environ['JAX_PLATFORMS'] = 'cpu'

def pytest_configure(config: pytest.Config) -> None:
    set_jax_cpu_backend_if_no_gpu()

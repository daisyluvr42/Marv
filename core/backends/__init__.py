from core.backends.base import BaseCoreBackend
from core.backends.mock import MockCoreBackend
from core.backends.mock_v2 import MockV2CoreBackend


def get_backend(name: str) -> BaseCoreBackend:
    if name == "mock":
        return MockCoreBackend()
    if name == "mock_v2":
        return MockV2CoreBackend()
    raise ValueError(f"Unsupported backend: {name}")

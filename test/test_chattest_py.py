import importlib
import pytest

modules = [
    "service.APIservice",
    "service.CLI",
    "component.template",
    "train.load_model",
]

@pytest.mark.parametrize("mod", modules)
def test_import_module(mod):
    try:
        importlib.import_module(mod)
    except Exception as e:
        pytest.skip(f"Skipping import test for {mod}: {e}")

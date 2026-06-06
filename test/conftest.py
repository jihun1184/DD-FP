# test/conftest.py
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-gpu",
        action="store_true",
        default=False,
        help="Run GPU tests (requires cupy).",
    )
"""
Unit and regression test for the qem_ml package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import qem_ml


def test_qem_ml_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "qem_ml" in sys.modules

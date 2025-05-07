"""
    test_deepart.py

# Description
Tests the deepart package.

# Authors
- Sasha Petrenko <sap625@mst.edu>
"""

# --------------------------------------------------------------------------- #
# STANDARD IMPORTS
# --------------------------------------------------------------------------- #

import os
from pathlib import Path
import logging as lg
from dataclasses import dataclass
from typing import (
    List,
    Dict,
    Tuple,
)

# --------------------------------------------------------------------------- #
# CUSTOM IMPORTS
# --------------------------------------------------------------------------- #

import pytest
# import numpy as np
# import pandas as pd
# import sklearn.metrics as skm
from torch.utils.data import DataLoader

# --------------------------------------------------------------------------- #
# LOCAL IMPORTS
# --------------------------------------------------------------------------- #

import src.deepart as da

print(f"\nTesting path is: {os.getcwd()}")


# --------------------------------------------------------------------------- #
# DATACLASSES
# --------------------------------------------------------------------------- #


@dataclass
class TestData():
    """
    A container dataclass for test data.
    """

    # The test dataset dictionary
    # datasets: Dict
    train: DataLoader
    test: DataLoader

    # Tells pytest that this is not a test class
    __test__ = False

    # def count(self, dataset: str) -> int:
    #     """
    #     Returns the number of samples in a dataset entry.

    #     Parameters
    #     ----------
    #     dataset : str
    #         The key corresponding to which dataset you wish to get a count of.

    #     Returns
    #     -------
    #     int
    #         The number of samples in self.datasets[dataset].
    #     """

    #     return len(self.datasets[dataset]["labels"])


# --------------------------------------------------------------------------- #
# FIXTURES
# --------------------------------------------------------------------------- #


# Set the fixture scope to the testing session to load the data once
@pytest.fixture(scope="session")
def data() -> TestData:
    """
    Data loading test fixture.

    This fixture is run once for the entire pytest session.
    """

    lg.info("LOADING DATA")

    lg.info(dir(da))

    # data_path = Path("tests", "data")
    train, test = da.data.get_data()

    # Instantiate and return the TestData object
    # return TestData(data_dict)
    return TestData(train, test)


# --------------------------------------------------------------------------- #
# TESTS
# --------------------------------------------------------------------------- #


class TestDeepART:
    """
    Pytest class containing DeepART unit tests.
    """

    def test_load_data(self, data: TestData):
        """
        Test loading the partitioning data.

        Parameters
        ----------
        data : TestData
            The data loaded as a pytest fixture.
        """

        lg.info("--- TESTING DATA LOADING ---")
        lg.info(f"Data location: {id(data)}")

        # for value in data.datasets.values():
        #     log_data(value)

# --------------------------------------------------------------------------- #
# TESTS
# --------------------------------------------------------------------------- #


class TestCVI:
    """
    Pytest class containing CVI/ICVI unit tests.
    """

    def test_load_data(self, data: TestData):
        """
        Test loading the partitioning data.

        Parameters
        ----------
        data : TestData
            The data loaded as a pytest fixture.
        """

        lg.info("--- TESTING DATA LOADING ---")
        lg.info(f"Data location: {id(data)}")

        # for value in data.datasets.values():
        #     log_data(value)

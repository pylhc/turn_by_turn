from pathlib import Path

import numpy as np
import pytest

from turn_by_turn.constants import PLANES, PRINT_PRECISION
from turn_by_turn.errors import DataTypeError
from turn_by_turn.io import write_lhc_ascii, write_tbt
from turn_by_turn.structures import TbtData
from turn_by_turn.leir import read_tbt

INPUTS_DIR = Path(__file__).parent / "inputs"
ASCII_PRECISION = 0.6 / np.power(10, PRINT_PRECISION)  # not 0.5 due to rounding issues


def test_tbt_write_read_ascii(_sdds_file, _test_file):
    origin = read_tbt(_sdds_file, reorder_planes=False)
    write_lhc_ascii(_test_file, origin)
    new = read_tbt(_test_file, reorder_planes=False)
    compare_tbt(origin, new, True)


# ----- Helpers ----- #


def compare_tbt(origin: TbtData, new: TbtData, no_binary: bool, max_deviation=ASCII_PRECISION) -> None:
    assert new.nturns == origin.nturns
    assert new.nbunches == origin.nbunches
    assert new.bunch_ids == origin.bunch_ids
    for index in range(origin.nbunches):
        for plane in PLANES:
            assert np.all(new.matrices[index][plane].index == origin.matrices[index][plane].index)
            origin_mat = origin.matrices[index][plane].to_numpy()
            new_mat = new.matrices[index][plane].to_numpy()
            if no_binary:
                assert np.nanmax(np.abs(origin_mat - new_mat)) < max_deviation
            else:
                assert np.all(origin_mat == new_mat)


def create_data(phases, nbpm, function) -> np.ndarray:
    return np.ones((nbpm, len(phases))) * function(phases)


@pytest.fixture()
def _test_file(tmp_path) -> Path:
    yield tmp_path / "test_leir"


@pytest.fixture()
def _sdds_file() -> Path:
    return INPUTS_DIR / "test_leir.sdds"



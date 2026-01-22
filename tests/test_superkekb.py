from pathlib import Path

import numpy as np
import pytest

import turn_by_turn.errors
from turn_by_turn.constants import PRINT_PRECISION
from turn_by_turn.errors import DataTypeError
from turn_by_turn.io import read_tbt, write_lhc_ascii, write_tbt
from turn_by_turn.structures import TbtData

INPUTS_DIR = Path(__file__).parent / "inputs"
ASCII_PRECISION = 0.6 / np.power(10, PRINT_PRECISION)  # not 0.5 due to rounding issues


def test_tbt_read_write_sdds(_superkekb_file, _test_file):
    origin = read_tbt(_superkekb_file, datatype="superkekb")

    # Save as a SDDS file and reopen it
    write_tbt(_test_file, origin)
    new = read_tbt(f"{_test_file}.sdds")
    compare_tbt(origin, new, no_binary=False)


def test_tbt_write(_superkekb_file, _test_file):
    origin = read_tbt(_superkekb_file, datatype="superkekb")

    # Save as a SuperKEKB file, should raise NotImplementedError
    with pytest.raises(turn_by_turn.errors.DataTypeError):
        write_tbt(_test_file, origin, datatype="superkekb")


# ----- Helpers ----- #


def compare_tbt(
    origin: TbtData,
    new: TbtData,
    no_binary: bool,
    max_deviation=ASCII_PRECISION,
    is_tracking_data: bool = False,
) -> None:
    assert new.nturns == origin.nturns
    assert new.nbunches == origin.nbunches
    assert new.bunch_ids == origin.bunch_ids
    for index in range(origin.nbunches):
        # In matrices are either TransverseData or TrackingData and we can get all the fields from the `fieldnames` classmethod
        for field in origin.matrices[0].fieldnames():
            assert np.all(
                new.matrices[index][field].index == origin.matrices[index][field].index
            )
            origin_mat = origin.matrices[index][field].to_numpy()
            new_mat = new.matrices[index][field].to_numpy()
            if no_binary:
                assert np.nanmax(np.abs(origin_mat - new_mat)) < max_deviation
            else:
                assert np.all(origin_mat == new_mat)


@pytest.fixture()
def _test_file(tmp_path) -> Path:
    return tmp_path / "test_file"


@pytest.fixture()
def _superkekb_file() -> Path:
    return INPUTS_DIR / "test_superkekb"

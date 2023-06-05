from pathlib import Path

import numpy as np
import pytest

from turn_by_turn.constants import PRINT_PRECISION
from turn_by_turn.errors import DataTypeError
from turn_by_turn.io import read_tbt, write_lhc_ascii, write_tbt
from turn_by_turn.structures import TbtData

INPUTS_DIR = Path(__file__).parent / "inputs"
ASCII_PRECISION = 0.6 / np.power(10, PRINT_PRECISION)  # not 0.5 due to rounding issues


@pytest.mark.parametrize("datatype", ["invalid", "not_supported"])
def test_tbt_read_raises_on_invalid_datatype(_sdds_file, caplog, datatype):
    with pytest.raises(DataTypeError):
        _ = read_tbt(_sdds_file, datatype=datatype)

    for record in caplog.records:
        assert record.levelname == "ERROR"


@pytest.mark.parametrize("datatype", ["invalid", "not_supported"])
def test_tbt_write_raises_on_invalid_datatype(_sdds_file, caplog, datatype):
    with pytest.raises(DataTypeError):
        write_tbt(_sdds_file, tbt_data=None, datatype=datatype)

    for record in caplog.records:
        assert record.levelname == "ERROR"


def test_tbt_write_read_sdds_binary(_sdds_file, _test_file):
    origin = read_tbt(_sdds_file)
    write_tbt(_test_file, origin)
    new = read_tbt(f"{_test_file}.sdds")
    compare_tbt(origin, new, False)


def test_tbt_write_read_sdds_binary_with_noise(_sdds_file, _test_file):
    origin = read_tbt(_sdds_file)
    write_tbt(_test_file, origin, noise=2)
    new = read_tbt(f"{_test_file}.sdds")

    with pytest.raises(AssertionError):  # should be different
        compare_tbt(origin, new, False)


def test_tbt_write_read_ascii(_sdds_file, _test_file):
    origin = read_tbt(_sdds_file)
    write_lhc_ascii(_test_file, origin)
    new = read_tbt(_test_file)
    compare_tbt(origin, new, True)


# ----- Helpers ----- #


def compare_tbt(origin: TbtData, new: TbtData, no_binary: bool, max_deviation = ASCII_PRECISION, is_tracking_data: bool = False) -> None:
    assert new.nturns == origin.nturns
    assert new.nbunches == origin.nbunches
    assert new.bunch_ids == origin.bunch_ids
    for index in range(origin.nbunches):
        # In matrices are either TransverseData or TrackingData and we can get all the fields from the `fieldnames` classmethod
        for field in origin.matrices[0].fieldnames():
            assert np.all(new.matrices[index][field].index == origin.matrices[index][field].index)
            origin_mat = origin.matrices[index][field].to_numpy()
            new_mat = new.matrices[index][field].to_numpy()
            if no_binary:
                assert np.nanmax(np.abs(origin_mat - new_mat)) < max_deviation
            else:
                assert np.all(origin_mat == new_mat)


def create_data(phases, nbpm, function) -> np.ndarray:
    return np.ones((nbpm, len(phases))) * function(phases)


@pytest.fixture()
def _test_file(tmp_path) -> Path:
    yield tmp_path / "test_file"


@pytest.fixture()
def _sdds_file() -> Path:
    return INPUTS_DIR / "test_file.sdds"



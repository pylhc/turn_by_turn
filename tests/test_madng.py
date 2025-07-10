from pathlib import Path

import pytest

from tests.test_lhc_and_general import INPUTS_DIR, compare_tbt
from turn_by_turn import madng, read_tbt, write_tbt
from turn_by_turn.structures import TbtData


def test_read_ng(_ng_file: Path, example_fake_tbt: TbtData):
    # Check directly from the module
    new = madng.read_tbt(_ng_file)
    compare_tbt(example_fake_tbt, new, no_binary=True)

    # Check from the main function
    new = read_tbt(_ng_file, datatype="madng")
    compare_tbt(example_fake_tbt, new, no_binary=True)


def test_write_ng(_ng_file: Path, tmp_path: Path, example_fake_tbt: TbtData):
    # Write the data
    from_tbt = tmp_path / "from_tbt.tfs"
    madng.write_tbt(from_tbt, example_fake_tbt)

    # Read the written data
    new_tbt = madng.read_tbt(from_tbt)
    compare_tbt(example_fake_tbt, new_tbt, no_binary=True)

    # Check from the main function
    written_tbt = read_tbt(_ng_file, datatype="madng")
    write_tbt(from_tbt, written_tbt, datatype="madng")

    new_tbt = read_tbt(from_tbt, datatype="madng")
    compare_tbt(written_tbt, new_tbt, no_binary=True)
    assert written_tbt.date == new_tbt.date


def test_error_ng(_error_file: Path):
    with pytest.raises(ValueError):
        read_tbt(_error_file, datatype="madng")


# ---- Fixtures ---- #
@pytest.fixture
def _ng_file(tmp_path: Path) -> Path:
    return INPUTS_DIR / "madng" / "fodo_track.tfs"


@pytest.fixture
def _error_file(tmp_path: Path) -> Path:
    return INPUTS_DIR / "madng" / "fodo_track_error.tfs"

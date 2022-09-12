
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.test_lhc_and_general import compare_tbt, INPUTS_DIR
from turn_by_turn import ptc, trackone
from turn_by_turn.errors import PTCFormatError
from turn_by_turn.structures import TbtData, TransverseData


def test_read_ptc(_ptc_file):
    new = ptc.read_tbt(_ptc_file)
    origin = _original_trackone()
    compare_tbt(origin, new, True)


def test_read_ptc_raises_on_invalid_file(_invalid_ptc_file):
    with pytest.raises(PTCFormatError):
        _ = ptc.read_tbt(_invalid_ptc_file)


def test_read_ptc_defaults_date(_ptc_file_no_date):
    new = ptc.read_tbt(_ptc_file_no_date)
    assert new.date.day == datetime.today().day
    assert new.date.tzname() == "UTC"


def test_read_ptc_sci(_ptc_file_sci):
    new = ptc.read_tbt(_ptc_file_sci)
    origin = _original_trackone()
    compare_tbt(origin, new, True)


def test_read_ptc_looseparticles(_ptc_file_losses):
    new = ptc.read_tbt(_ptc_file_losses)
    assert len(new.matrices) == 3
    assert len(new.matrices[0].X.columns) == 9
    assert all(new.matrices[0].X.index == np.array([f"BPM{i+1}" for i in range(3)]))
    assert not new.matrices[0].X.isna().any().any()


def test_read_trackone(_ptc_file):
    new = trackone.read_tbt(_ptc_file)
    origin = _original_trackone(True)
    compare_tbt(origin, new, True)


def test_read_trackone_sci(_ptc_file_sci):
    new = trackone.read_tbt(_ptc_file_sci)
    origin = _original_trackone(True)
    compare_tbt(origin, new, True)


def test_read_trackone_looseparticles(_ptc_file_losses):
    new = trackone.read_tbt(_ptc_file_losses)
    assert len(new.matrices) == 3
    assert len(new.matrices[0].X.columns) == 9
    assert all(new.matrices[0].X.index == np.array([f"BPM{i+1}" for i in range(3)]))
    assert not new.matrices[0].X.isna().any().any()


def _original_trackone(track: bool = False) -> TbtData:
    names = np.array(["C1.BPM1"])
    matrix = [
        TransverseData(
            X=pd.DataFrame(index=names, data=[[0.001, -0.0003606, -0.00165823, -0.00266631]]),
            Y=pd.DataFrame(index=names, data=[[0.001, 0.00070558, -0.00020681, -0.00093807]]),
        ),
        TransverseData(
            X=pd.DataFrame(index=names, data=[[0.0011, -0.00039666, -0.00182406, -0.00293294]]),
            Y=pd.DataFrame(index=names, data=[[0.0011, 0.00077614, -0.00022749, -0.00103188]]),
        ),
    ]
    origin = TbtData(matrix, None, [0, 1] if track else [1, 2], 4)
    return origin


@pytest.fixture()
def _ptc_file_no_date() -> Path:
    return INPUTS_DIR / "test_trackone_no_date"


@pytest.fixture()
def _ptc_file_losses() -> Path:
    return INPUTS_DIR / "test_trackone_losses"


@pytest.fixture()
def _ptc_file_sci() -> Path:
    return INPUTS_DIR / "test_trackone_sci"


@pytest.fixture()
def _ptc_file() -> Path:
    return INPUTS_DIR / "test_trackone"


@pytest.fixture()
def _invalid_ptc_file() -> Path:
    return INPUTS_DIR / "test_wrong_ptc"


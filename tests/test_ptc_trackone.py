
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.test_lhc_and_general import ASCII_PRECISION, INPUTS_DIR, compare_tbt
from turn_by_turn import ptc, trackone
from turn_by_turn.errors import PTCFormatError
from turn_by_turn.structures import TbtData, TrackingData, TransverseData


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


def test_read_trackone_simdata(_ptc_file):
    new = trackone.read_tbt(_ptc_file, is_tracking_data=True)  # read all fields (includes PX, PY, T, PT, S, E)
    origin = _original_simulation_data()
    compare_tbt(origin, new, True, is_tracking_data=True)


# ----- Helpers ----- #


def _original_trackone(track: bool = False) -> TbtData:
    names = np.array(["C1.BPM1"])
    matrix = [
        TransverseData(  # first "bunch"
            X=pd.DataFrame(index=names, data=[[0.001, -0.0003606, -0.00165823, -0.00266631]]),
            Y=pd.DataFrame(index=names, data=[[0.001, 0.00070558, -0.00020681, -0.00093807]]),
        ),
        TransverseData(  # second "bunch"
            X=pd.DataFrame(index=names, data=[[0.0011, -0.00039666, -0.00182406, -0.00293294]]),
            Y=pd.DataFrame(index=names, data=[[0.0011, 0.00077614, -0.00022749, -0.00103188]]),
        ),
    ]
    origin = TbtData(matrix, None, [0, 1] if track else [1, 2], 4)
    return origin


def _original_simulation_data() -> TbtData:
    names = np.array(["C1.BPM1"])
    matrices = [
        TrackingData(  # first "bunch"
            X=pd.DataFrame(index=names, data=[[0.001, -0.000361, -0.001658, -0.002666]]),
            PX=pd.DataFrame(index=names, data=[[0.0, -0.000202, -0.000368, -0.00047]]),
            Y=pd.DataFrame(index=names, data=[[0.001,  0.000706, -0.000207, -0.000938]]),
            PY=pd.DataFrame(index=names, data=[[0.0, -0.000349, -0.000392, -0.000092]]),
            T=pd.DataFrame(index=names, data=[[0.0, -0.000008, -0.000015, -0.000023]]),
            PT=pd.DataFrame(index=names, data=[[0, 0, 0, 0]]),
            S=pd.DataFrame(index=names, data=[[0, 0, 0, 0]]),
            E=pd.DataFrame(index=names, data=[[500.00088,  500.00088,  500.00088,  500.00088]]),
        ),
        TrackingData(  # second "bunch"
            X=pd.DataFrame(index=names, data=[[0.0011, -0.000397, -0.001824, -0.002933]]),
            PX=pd.DataFrame(index=names, data=[[0.0, -0.000222, -0.000405, -0.000517]]),
            Y=pd.DataFrame(index=names, data=[[0.0011,  0.000776, -0.000227, -0.001032]]),
            PY=pd.DataFrame(index=names, data=[[0.0, -0.000384, -0.000431, -0.000101]]),
            T=pd.DataFrame(index=names, data=[[-0.0, -0.000009, -0.000018, -0.000028]]),
            PT=pd.DataFrame(index=names, data=[[0, 0, 0, 0]]),
            S=pd.DataFrame(index=names, data=[[0, 0, 0, 0]]),
            E=pd.DataFrame(index=names, data=[[500.00088,  500.00088,  500.00088,  500.00088]]),
        )
    ]
    origin = TbtData(matrices, date=None, bunch_ids=[0, 1], nturns=4)  # [0, 1] for bunch_ids because it's from tracking
    return origin


# ----- Fixtures ----- #


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


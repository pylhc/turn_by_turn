
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import h5py

from turn_by_turn.constants import PRINT_PRECISION
from turn_by_turn.errors import DataTypeError
from turn_by_turn.structures import TbtData, TransverseData
from tests.test_lhc_and_general import create_data, compare_tbt

from turn_by_turn.doros import N_ORBIT_SAMPLES, read_tbt, write_tbt, DEFAULT_BUNCH_ID, POSITIONS

INPUTS_DIR = Path(__file__).parent / "inputs"


def test_read_write_real_data(tmp_path):
    tbt = read_tbt(INPUTS_DIR / "test_doros.h5", bunch_id=10)

    assert tbt.nbunches == 1
    assert len(tbt.matrices) == 1
    assert tbt.nturns == 50000
    assert tbt.matrices[0].X.shape == (3, tbt.nturns)
    assert tbt.matrices[0].Y.shape == (3, tbt.nturns)
    assert len(set(tbt.matrices[0].X.index)) == 3
    assert np.all(tbt.matrices[0].X.index == tbt.matrices[0].Y.index)

    file_path = tmp_path / "test_file.h5"
    write_tbt(tbt, file_path)
    new = read_tbt(file_path, bunch_id=10)
    compare_tbt(tbt, new, no_binary=False)


def test_write_read(tmp_path):
    tbt = _tbt_data()
    file_path = tmp_path / "test_file.h5"
    write_tbt(tbt, file_path)
    new = read_tbt(file_path)
    compare_tbt(tbt, new, no_binary=False)


def test_read_raises_different_bpm_lengths(tmp_path):
    tbt = _tbt_data()
    file_path = tmp_path / "test_file.h5"
    write_tbt(tbt, file_path)

    bpm = tbt.matrices[0].X.index[0]

    # modify the BPM lengths in the file
    with h5py.File(file_path, "r+") as h5f:
        delta = 10
        del h5f[bpm][N_ORBIT_SAMPLES]
        h5f[bpm][N_ORBIT_SAMPLES] = [tbt.matrices[0].X.shape[1] - delta]
        for key in POSITIONS.values():
            data = h5f[bpm][key][:-delta]
            del h5f[bpm][key]
            h5f[bpm][key] = data

    with pytest.raises(ValueError) as e:
        read_tbt(file_path)
    assert "Not all BPMs have the same number of turns!" in str(e)


def test_read_raises_on_different_bpm_lengths_in_data(tmp_path):
    tbt = _tbt_data()
    file_path = tmp_path / "test_file.h5"
    write_tbt(tbt, file_path)

    bpms = [tbt.matrices[0].X.index[i] for i in (0, 2)]
    
    # modify the BPM lengths in the file
    with h5py.File(file_path, "r+") as h5f:
        for bpm in bpms:
            del h5f[bpm][N_ORBIT_SAMPLES]
            h5f[bpm][N_ORBIT_SAMPLES] = [tbt.matrices[0].X.shape[1] + 10]

    with pytest.raises(ValueError) as e:
        read_tbt(file_path)
    assert "Found BPMs with different data lengths" in str(e)
    assert all(bpm in str(e) for bpm in bpms)


def _tbt_data() -> TbtData:
    """TbT data for testing. Adding random noise, so that the data is different per BPM."""
    nturns = 2000
    bpms = ["TBPM1", "TBPM2", "TBPM3", "TBPM4"]

    return TbtData(
        matrices=[
            TransverseData(
                X=pd.DataFrame(
                    index=bpms,
                    data=create_data(
                        np.linspace(-np.pi, np.pi, nturns, endpoint=False), 
                        nbpm=len(bpms), function=np.sin, noise=0.02
                    ),
                    dtype=float,
                ),
                Y=pd.DataFrame(
                    index=bpms,
                    data=create_data(
                        np.linspace(-np.pi, np.pi, nturns, endpoint=False), 
                        nbpm=len(bpms), function=np.cos, noise=0.015
                    ),
                    dtype=float,
                ),
            )
        ],
        date=datetime.now(),
        bunch_ids=[DEFAULT_BUNCH_ID],
        nturns=nturns,
    )
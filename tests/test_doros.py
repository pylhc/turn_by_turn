from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np
import pandas as pd
import pytest

import turn_by_turn as tbt
from tests.test_lhc_and_general import compare_tbt, create_data
from turn_by_turn.doros import DEFAULT_BUNCH_ID, DataKeys, read_tbt, write_tbt
from turn_by_turn.structures import TbtData, TransverseData

if TYPE_CHECKING:
    from turn_by_turn.constants import MetaDict

INPUTS_DIR = Path(__file__).parent / "inputs"


@pytest.mark.parametrize("datatype", DataKeys.types())
def test_read_write_real_data(tmp_path, datatype):
    tbt_data = read_tbt(INPUTS_DIR / "test_doros.h5", bunch_id=10, data_type=datatype)

    assert tbt_data.nbunches == 1
    assert len(tbt_data.matrices) == 1
    assert tbt_data.nturns == 50000
    assert tbt_data.matrices[0].X.shape == (3, tbt_data.nturns)
    assert tbt_data.matrices[0].Y.shape == (3, tbt_data.nturns)
    assert len(set(tbt_data.matrices[0].X.index)) == 3
    assert np.all(tbt_data.matrices[0].X.index == tbt_data.matrices[0].Y.index)

    file_path = tmp_path / "test_file.h5"
    write_tbt(file_path, tbt_data, data_type=datatype)
    new = read_tbt(file_path, bunch_id=10, data_type=datatype)
    compare_tbt(tbt_data, new, no_binary=False)


@pytest.mark.parametrize("datatype", DataKeys.types())
def test_write_read(tmp_path, datatype):
    tbt_data = _tbt_data()
    file_path = tmp_path / "test_file.h5"
    write_tbt(file_path, tbt_data, data_type=datatype)
    new = read_tbt(file_path, data_type=datatype)
    compare_tbt(tbt_data, new, no_binary=False)


@pytest.mark.parametrize("datatype", ["doros_oscillations", "doros_positions"])
def test_write_read_via_io_module(tmp_path, datatype):
    tbt_data = _tbt_data()
    file_path = tmp_path / "test_file.h5"
    tbt.write(file_path, tbt_data, datatype=datatype)
    new = tbt.read(file_path, datatype=datatype)
    compare_tbt(tbt_data, new, no_binary=False)


def test_read_raises_different_bpm_lengths(tmp_path):
    tbt_data = _tbt_data()
    file_path = tmp_path / "test_file.h5"
    data_type = DataKeys.OSCILLATIONS
    write_tbt(file_path, tbt_data, data_type=data_type)
    keys = DataKeys.get_data_keys(data_type)

    bpm = tbt_data.matrices[0].X.index[0]

    # modify the BPM lengths in the file
    with h5py.File(file_path, "r+") as h5f:
        delta = 10
        del h5f[bpm][keys.n_samples]
        h5f[bpm][keys.n_samples] = [tbt_data.matrices[0].X.shape[1] - delta]
        for key in keys.data.values():
            data = h5f[bpm][key][:-delta]
            del h5f[bpm][key]
            h5f[bpm][key] = data

    with pytest.raises(ValueError) as e:
        read_tbt(file_path, data_type=DataKeys.OSCILLATIONS)
    assert "Not all BPMs have the same number of turns!" in str(e)


def test_read_raises_on_different_bpm_lengths_in_data(tmp_path):
    tbt_data = _tbt_data()
    file_path = tmp_path / "test_file.h5"
    data_type = DataKeys.OSCILLATIONS
    keys = DataKeys.get_data_keys(data_type)

    write_tbt(file_path, tbt_data, data_type=data_type)

    bpms = [tbt_data.matrices[0].X.index[i] for i in (0, 2)]

    # modify the BPM lengths in the file
    with h5py.File(file_path, "r+") as h5f:
        for bpm in bpms:
            del h5f[bpm][keys.n_samples]
            h5f[bpm][keys.n_samples] = [tbt_data.matrices[0].X.shape[1] + 10]

    with pytest.raises(ValueError) as e:
        read_tbt(file_path, data_type=data_type)
    assert "Found BPMs with different data lengths" in str(e)
    assert all(bpm in str(e) for bpm in bpms)


def _tbt_data() -> TbtData:
    """TbT data for testing. Adding random noise, so that the data is different per BPM."""
    nturns = 2000
    bpms = ["TBPM1", "TBPM2", "TBPM3", "TBPM4"]
    meta: MetaDict = {
        "date": datetime.now(),
    }

    return TbtData(
        matrices=[
            TransverseData(
                X=pd.DataFrame(
                    index=bpms,
                    data=create_data(
                        np.linspace(-np.pi, np.pi, nturns, endpoint=False),
                        nbpm=len(bpms),
                        function=np.sin,
                        noise=0.02,
                    ),
                    dtype=float,
                ),
                Y=pd.DataFrame(
                    index=bpms,
                    data=create_data(
                        np.linspace(-np.pi, np.pi, nturns, endpoint=False),
                        nbpm=len(bpms),
                        function=np.cos,
                        noise=0.015,
                    ),
                    dtype=float,
                ),
            )
        ],
        nturns=nturns,
        bunch_ids=[DEFAULT_BUNCH_ID],
        meta=meta,
    )

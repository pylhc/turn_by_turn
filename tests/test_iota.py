from datetime import datetime

import h5py
import numpy as np
import pandas as pd
import pytest

from tests.test_lhc_and_general import create_data, compare_tbt
from turn_by_turn import iota
from turn_by_turn.errors import HDF5VersionError
from turn_by_turn.structures import TbtData, TransverseData


def test_tbt_read_hdf5(_hdf5_file):
    origin = _hdf5_file_content()
    new = iota.read_tbt(_hdf5_file, hdf5_version=1)
    compare_tbt(origin, new, False)


def test_tbt_read_hdf5_v2(_hdf5_file_v2):
    origin = _hdf5_file_content()
    new = iota.read_tbt(_hdf5_file_v2)
    compare_tbt(origin, new, False)


def test_tbt_raises_on_wrong_hdf5_version(_hdf5_file):
    with pytest.raises(HDF5VersionError):
        new = iota.read_tbt(_hdf5_file, hdf5_version=2)


def _hdf5_file_content() -> TbtData:
    """ TbT data as had been written out to hdf5 files (see below). """
    return TbtData(
        matrices=[
            TransverseData(
                X=pd.DataFrame(
                    index=["IBPMA1C", "IBPME2R"],
                    data=create_data(np.linspace(-np.pi, np.pi, 2000, endpoint=False), 2, np.sin),
                    dtype=float,
                ),
                Y=pd.DataFrame(
                    index=["IBPMA1C", "IBPME2R"],
                    data=create_data(np.linspace(-np.pi, np.pi, 2000, endpoint=False), 2, np.cos),
                    dtype=float,
                ),
            )
        ],
        date=datetime.now(),
        bunch_ids=[1],
        nturns=2000,
    )


@pytest.fixture()
def _hdf5_file(tmp_path) -> h5py.File:
    """ IOTA File standard. """
    with h5py.File(tmp_path / "test_file.hdf5", "w") as hd5_file:
        hd5_file.create_dataset(
            "N:IBE2RH",
            data=create_data(np.linspace(-np.pi, np.pi, 2000, endpoint=False), 1, np.sin).flatten(),
        )
        hd5_file.create_dataset(
            "N:IBE2RV",
            data=create_data(np.linspace(-np.pi, np.pi, 2000, endpoint=False), 1, np.cos).flatten(),
        )
        hd5_file.create_dataset(
            "N:IBE2RS",
            data=create_data(np.linspace(0, 20, 2000, endpoint=False), 1, np.exp).flatten(),
        )

        hd5_file.create_dataset(
            "N:IBA1CH",
            data=create_data(np.linspace(-np.pi, np.pi, 2000, endpoint=False), 1, np.sin).flatten(),
        )
        hd5_file.create_dataset(
            "N:IBA1CV",
            data=create_data(np.linspace(-np.pi, np.pi, 2000, endpoint=False), 1, np.cos).flatten(),
        )
        hd5_file.create_dataset(
            "N:IBA1CS",
            data=create_data(np.linspace(0, 20, 2000, endpoint=False), 1, np.exp).flatten(),
        )
    yield tmp_path / "test_file.hdf5"


@pytest.fixture()
def _hdf5_file_v2(tmp_path) -> h5py.File:
    """ IOTA File standard. """
    with h5py.File(tmp_path / "test_file_v2.hdf5", "w") as hd5_file:
        hd5_file.create_group("A1C")
        hd5_file["A1C"].create_dataset(
            "Horizontal",
            data=create_data(np.linspace(-np.pi, np.pi, 2000, endpoint=False), 1, np.sin).flatten(),
        )
        hd5_file["A1C"].create_dataset(
            "Vertical",
            data=create_data(np.linspace(-np.pi, np.pi, 2000, endpoint=False), 1, np.cos).flatten(),
        )
        hd5_file["A1C"].create_dataset(
            "Intensity",
            data=create_data(np.linspace(0, 20, 2000, endpoint=False), 1, np.exp).flatten(),
        )

        hd5_file.create_group("E2R")
        hd5_file["E2R"].create_dataset(
            "Horizontal",
            data=create_data(np.linspace(-np.pi, np.pi, 2000, endpoint=False), 1, np.sin).flatten(),
        )
        hd5_file["E2R"].create_dataset(
            "Vertical",
            data=create_data(np.linspace(-np.pi, np.pi, 2000, endpoint=False), 1, np.cos).flatten(),
        )
        hd5_file["E2R"].create_dataset(
            "Intensity",
            data=create_data(np.linspace(0, 20, 2000, endpoint=False), 1, np.exp).flatten(),
        )
    yield tmp_path / "test_file_v2.hdf5"

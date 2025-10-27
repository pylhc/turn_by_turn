from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from tests.test_lhc_and_general import compare_tbt, create_data
from turn_by_turn import iota
from turn_by_turn.errors import HDF5VersionError
from turn_by_turn.structures import TbtData, TransverseData


def test_tbt_read_hdf5(_hdf5_file_v1, _hdf5_file_content):
    new = iota.read_tbt(_hdf5_file_v1, version=1)
    compare_tbt(_hdf5_file_content, new, no_binary=False)


def test_tbt_read_hdf5_v2(_hdf5_file_v2, _hdf5_file_content):
    new = iota.read_tbt(_hdf5_file_v2)
    compare_tbt(_hdf5_file_content, new, no_binary=False)


def test_tbt_raises_on_wrong_hdf5_version(_hdf5_file_v1, _hdf5_file_v2):
    with pytest.raises(HDF5VersionError):
        iota.read_tbt(_hdf5_file_v1, version=2)

    with pytest.raises(HDF5VersionError):
        iota.read_tbt(_hdf5_file_v2, version=1)



@pytest.fixture(scope="module")
def _hdf5_file_content() -> TbtData:
    """TbT data as had been written out to hdf5 files (see below)."""
    return TbtData(
        matrices=[
            TransverseData(
                X=pd.DataFrame(
                    index=["IBPMA1C", "IBPME2R"],
                    data=create_data(np.linspace(-np.pi, np.pi, 2000, endpoint=False), 2, np.sin, noise=0.02),
                    dtype=float,
                ),
                Y=pd.DataFrame(
                    index=["IBPMA1C", "IBPME2R"],
                    data=create_data(np.linspace(-np.pi, np.pi, 2000, endpoint=False), 2, np.cos, noise=0.015),
                    dtype=float,
                ),
            )
        ],
        bunch_ids=[1],
        nturns=2000,
    )


@pytest.fixture()
def _hdf5_file_v1(tmp_path, _hdf5_file_content) -> Path:
    """IOTA File v1 standard."""
    content: TransverseData = _hdf5_file_content.matrices[0]

    with h5py.File(tmp_path / "test_file.hdf5", "w") as hd5_file:
        hd5_file.create_dataset(
            "N:IBE2RH",
            data=content.X.loc["IBPME2R"].to_numpy(),
        )
        hd5_file.create_dataset(
            "N:IBE2RV",
            data=content.Y.loc["IBPME2R"].to_numpy(),
        )
        hd5_file.create_dataset(
            "N:IBE2RS",
            data=create_data(np.linspace(0, 20, 2000, endpoint=False), 1, np.exp).flatten(),
        )

        hd5_file.create_dataset(
            "N:IBA1CH",
            data=content.X.loc["IBPMA1C"].to_numpy(),
        )
        hd5_file.create_dataset(
            "N:IBA1CV",
            data=content.Y.loc["IBPMA1C"].to_numpy(),
        )
        hd5_file.create_dataset(
            "N:IBA1CS",
            data=create_data(np.linspace(0, 20, 2000, endpoint=False), 1, np.exp).flatten(),
        )
    return tmp_path / "test_file.hdf5"


@pytest.fixture()
def _hdf5_file_v2(tmp_path, _hdf5_file_content) -> Path:
    """IOTA File v2 standard."""
    content: TransverseData = _hdf5_file_content.matrices[0]

    with h5py.File(tmp_path / "test_file_v2.hdf5", "w") as hd5_file:
        hd5_file.create_group("A1C")
        hd5_file["A1C"].create_dataset(
            "Horizontal",
            data=content.X.loc["IBPMA1C"].to_numpy(),
        )
        hd5_file["A1C"].create_dataset(
            "Vertical",
            data=content.Y.loc["IBPMA1C"].to_numpy(),
        )
        hd5_file["A1C"].create_dataset(
            "Intensity",
            data=create_data(np.linspace(0, 20, 2000, endpoint=False), 1, np.exp).flatten(),
        )

        hd5_file.create_group("E2R")
        hd5_file["E2R"].create_dataset(
            "Horizontal",
            data=content.X.loc["IBPME2R"].to_numpy(),
        )
        hd5_file["E2R"].create_dataset(
            "Vertical",
            data=content.Y.loc["IBPME2R"].to_numpy(),
        )
        hd5_file["E2R"].create_dataset(
            "Intensity",
            data=create_data(np.linspace(0, 20, 2000, endpoint=False), 1, np.exp).flatten(),
        )
    return tmp_path / "test_file_v2.hdf5"

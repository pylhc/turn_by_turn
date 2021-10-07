from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from turn_by_turn.constants import PLANES, PRINT_PRECISION
from turn_by_turn.errors import DataTypeError, ExclusiveArgumentsError, HDF5VersionError, PTCFormatError
from turn_by_turn.io import read_tbt, write_lhc_ascii, write_tbt
from turn_by_turn.readers import iota, ptc, trackone
from turn_by_turn.structures import TbtData
from turn_by_turn.utils import add_noise, generate_average_tbtdata

INPUTS_DIR = Path(__file__).parent / "inputs"
ASCII_PRECISION = 0.5 / np.power(10, PRINT_PRECISION)


@pytest.mark.parametrize("datatype", ["invalid", "not_supported"])
def test_tbt_read_raises_on_invalid_datatype(_sdds_file, caplog, datatype):
    with pytest.raises(DataTypeError):
        _ = read_tbt(_sdds_file, datatype=datatype)

    for record in caplog.records:
        assert record.levelname == "ERROR"


def test_tbt_write_read_sdds_binary(_sdds_file, _test_file):
    origin = read_tbt(_sdds_file)
    write_tbt(_test_file, origin)
    new = read_tbt(f"{_test_file}.sdds")
    _compare_tbt(origin, new, False)


def test_tbt_write_read_sdds_binary_with_noise(_sdds_file, _test_file):
    origin = read_tbt(_sdds_file)
    write_tbt(_test_file, origin, noise=2)
    new = read_tbt(f"{_test_file}.sdds")

    with pytest.raises(AssertionError):  # should be different
        _compare_tbt(origin, new, False)


def test_tbt_read_hdf5(_hdf5_file):

    origin = TbtData(
        matrices=[
            {
                "X": pd.DataFrame(
                    index=["IBPMA1C", "IBPME2R"],
                    data=_create_data(np.linspace(-np.pi, np.pi, 2000, endpoint=False), 2, np.sin),
                    dtype=float,
                ),
                "Y": pd.DataFrame(
                    index=["IBPMA1C", "IBPME2R"],
                    data=_create_data(np.linspace(-np.pi, np.pi, 2000, endpoint=False), 2, np.cos),
                    dtype=float,
                ),
            }
        ],
        date=datetime.now(),
        bunch_ids=[1],
        nturns=2000,
    )
    new = iota.read_tbt(_hdf5_file, hdf5_version=1)
    _compare_tbt(origin, new, False)


def test_tbt_read_hdf5_v2(_hdf5_file_v2):

    origin = TbtData(
        matrices=[
            {
                "X": pd.DataFrame(
                    index=["IBPMA1C", "IBPME2R"],
                    data=_create_data(np.linspace(-np.pi, np.pi, 2000, endpoint=False), 2, np.sin),
                    dtype=float,
                ),
                "Y": pd.DataFrame(
                    index=["IBPMA1C", "IBPME2R"],
                    data=_create_data(np.linspace(-np.pi, np.pi, 2000, endpoint=False), 2, np.cos),
                    dtype=float,
                ),
            }
        ],
        date=datetime.now(),
        bunch_ids=[1],
        nturns=2000,
    )
    new = iota.read_tbt(_hdf5_file_v2)
    _compare_tbt(origin, new, False)


def test_tbt_raises_on_wrong_hdf5_version(_hdf5_file):
    with pytest.raises(HDF5VersionError):
        new = iota.read_tbt(_hdf5_file, hdf5_version=2)


def test_compare_average_Tbtdata():
    npart = 10
    data = {
        plane: np.concatenate(
            [
                [
                    _create_data(
                        np.linspace(1, 10, 10, endpoint=False, dtype=int),
                        2,
                        (lambda x: np.random.randn(len(x))),
                    )
                ]
                for _ in range(npart)
            ],
            axis=0,
        )
        for plane in PLANES
    }

    origin = TbtData(
        matrices=[
            {
                "X": pd.DataFrame(index=["IBPMA1C", "IBPME2R"], data=data["X"][i], dtype=float),
                "Y": pd.DataFrame(index=["IBPMA1C", "IBPME2R"], data=data["Y"][i], dtype=float),
            }
            for i in range(npart)
        ],
        date=datetime.now(),
        bunch_ids=range(npart),
        nturns=10,
    )

    new = TbtData(
        matrices=[
            {
                "X": pd.DataFrame(index=["IBPMA1C", "IBPME2R"], data=np.mean(data["X"], axis=0), dtype=float),
                "Y": pd.DataFrame(index=["IBPMA1C", "IBPME2R"], data=np.mean(data["Y"], axis=0), dtype=float),
            }
        ],
        date=datetime.now(),
        bunch_ids=[1],
        nturns=10,
    )

    _compare_tbt(generate_average_tbtdata(origin), new, False)


def test_tbt_read_ptc(_ptc_file):
    new = ptc.read_tbt(_ptc_file)
    origin = _original_trackone()
    _compare_tbt(origin, new, True)


def test_tbt_read_ptc_raises_on_invalid_file(_invalid_ptc_file):
    with pytest.raises(PTCFormatError):
        _ = ptc.read_tbt(_invalid_ptc_file)


def test_tbt_read_ptc_defaults_date(_ptc_file_no_date):
    new = ptc.read_tbt(_ptc_file_no_date)
    assert new.date.day == datetime.today().day
    assert new.date.tzname() == "UTC"


def test_tbt_read_trackone(_ptc_file):
    new = trackone.read_tbt(_ptc_file)
    origin = _original_trackone(True)
    _compare_tbt(origin, new, True)


def test_tbt_read_ptc_sci(_ptc_file_sci):
    new = ptc.read_tbt(_ptc_file_sci)
    origin = _original_trackone()
    _compare_tbt(origin, new, True)


def test_tbt_read_trackone_sci(_ptc_file_sci):
    new = trackone.read_tbt(_ptc_file_sci)
    origin = _original_trackone(True)
    _compare_tbt(origin, new, True)


def test_tbt_read_ptc_looseparticles(_ptc_file_losses):
    new = ptc.read_tbt(_ptc_file_losses)
    assert len(new.matrices) == 3
    assert len(new.matrices[0]["X"].columns) == 9
    assert all(new.matrices[0]["X"].index == np.array([f"BPM{i+1}" for i in range(3)]))
    assert not new.matrices[0]["X"].isna().any().any()


def test_tbt_read_trackone_looseparticles(_ptc_file_losses):
    new = trackone.read_tbt(_ptc_file_losses)
    assert len(new.matrices) == 3
    assert len(new.matrices[0]["X"].columns) == 9
    assert all(new.matrices[0]["X"].index == np.array([f"BPM{i+1}" for i in range(3)]))
    assert not new.matrices[0]["X"].isna().any().any()


def test_tbt_write_read_ascii(_sdds_file, _test_file):
    origin = read_tbt(_sdds_file)
    write_lhc_ascii(_test_file, origin)
    new = read_tbt(_test_file)
    _compare_tbt(origin, new, True)


def test_noise_addition():
    array = _create_data(np.linspace(-np.pi, np.pi, 2000, endpoint=False), 1, np.sin).flatten()
    noised = add_noise(array, noise=0)
    np.testing.assert_array_equal(array, noised)

    noised = add_noise(array, sigma=0)
    np.testing.assert_array_equal(array, noised)

    noised = add_noise(array, noise=5)
    assert np.std(array) != np.std(noised)
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(array, noised)

    noised = add_noise(array, sigma=1)
    assert np.std(array) != np.std(noised)
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(array, noised)


def test_add_noise_raises_on_both_arguments():
    array = _create_data(np.linspace(-np.pi, np.pi, 2000, endpoint=False), 1, np.sin).flatten()
    with pytest.raises(ExclusiveArgumentsError):
        _ = add_noise(array, noise=5, sigma=1)


# ----- Helpers ----- #


def _compare_tbt(origin: TbtData, new: TbtData, no_binary: bool, max_deviation=ASCII_PRECISION) -> None:
    assert new.nturns == origin.nturns
    assert new.nbunches == origin.nbunches
    assert new.bunch_ids == origin.bunch_ids
    for index in range(origin.nbunches):
        for plane in PLANES:
            assert np.all(new.matrices[index][plane].index == origin.matrices[index][plane].index)
            origin_mat = origin.matrices[index][plane].to_numpy()
            new_mat = new.matrices[index][plane].to_numpy()
            if no_binary:
                assert np.max(np.abs(origin_mat - new_mat)) < max_deviation
            else:
                assert np.all(origin_mat == new_mat)


def _original_trackone(track: bool = False) -> TbtData:
    names = np.array(["C1.BPM1"])
    matrix = [
        dict(
            X=pd.DataFrame(index=names, data=[[0.001, -0.0003606, -0.00165823, -0.00266631]]),
            Y=pd.DataFrame(index=names, data=[[0.001, 0.00070558, -0.00020681, -0.00093807]]),
        ),
        dict(
            X=pd.DataFrame(index=names, data=[[0.0011, -0.00039666, -0.00182406, -0.00293294]]),
            Y=pd.DataFrame(index=names, data=[[0.0011, 0.00077614, -0.00022749, -0.00103188]]),
        ),
    ]
    origin = TbtData(matrix, None, [0, 1] if track else [1, 2], 4)
    return origin


def _create_data(nturns, nbpm, function) -> np.ndarray:
    return np.ones((nbpm, len(nturns))) * function(nturns)


@pytest.fixture()
def _hdf5_file(tmp_path) -> h5py.File:
    with h5py.File(tmp_path / "test_file.hdf5", "w") as hd5_file:
        hd5_file.create_dataset(
            "N:IBE2RH",
            data=_create_data(np.linspace(-np.pi, np.pi, 2000, endpoint=False), 1, np.sin).flatten(),
        )
        hd5_file.create_dataset(
            "N:IBE2RV",
            data=_create_data(np.linspace(-np.pi, np.pi, 2000, endpoint=False), 1, np.cos).flatten(),
        )
        hd5_file.create_dataset(
            "N:IBE2RS",
            data=_create_data(np.linspace(0, 20, 2000, endpoint=False), 1, np.exp).flatten(),
        )

        hd5_file.create_dataset(
            "N:IBA1CH",
            data=_create_data(np.linspace(-np.pi, np.pi, 2000, endpoint=False), 1, np.sin).flatten(),
        )
        hd5_file.create_dataset(
            "N:IBA1CV",
            data=_create_data(np.linspace(-np.pi, np.pi, 2000, endpoint=False), 1, np.cos).flatten(),
        )
        hd5_file.create_dataset(
            "N:IBA1CS",
            data=_create_data(np.linspace(0, 20, 2000, endpoint=False), 1, np.exp).flatten(),
        )
    yield tmp_path / "test_file.hdf5"


@pytest.fixture()
def _hdf5_file_v2(tmp_path) -> h5py.File:
    with h5py.File(tmp_path / "test_file_v2.hdf5", "w") as hd5_file:
        hd5_file.create_group("A1C")
        hd5_file["A1C"].create_dataset(
            "Horizontal",
            data=_create_data(np.linspace(-np.pi, np.pi, 2000, endpoint=False), 1, np.sin).flatten(),
        )
        hd5_file["A1C"].create_dataset(
            "Vertical",
            data=_create_data(np.linspace(-np.pi, np.pi, 2000, endpoint=False), 1, np.cos).flatten(),
        )
        hd5_file["A1C"].create_dataset(
            "Intensity",
            data=_create_data(np.linspace(0, 20, 2000, endpoint=False), 1, np.exp).flatten(),
        )

        hd5_file.create_group("E2R")
        hd5_file["E2R"].create_dataset(
            "Horizontal",
            data=_create_data(np.linspace(-np.pi, np.pi, 2000, endpoint=False), 1, np.sin).flatten(),
        )
        hd5_file["E2R"].create_dataset(
            "Vertical",
            data=_create_data(np.linspace(-np.pi, np.pi, 2000, endpoint=False), 1, np.cos).flatten(),
        )
        hd5_file["E2R"].create_dataset(
            "Intensity",
            data=_create_data(np.linspace(0, 20, 2000, endpoint=False), 1, np.exp).flatten(),
        )
    yield tmp_path / "test_file_v2.hdf5"


@pytest.fixture()
def _test_file(tmp_path) -> Path:
    yield tmp_path / "test_file"


@pytest.fixture()
def _sdds_file() -> Path:
    return INPUTS_DIR / "test_file.sdds"


@pytest.fixture()
def _ptc_file() -> Path:
    return INPUTS_DIR / "test_trackone"


@pytest.fixture()
def _invalid_ptc_file() -> Path:
    return INPUTS_DIR / "test_wrong_ptc"


@pytest.fixture()
def _ptc_file_no_date() -> Path:
    return INPUTS_DIR / "test_trackone_no_date"


@pytest.fixture()
def _ptc_file_losses() -> Path:
    return INPUTS_DIR / "test_trackone_losses"


@pytest.fixture()
def _ptc_file_sci() -> Path:
    return INPUTS_DIR / "test_trackone_sci"

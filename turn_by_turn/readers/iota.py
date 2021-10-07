"""
Iota
----

Data handling for turn-by-turn measurement files from ``Iota`` (files in hdf5 format).
"""
import logging
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Union

import h5py
import numpy as np
import pandas as pd

from turn_by_turn.constants import PLANES
from turn_by_turn.structures import TbtData

LOGGER = logging.getLogger()

VERSIONS = (1, 2)
PLANES_CONV: Dict[int, Dict[str, str]] = {
    1: {"X": "H", "Y": "V"},
    2: {"X": "Horizontal", "Y": "Vertical"},
}


def read_tbt(file_path: Union[str, Path]) -> TbtData:
    """
    Reads turn-by-turn data from ``IOITA``'s **hdf5** format file.
    As there are 2 possible versions of the HDF5 format, this will try them both successively.

    Args:
        file_path (Union[str, Path]): path to the turn-by-turn measurement file.

    Returns:
        A ``TbTData`` object with the loaded data.
    """
    file_path = Path(file_path)
    LOGGER.debug(f"Reading Iota file at path: '{file_path.absolute()}'")
    hdf_file = h5py.File(file_path, "r")
    bunch_ids = [1]
    date = datetime.now()

    for version in VERSIONS[::-1]:
        try:
            bpm_names = FUNCTIONS[version]["get_bpm_names"](hdf_file)
            nturns = FUNCTIONS[version]["get_nturns"](hdf_file, version)
            matrices = [
                {
                    plane: pd.DataFrame(
                        index=bpm_names,
                        data=FUNCTIONS[version]["get_tbtdata"](hdf_file, plane, version),
                        dtype=float,
                    )
                    for plane in PLANES
                }
            ]
            return TbtData(matrices, date, bunch_ids, nturns)

        except TypeError as error:
            LOGGER.error("An unhandled TypeError occured during reading.")
        except KeyError as error:
            LOGGER.error("An unhandled KeyError occured during reading.")


def _get_turn_by_turn_data_v1(hdf5_v1_file: h5py.File, plane: str, version: int) -> np.ndarray:
    """Go through the file to determine the turn-by-turn data as a numpy array, for an hdf5 v1 file."""
    keys = [key for key in hdf5_v1_file.keys() if (key.endswith(PLANES_CONV[version][plane]))]
    nbpm = len(keys)
    nturn = FUNCTIONS[version]["get_nturns"](hdf5_v1_file, version)
    data = np.zeros((nbpm, nturn))
    for i, key in enumerate(keys):
        data[i, :] = hdf5_v1_file[key][:nturn]
    return data


def _get_list_of_bpmnames_v1(hdf5_v1_file: h5py.File) -> np.ndarray:
    """Go through the file to determine the list of BPMs, for an hdf5 v1 file."""
    bpms = [f"IBPM{key[4:-1]}" for key in list(hdf5_v1_file.keys()) if check_key_v1(key)]
    return np.unique(bpms)


def _get_number_of_turns_v1(hdf5_v1_file: h5py.File, version: int) -> int:
    """Go through the file to determine the number of turns, for an hdf5 v1 file."""
    lengths = [len(hdf5_v1_file[key]) for key in list(hdf5_v1_file.keys()) if check_key_v1(key)]
    return np.min(lengths)


def _get_turn_by_turn_data_v2(hdf5_v2_file: h5py.File, plane: str, version: int) -> np.ndarray:
    """Go through the file to determine the turn-by-turn data as a numpy array, for an hdf5 v2 file."""
    keys = [key for key in hdf5_v2_file.keys() if not key.startswith("N:")]
    if not keys:
        raise TypeError("Wrong version of converter was used.")
    nbpm = len(keys)
    nturn = FUNCTIONS[version]["get_nturns"](hdf5_v2_file, version)
    data = np.zeros((nbpm, nturn))
    for i, key in enumerate(keys):
        data[i, :] = hdf5_v2_file[key][PLANES_CONV[version][plane]][:nturn]

    return data


def _get_list_of_bpmnames_v2(hdf5_v2_file: h5py.File) -> np.ndarray:
    """Go through the file to determine the list of BPMs, for an hdf5 v2 file."""
    bpms = [f"IBPM{key}" for key in list(hdf5_v2_file.keys()) if check_key_v2(key)]
    if not bpms:
        raise TypeError("Wrong version of converter was used.")
    return np.unique(bpms)


def _get_number_of_turns_v2(hdf5_v2_file: h5py.File, version: int) -> int:
    """Go through the file to determine the number of turns, for an hdf5 v2 file."""
    lengths = np.array(
        [
            (
                len(hdf5_v2_file[key][PLANES_CONV[version]["X"]]),
                len(hdf5_v2_file[key][PLANES_CONV[version]["Y"]]),
            )
            for key in list(hdf5_v2_file.keys())
            if check_key_v2(key)
        ]
    )
    return np.min(lengths)


def check_key_v2(key) -> bool:
    return not (("NL" in key) or key.startswith("N:"))


def check_key_v1(key) -> bool:
    return ("state" not in key) or key.startswith("N:")


FUNCTIONS: Dict[int, Dict[str, Callable]] = {
    1: {
        "get_bpm_names": _get_list_of_bpmnames_v1,
        "get_nturns": _get_number_of_turns_v1,
        "get_tbtdata": _get_turn_by_turn_data_v1,
    },
    2: {
        "get_bpm_names": _get_list_of_bpmnames_v2,
        "get_nturns": _get_number_of_turns_v2,
        "get_tbtdata": _get_turn_by_turn_data_v2,
    },
}

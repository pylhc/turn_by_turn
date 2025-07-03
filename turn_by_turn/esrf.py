"""
ESRF
----

Data handling for turn-by-turn measurement files from ``ESRF`` (files in **matlab** format).
This module is untested and should be considered experimental at the moment.
"""

import json
import logging
from pathlib import Path
from typing import Tuple, Union

import numpy as np
from scipy.io import loadmat

from turn_by_turn.structures import TbtData
from turn_by_turn.utils import numpy_to_tbt

BPM_NAMES_FILE: str = "bpm_names.json"
LOGGER = logging.getLogger(__name__)


def read_tbt(file_path: Union[str, Path]) -> TbtData:
    """
    Reads turn-by-turn data from the ``ESRF``'s **Matlab** format file.

    Args:
        file_path (Union[str, Path]): path to the turn-by-turn measurement file.

    Returns:
        A ``TbTData`` object with the loaded data.
    """
    file_path = Path(file_path)
    LOGGER.debug(f"Reading ESRF file at path: '{file_path.absolute()}'")
    names, matrix = load_esrf_mat_file(file_path)
    return numpy_to_tbt(names, matrix)


def load_esrf_mat_file(infile: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads the **ESRF** TbT ``Matlab`` file, checks for nans and matrices duplicities from consecutive kicks.

    Args:
        infile (Union[str, Path]): path to the turn-by-turn measurement file.

    Returns:
        A 1D numpy array of BPM names and a 4D Numpy array [quantity, BPM, particle/bunch No.,
        turn No.] quantities in order [x, y]
    """
    esrf_data = loadmat(infile)  # accepts str or pathlib.Path
    hor, ver = esrf_data["allx"], esrf_data["allz"]

    if hor.shape != ver.shape:
        LOGGER.error("Number of turns, BPMs or measurements in X and Y do not match")
        raise ValueError("Unequal transverse data shape")

    # TODO change for tfs file got from accelerator class
    # Need input from someone with ESRF files experience, where exactly should we look for this file?
    bpm_names = json.loads((Path(infile).parent / BPM_NAMES_FILE).read_text())  # weird?

    if hor.shape[1] != len(bpm_names):
        LOGGER.error("Number of bpms does not match with accelerator class")
        raise ValueError("Mismatch to model!")

    tbt_data = _check_esrf_tbt_data(np.transpose(np.array([hor, ver]), axes=[0, 2, 3, 1]))
    return np.array(bpm_names), tbt_data


def _check_esrf_tbt_data(tbt_data: np.ndarray) -> np.ndarray:
    tbt_data[np.isnan(np.sum(tbt_data, axis=3)), :] = 0.0
    # check if contains the same matrices as in previous kick
    mask_prev = (
        np.concatenate(
            (
                np.ones((tbt_data.shape[0], tbt_data.shape[1], 1)),
                np.sum(np.abs(np.diff(tbt_data, axis=2)), axis=3),
            ),
            axis=2,
        )
        == 0.0
    )
    tbt_data[mask_prev, :] = 0.0
    return tbt_data

"""
IO
--

This module contains high-level I/O functions to read and write tur-by-turn data objects in different
formats. While data can be loaded from the formats of different machines / codes, each format getting its
own reader module, writing functionality is at the moment always done in the ``LHC``'s **SDDS** format.
"""
import logging
from pathlib import Path
from typing import Union

import numpy as np
import sdds

from turn_by_turn import esrf, iota, lhc, ptc, trackone, sps
from turn_by_turn.constants import PLANE_TO_NUM, PLANES
from turn_by_turn.errors import DataTypeError
from turn_by_turn.structures import TbtData
from turn_by_turn.utils import add_noise
from turn_by_turn.ascii import write_ascii

LOGGER = logging.getLogger()

write_lhc_ascii = write_ascii  # Backwards compatibility <0.4
DATA_READERS = dict(
    lhc=lhc,
    sps=sps,
    iota=iota,
    esrf=esrf,
    ptc=ptc,
    trackone=trackone,
)


def read_tbt(file_path: Union[str, Path], datatype: str = "lhc") -> TbtData:
    """
    Calls the appropriate loader for the provided matrices type and returns a ``TbtData`` object of the
    loaded matrices.

    Args:
        file_path (Union[str, Path]): path to a file containing TbtData.
        datatype (str): type of matrices in the file, determines the reader to use. Case-insensitive,
            defaults to ``lhc``.

    Returns:
        A ``TbtData`` object with the loaded matrices.
    """
    file_path = Path(file_path)
    LOGGER.info(f"Loading turn-by-turn matrices from '{file_path}'")
    try:
        return DATA_READERS[datatype.lower()].read_tbt(file_path)
    except KeyError as error:
        LOGGER.exception(
            f"Unsupported datatype '{datatype}' was provided, should be one of {list(DATA_READERS.keys())}"
        )
        raise DataTypeError(datatype) from error


def write_tbt(output_path: Union[str, Path], tbt_data: TbtData, noise: float = None, seed: int = None) -> None:
    """
    Write a ``TbtData`` object's data to file, in the ``LHC``'s **SDDS** format.

    Args:
        output_path (Union[str, Path]): path to a the disk location where to write the data.
        tbt_data (TbtData): the ``TbtData`` object to write to disk.
        noise (float): optional noise to add to the data.
        seed (int): A given seed to initialise the RNG if one chooses to add noise. This is useful
            to ensure the exact same RNG state across operations. Defaults to `None`, which means
            any new RNG operation in noise addition will pull fresh entropy from the OS.
    """
    output_path = Path(output_path)
    LOGGER.info(f"Writing TbTdata in binary SDDS (LHC) format at '{output_path.absolute()}'")

    data: np.ndarray = _matrices_to_array(tbt_data)

    if noise is not None:
        data = add_noise(data, noise, seed=seed)

    definitions = [
        sdds.classes.Parameter(lhc.ACQ_STAMP, "llong"),
        sdds.classes.Parameter(lhc.N_BUNCHES, "long"),
        sdds.classes.Parameter(lhc.N_TURNS, "long"),
        sdds.classes.Array(lhc.BUNCH_ID, "long"),
        sdds.classes.Array(lhc.BPM_NAMES, "string"),
        sdds.classes.Array(lhc.POSITIONS["X"], "float"),
        sdds.classes.Array(lhc.POSITIONS["Y"], "float"),
    ]
    values = [
        tbt_data.date.timestamp() * 1e9,
        tbt_data.nbunches,
        tbt_data.nturns,
        tbt_data.bunch_ids,
        tbt_data.matrices[0].X.index.to_numpy(),
        np.ravel(data[PLANE_TO_NUM["X"]]),
        np.ravel(data[PLANE_TO_NUM["Y"]]),
    ]
    sdds.write(sdds.SddsFile("SDDS1", None, definitions, values), f"{output_path}.sdds")


def _matrices_to_array(tbt_data: TbtData) -> np.ndarray:
    """
    Convert  the matrices of a ``TbtData`` object to a numpy array.

    Args:
        tbt_data (TbtData): ``TbtData`` object to convert the data from.

    Returns:
        A numpy array with the matrices data.
    """
    LOGGER.debug("Getting number of BPMs from the measurement data.")
    n_bpms = tbt_data.matrices[0].X.index.size
    data = np.empty((2, n_bpms, tbt_data.nbunches, tbt_data.nturns), dtype=float)

    LOGGER.debug("Converting matrices data ")
    for index in range(tbt_data.nbunches):
        for plane in PLANES:
            data[PLANE_TO_NUM[plane], :, index, :] = tbt_data.matrices[index][plane].to_numpy()
    return data

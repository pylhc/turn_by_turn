"""
IO
--

This module contains high-level I/O functions to read and write tur-by-turn data objects in different
formats. While data can be loaded from the formats of different machines / codes, each format getting its
own reader module, writing functionality is at the moment always done in the ``LHC``'s **SDDS** format.
"""
import logging
from datetime import datetime
from pathlib import Path
from typing import TextIO, Union

import numpy as np
import sdds

from turn_by_turn.constants import FORMAT_STRING, PLANE_TO_NUM, PLANES
from turn_by_turn.errors import DataTypeError
from turn_by_turn import esrf, iota, lhc, ptc, trackone
from turn_by_turn.structures import TbtData
from turn_by_turn.utils import add_noise

LOGGER = logging.getLogger()

DATA_READERS = dict(
    lhc=lhc,
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


def write_tbt(output_path: Union[str, Path], tbt_data: TbtData, noise: float = None) -> None:
    """
    Write a ``TbtData`` object's data to file, in the ``LHC``'s **SDDS** format.

    Args:
        output_path (Union[str, Path]): path to a the disk location where to write the data.
        tbt_data (TbtData): the ``TbtData`` object to write to disk.
        noise (float): optional noise to add to the data.
    """
    output_path = Path(output_path)
    LOGGER.info(f"Writing TbTdata in binary SDDS (LHC) format at '{output_path.absolute()}'")

    data: np.ndarray = _matrices_to_array(tbt_data)

    if noise is not None:
        data = add_noise(data, noise)

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


def write_lhc_ascii(output_path: Union[str, Path], tbt_data: TbtData) -> None:
    """
    Write a ``TbtData`` object's data to file, in the ASCII **SDDS** format.

    Args:
        output_path (Union[str, Path]): path to a the disk locatino where to write the data.
        tbt_data (TbtData): the ``TbtData`` object to write to disk.
    """
    output_path = Path(output_path)
    LOGGER.info(f"Writing TbTdata in ASCII SDDS (LHC) format at '{output_path.absolute()}'")

    for bunch_id in range(tbt_data.nbunches):
        LOGGER.debug(f"Writing data for bunch {bunch_id}")
        suffix = f"_{tbt_data.bunch_ids[bunch_id]}" if tbt_data.nbunches > 1 else ""
        with output_path.with_suffix(suffix).open("w") as output_file:
            _write_header(tbt_data, bunch_id, output_file)
            _write_tbt_data(tbt_data, bunch_id, output_file)


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


def _write_header(tbt_data: TbtData, bunch_id: int, output_file: TextIO) -> None:
    """
    Write the appropriate headers for a ``TbtData`` object's given bunch_id in the ASCII **SDDS**  format.
    """
    output_file.write("#SDDSASCIIFORMAT v1\n")
    output_file.write(
        f"#Created: {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')} By: Python turn_by_turn Package\n"
    )
    output_file.write(f"#Number of turns: {tbt_data.nturns}\n")
    output_file.write(f"#Number of horizontal monitors: {tbt_data.matrices[bunch_id].X.index.size}\n")
    output_file.write(f"#Number of vertical monitors: {tbt_data.matrices[bunch_id].Y.index.size}\n")
    output_file.write(f"#Acquisition date: {tbt_data.date.strftime('%Y-%m-%d at %H:%M:%S')}\n")


def _write_tbt_data(tbt_data: TbtData, bunch_id: int, output_file: TextIO) -> None:
    """Write a ``TbtData`` object's data for the given bunch_id to disk in the ASCII **SDDS** format."""
    row_format = "{} {} {}  " + FORMAT_STRING * tbt_data.nturns + "\n"
    for plane in PLANES:
        for bpm_index, bpm_name in enumerate(tbt_data.matrices[bunch_id][plane].index):
            samples = tbt_data.matrices[bunch_id][plane].loc[bpm_name, :].to_numpy()
            output_file.write(row_format.format(PLANE_TO_NUM[plane], bpm_name, bpm_index, *samples))

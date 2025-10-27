"""
LHC
---

Data handling for turn-by-turn measurement files from the ``LHC`` (files in **SDDS** format).
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sdds
from dateutil import tz

from turn_by_turn.ascii import is_ascii_file
from turn_by_turn.ascii import read_tbt as read_ascii
from turn_by_turn.constants import PLANE_TO_NUM, PLANES, MetaDict
from turn_by_turn.structures import TbtData, TransverseData
from turn_by_turn.utils import matrices_to_array

LOGGER = logging.getLogger(__name__)

# IDs
N_BUNCHES: str = "nbOfCapBunches"
BUNCH_ID: str = "BunchId"
HOR_BUNCH_ID: str = "horBunchId"
N_TURNS: str = "nbOfCapTurns"
ACQ_STAMP: str = "acqStamp"
BPM_NAMES: str = "bpmNames"


POSITIONS: dict[str, str] = {
    "X": "horPositionsConcentratedAndSorted",
    "Y": "verPositionsConcentratedAndSorted",
}


def read_tbt(file_path: str | Path) -> TbtData:
    """
    Reads turn-by-turn data from the ``LHC``'s **SDDS** format file.
    Will first determine if it is in ASCII format to figure out which reading method to use.

    Args:
        file_path (Union[str, Path]): path to the turn-by-turn measurement file.

    Returns:
        A ``TbTData`` object with the loaded data.
    """
    file_path = Path(file_path)
    LOGGER.debug(f"Reading LHC file at path: '{file_path.absolute()}'")

    if is_ascii_file(file_path):
        return read_ascii(file_path)

    sdds_file = sdds.read(file_path)
    nbunches = sdds_file.values[N_BUNCHES]
    bunch_ids = sdds_file.values[BUNCH_ID if BUNCH_ID in sdds_file.values else HOR_BUNCH_ID]

    if len(bunch_ids) > nbunches:
        bunch_ids = bunch_ids[:nbunches]

    nturns = sdds_file.values[N_TURNS]
    date = datetime.fromtimestamp(sdds_file.values[ACQ_STAMP] / 1e9, tz=tz.tzutc())
    bpm_names = sdds_file.values[BPM_NAMES]
    nbpms = len(bpm_names)
    data = {k: sdds_file.values[POSITIONS[k]].reshape((nbpms, nbunches, nturns)) for k in PLANES}

    matrices = [
        TransverseData(
            X=pd.DataFrame(index=bpm_names, data=data["X"][:, idx, :], dtype=float),
            Y=pd.DataFrame(index=bpm_names, data=data["Y"][:, idx, :], dtype=float),
        )
        for idx in range(nbunches)
    ]

    meta: MetaDict = {
        "file": file_path,
        "source_datatype": "lhc",
        "date": date,
    }
    return TbtData(matrices, nturns=nturns, bunch_ids=bunch_ids, meta=meta)


def write_tbt(output_path: str | Path, tbt_data: TbtData) -> None:
    """
    Write a ``TbtData`` object's data to file, in the ``LHC``'s **SDDS** format.

    Args:
        output_path (Union[str, Path]): path to a the disk location where to write the data.
        tbt_data (TbtData): the ``TbtData`` object to write to disk.
    """
    output_path = Path(output_path)
    LOGGER.info(f"Writing TbTdata in binary SDDS (LHC) format at '{output_path.absolute()}'")

    data: np.ndarray = matrices_to_array(tbt_data)

    definitions = [
        sdds.classes.Parameter(ACQ_STAMP, "llong"),
        sdds.classes.Parameter(N_BUNCHES, "long"),
        sdds.classes.Parameter(N_TURNS, "long"),
        sdds.classes.Array(BUNCH_ID, "long"),
        sdds.classes.Array(BPM_NAMES, "string"),
        sdds.classes.Array(POSITIONS["X"], "float"),
        sdds.classes.Array(POSITIONS["Y"], "float"),
    ]
    values = [
        tbt_data.meta.get("date", datetime.now(tz=tz.tzutc())).timestamp() * 1e9,
        tbt_data.nbunches,
        tbt_data.nturns,
        tbt_data.bunch_ids,
        tbt_data.matrices[0].X.index.to_numpy(),
        np.ravel(data[PLANE_TO_NUM["X"]]),
        np.ravel(data[PLANE_TO_NUM["Y"]]),
    ]
    sdds.write(sdds.SddsFile("SDDS1", None, definitions, values), output_path)

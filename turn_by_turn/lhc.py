"""
LHC
---

Data handling for turn-by-turn measurement files from the ``LHC`` (files in **SDDS** format).
"""
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Union

import pandas as pd
import sdds
from dateutil import tz

from turn_by_turn.ascii import is_ascii_file, read_ascii
from turn_by_turn.constants import PLANES

from turn_by_turn.structures import TbtData, TransverseData

LOGGER = logging.getLogger()

# IDs
N_BUNCHES: str = "nbOfCapBunches"
BUNCH_ID: str = "BunchId"
HOR_BUNCH_ID: str = "horBunchId"
N_TURNS: str = "nbOfCapTurns"
ACQ_STAMP: str = "acqStamp"
BPM_NAMES: str = "bpmNames"


POSITIONS: Dict[str, str] = {
    "X": "horPositionsConcentratedAndSorted",
    "Y": "verPositionsConcentratedAndSorted",
}


def read_tbt(file_path: Union[str, Path]) -> TbtData:
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
        matrices, date = read_ascii(file_path)
        return TbtData(matrices, date, [0], matrices[0].X.shape[1])

    sdds_file = sdds.read(file_path)
    nbunches = sdds_file.values[N_BUNCHES]
    bunch_ids = sdds_file.values[
        BUNCH_ID if BUNCH_ID in sdds_file.values else HOR_BUNCH_ID
    ]

    if len(bunch_ids) > nbunches:
        bunch_ids = bunch_ids[:nbunches]

    nturns = sdds_file.values[N_TURNS]
    date = datetime.utcfromtimestamp(sdds_file.values[ACQ_STAMP] / 1e9).replace(
        tzinfo=tz.tzutc()
    )
    bpm_names = sdds_file.values[BPM_NAMES]
    nbpms = len(bpm_names)
    data = {
        k: sdds_file.values[POSITIONS[k]].reshape((nbpms, nbunches, nturns))
        for k in PLANES
    }
    matrices = [
        TransverseData(
            X=pd.DataFrame(index=bpm_names, data=data["X"][:, idx, :], dtype=float),
            Y=pd.DataFrame(index=bpm_names, data=data["Y"][:, idx, :], dtype=float),
        )
        for idx in range(nbunches)
    ]
    return TbtData(matrices, date, bunch_ids, nturns)

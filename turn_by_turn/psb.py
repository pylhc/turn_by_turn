"""
PSB
---

Data handling for turn-by-turn measurement files from the ``PSB`` (Proton Synchrotron Booster)
(files in **SDDS** format).

.. note::
    The PSB file structure and reader behavior may change after the CERN LS3.
    Treat this reader as subject to updates (or removal) when new post-shutdown data format is known.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import sdds
from dateutil import tz

from turn_by_turn.constants import PLANES, MetaDict
from turn_by_turn.structures import TbtData, TransverseData

LOGGER = logging.getLogger(__name__)

# IDs
N_BUNCHES: str = "nbOfCapBunches"
BUNCH_ID: str = "BunchId"
HOR_BUNCH_ID: str = "horBunchId"
N_TURNS: str = "nbOfCapTurns"
ACQ_STAMP: str = "acqStampMilli"  # PSB uses milliseconds, different from LHC
BPM_NAMES: str = "bpmNames"


POSITIONS: dict[str, str] = {
    "X": "horPositionsConcentratedAndSorted",
    "Y": "verPositionsConcentratedAndSorted",
}


def read_tbt(file_path: str | Path) -> TbtData:
    """
    Reads turn-by-turn data from the ``PSB``'s **SDDS** format file.

    Args:
        file_path (Union[str, Path]): path to the turn-by-turn measurement file.

    Returns:
        A ``TbtData`` object with the loaded data.
    """
    file_path = Path(file_path)
    LOGGER.debug(f"Reading PSB file at path: '{file_path.absolute()}'")

    sdds_file = sdds.read(file_path)
    nbunches = sdds_file.values[N_BUNCHES]
    bunch_ids = sdds_file.values[BUNCH_ID if BUNCH_ID in sdds_file.values else HOR_BUNCH_ID]

    if len(bunch_ids) > nbunches:
        LOGGER.debug(
            f"Number of bunch IDs ({len(bunch_ids)}) exceeds number of bunches ({nbunches}). "
            f"Truncating bunch IDs to match number of bunches."
            f"This could happen when you kick fewer bunches than the total in the machine."
        )
        bunch_ids = bunch_ids[:nbunches]

    nturns = sdds_file.values[N_TURNS]
    # PSB uses milliseconds, so divide by 1e3 to get seconds for datetime conversion
    date = datetime.fromtimestamp(sdds_file.values[ACQ_STAMP] / 1e3, tz=tz.tzutc())
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
        "source_datatype": "psb",
        "date": date,
    }
    return TbtData(matrices, nturns=nturns, bunch_ids=bunch_ids, meta=meta)

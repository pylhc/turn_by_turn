"""
LHC
---

Data handling for turn-by-turn measurement files from the ``LHC`` (files in **SDDS** format).
"""
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import sdds
from dateutil import tz

from turn_by_turn.constants import (
    _ACQ_DATE_FORMAT,
    _ACQ_DATE_PREFIX,
    _ASCII_COMMENT,
    ACQ_STAMP,
    BPM_NAMES,
    BUNCH_ID,
    HOR_BUNCH_ID,
    N_BUNCHES,
    N_TURNS,
    NUM_TO_PLANE,
    PLANES,
)
from turn_by_turn.structures import TbtData, TransverseData

LOGGER = logging.getLogger()

POSITIONS: Dict[str, str] = {
    "X": "horPositionsConcentratedAndSorted",
    "Y": "verPositionsConcentratedAndSorted",
}


def read_tbt(file_path: Union[str, Path]) -> TbtData:
    """
    Reads turn-by-turn data from the ``LHC``'s **SDDS** format file. Will first determine if it is in ASCII
    format to figure out which reading method to use.

    Args:
        file_path (Union[str, Path]): path to the turn-by-turn measurement file.

    Returns:
        A ``TbTData`` object with the loaded data.
    """
    file_path = Path(file_path)
    LOGGER.debug(f"Reading LHC file at path: '{file_path.absolute()}'")

    if _is_ascii_file(file_path):
        matrices, date = _read_ascii(file_path)
        return TbtData(matrices, date, [0], matrices[0].X.shape[1])

    sdds_file = sdds.read(file_path)
    nbunches = sdds_file.values[N_BUNCHES]
    bunch_ids = sdds_file.values[BUNCH_ID if BUNCH_ID in sdds_file.values else HOR_BUNCH_ID]

    if len(bunch_ids) > nbunches:
        bunch_ids = bunch_ids[:nbunches]

    nturns = sdds_file.values[N_TURNS]
    date = datetime.utcfromtimestamp(sdds_file.values[ACQ_STAMP] / 1e9).replace(tzinfo=tz.tzutc())
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
    return TbtData(matrices, date, bunch_ids, nturns)


def _is_ascii_file(file_path: Union[str, Path]) -> bool:
    """
    Returns ``True`` only if the file looks like a readable LHC tbt ASCII file, else ``False``.

    Args:
        file_path (Union[str, Path]): path to the turn-by-turn measurement file.

    Returns:
        A boolean.
    """
    with Path(file_path).open("r") as file_data:
        try:
            for line in file_data:
                if line.strip() == "":
                    continue
                return line.startswith(_ASCII_COMMENT)
        except UnicodeDecodeError:
            return False
    return False


def _read_ascii(file_path: Union[str, Path]) -> Tuple[List[TransverseData], Optional[datetime]]:
    """
    Reads turn-by-turn data from an **LHC**'s ASCII SDDS format file, and return the date as well as
    parsed matrices for construction of a ``TbtData`` object.

    Args:
        file_path (Union[str, Path]): path to the turn-by-turn measurement file.

    Returns:
        Turn-by-turn data matrices and
    """
    data_lines = Path(file_path).read_text().splitlines()
    bpm_names = {"X": [], "Y": []}
    bpm_data = {"X": [], "Y": []}
    date = None  # will switch to TbtData.date's default if not found in file

    for line in data_lines:
        line = line.strip()

        if _ACQ_DATE_PREFIX in line:
            LOGGER.debug("Acquiring date from file")
            date = _parse_date(line)
            continue

        elif line == "" or line.startswith(_ASCII_COMMENT):  # empty or comment line
            continue

        else:  # data line, let's get samples
            plane_num, bpm_name, bpm_samples = _parse_samples(line)
            try:
                bpm_names[NUM_TO_PLANE[plane_num]].append(bpm_name)
                bpm_data[NUM_TO_PLANE[plane_num]].append(bpm_samples)
            except KeyError as error:
                raise ValueError(
                    f"Plane number '{plane_num}' found in file '{file_path}'.\n"
                    "Only '0' and '1' are allowed."
                ) from error

    matrices = [
        TransverseData(
            X=pd.DataFrame(index=bpm_names["X"], data=np.array(bpm_data["X"])),
            Y=pd.DataFrame(index=bpm_names["Y"], data=np.array(bpm_data["Y"])),
        )
    ]
    return matrices, date


# ----- Helpers ----- #


def _parse_samples(line: str) -> Tuple[str, str, np.ndarray]:
    """Parse a line from an LHC SDDS file into its different elements."""
    parts = line.split()
    plane_num = parts[0]
    bpm_name = parts[1]
    # bunch_id = part[2]  # not used, comment for clarification
    bpm_samples = np.array([float(part) for part in parts[3:]])
    return plane_num, bpm_name, bpm_samples


def _parse_date(line: str) -> datetime:
    """
    Parse a date timestamp line from an LHC SDDS file to a datetime object.
    If parsing of the default format fails, returns a filler datetime for today.
    """
    date_str = line.replace(_ACQ_DATE_PREFIX, "").replace(_ASCII_COMMENT, "").strip()
    try:
        return datetime.strptime(date_str, _ACQ_DATE_FORMAT)
    except ValueError:
        LOGGER.error("Could not parse date in file, defaulting to: Today, UTC")
        return datetime.today().replace(tzinfo=tz.tzutc())

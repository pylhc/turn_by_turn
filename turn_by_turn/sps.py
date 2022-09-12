"""
SPS
---

Data handling for turn-by-turn measurement files from the ``SPS`` (files in **SDDS** format).
"""
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import sdds
from dateutil import tz

from turn_by_turn.constants import NUM_TO_PLANE

from turn_by_turn.structures import TbtData, TransverseData

LOGGER = logging.getLogger()

def read_tbt(file_path: Union[str, Path]) -> TbtData:
    """
    Reads turn-by-turn data from the ``SPS``'s **SDDS** format file. Will first determine if it is in ASCII
    format to figure out which reading method to use.

    Args:
        file_path (Union[str, Path]): path to the turn-by-turn measurement file.

    Returns:
        A ``TbTData`` object with the loaded data.
    """
    file_path = Path(file_path)
    LOGGER.debug(f"Reading SPS file at path: '{file_path.absolute()}'")

    data_lines = Path(file_path).read_text().splitlines()
    if '#SDDSASCIIFORMAT' in data_lines[0]:
        matrices, date = _read_converted_ascii(file_path)
        return TbtData(matrices, date, [0], matrices[0].X.shape[1])

    sdds_file = sdds.read(file_path)

    nturns = sdds_file.values['nbOfTurns']
    date = datetime.utcfromtimestamp(sdds_file.values['timestamp'] / 1e9).replace(tzinfo=tz.tzutc())
    bpm_names = np.array(sdds_file.values['MonNames'])
    bpm_planes = np.array(sdds_file.values['MonPlanes'])

    ver_bpms = bpm_names[bpm_planes.astype(bool)]
    hor_bpms = bpm_names[np.logical_not(bpm_planes)]

    tbt_data_x = []
    tbt_data_y = []

    for bpm_name in hor_bpms:
        tbt_data_x.append(sdds_file.values[bpm_name])
    
    for bpm_name in ver_bpms:
        tbt_data_y.append(sdds_file.values[bpm_name])
    
    matrices = [TransverseData(
                    X=pd.DataFrame(index=hor_bpms, data=tbt_data_x, dtype=float),
                    Y=pd.DataFrame(index=ver_bpms, data=tbt_data_y, dtype=float),
                )]
    
    return TbtData(matrices, date, [0], nturns)


def _read_converted_ascii(file_path: Union[str, Path]) -> Tuple[List[TransverseData], Optional[datetime]]:
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

        if "Acquisition date:" in line:
            LOGGER.debug("Acquiring date from file")
            date = _parse_date(line)
            continue

        elif line == "" or line.startswith("#"):  # empty or comment line
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
    date_str = line.replace("Acquisition date:", "").replace("#", "").strip()
    try:
        return datetime.strptime(date_str, "%Y-%m-%d at %H:%M:%S")
    except ValueError:
        LOGGER.error("Could not parse date in file, defaulting to: Today, UTC")
        return datetime.today().replace(tzinfo=tz.tzutc())

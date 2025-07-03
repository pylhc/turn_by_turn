"""
ASCII
-----

Data handling for the special turn-by-turn ASCII files, that were used in the
past. They are not SDDS files, but instead more like table,
containing the columns:
- Plane (0 for horizontal, 1 for vertical)
- Observation point (i.e. BPM name)
- BPM index/longitunial location
- Value Turn 1, Turn 2, etc.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
import re
from typing import TextIO

import numpy as np
import pandas as pd
from dateutil import tz

from turn_by_turn.constants import FORMAT_STRING, PLANE_TO_NUM, PLANES, NUM_TO_PLANE
from turn_by_turn.structures import TbtData, TransverseData

LOGGER = logging.getLogger(__name__)

# ASCII IDs
ASCII_COMMENT: str = "#"
ASCII_ID = "SDDSASCIIFORMAT"
ACQ_DATE_PREFIX: str = "Acquisition date:"
ACQ_DATE_FORMAT: str = "%Y-%m-%d at %H:%M:%S"


def is_ascii_file(file_path: str | Path) -> bool:
    """
    Returns ``True`` only if the file looks like a readable TbT ASCII file, else ``False``.

    Args:
        file_path (str | Path): path to the turn-by-turn measurement file.

    Returns:
        A boolean.
    """
    with Path(file_path).open("r") as file_data:
        try:
            for line in file_data:
                # skip empty lines
                if line.strip() == "":
                    continue
                # return line.strip().startswith(f"#{ASCII_ID}")  # see _write_header
                return line.strip().startswith(ASCII_COMMENT)  #  e.g. PS does not follow the rules
        except UnicodeDecodeError:
            return False
    return False


# ----- Writer ----- #

def write_tbt(output_path: str | Path, tbt_data: TbtData) -> None:
    """
    Write a ``TbtData`` object's data to file, in the TbT ASCII format.

    Args:
        output_path (str | Path): path to the disk location where to write the data.
        tbt_data (TbtData): the ``TbtData`` object to write to disk.
    """
    output_path = Path(output_path)
    LOGGER.info(f"Writing TbTdata in ASCII format at '{output_path.absolute()}'")

    for bunch_id in range(tbt_data.nbunches):
        LOGGER.debug(f"Writing data for bunch {bunch_id}")
        suffix = f"_{tbt_data.bunch_ids[bunch_id]}" if tbt_data.nbunches > 1 else ""
        with output_path.with_suffix(suffix).open("w") as output_file:
            _write_header(tbt_data, bunch_id, output_file)
            _write_tbt_data(tbt_data, bunch_id, output_file)


def _write_header(tbt_data: TbtData, bunch_id: int, output_file: TextIO) -> None:
    """
    Write the appropriate headers for a ``TbtData`` object's given bunch_id in the ASCII format.
    """
    output_file.write(f"#{ASCII_ID} v1\n")
    output_file.write(f"#Created: {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')} By: Python turn_by_turn Package\n")
    output_file.write(f"#Number of turns: {tbt_data.nturns}\n")
    output_file.write(f"#Number of horizontal monitors: {tbt_data.matrices[bunch_id].X.index.size}\n")
    output_file.write(f"#Number of vertical monitors: {tbt_data.matrices[bunch_id].Y.index.size}\n")
    output_file.write(f"#Acquisition date: {tbt_data.date.strftime('%Y-%m-%d at %H:%M:%S')}\n")


def _write_tbt_data(tbt_data: TbtData, bunch_id: int, output_file: TextIO) -> None:
    """Write a ``TbtData`` object's data for the given bunch_id to disk in the ASCII format."""
    row_format = "{} {} {}  " + FORMAT_STRING * tbt_data.nturns + "\n"
    for plane in PLANES:
        for bpm_index, bpm_name in enumerate(tbt_data.matrices[bunch_id][plane].index):
            samples = tbt_data.matrices[bunch_id][plane].loc[bpm_name, :].to_numpy()
            output_file.write(row_format.format(PLANE_TO_NUM[plane], bpm_name, bpm_index, *samples))


# ----- Reader ----- #

def read_tbt(file_path: str | Path, bunch_id: int = None) -> TbtData:
    """
    Reads turn-by-turn data from an ASCII turn-by-turn format file, and return the date as well as
    parsed matrices for construction of a ``TbtData`` object.

    Args:
        file_path (str | Path): path to the turn-by-turn measurement file.
        bunch_id (int, optional): the bunch id associated with this file. 
                                  Defaults to `None`, but is then attempted to parsed
                                  from the filename. If not found, `0` is used.

    Returns:
        Turn-by-turn data
    """
    data_lines = Path(file_path).read_text().splitlines()
    bpm_names = {"X": [], "Y": []}
    bpm_data = {"X": [], "Y": []}
    date = None  # will switch to TbtData.date's default if not found in file
    
    if bunch_id is None:
        bunch_id = _parse_bunch_id(file_path)

    for line in data_lines:
        line = line.strip()

        if ACQ_DATE_PREFIX in line:
            LOGGER.debug("Acquiring date from file")
            date = _parse_date(line)
            continue

        elif line == "" or line.startswith(ASCII_COMMENT):  # empty or comment line
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
    return TbtData(matrices=matrices, date=date, bunch_ids=[bunch_id], nturns=matrices[0].X.shape[1])


# ----- Helpers ----- #


def _parse_samples(line: str) -> tuple[str, str, np.ndarray]:
    """Parse a line into its different elements."""
    parts = line.split()
    plane_num = parts[0]
    bpm_name = parts[1]
    # bpm_location = part[2]  # not used, comment for clarification
    bpm_samples = np.array([float(part) for part in parts[3:]])
    return plane_num, bpm_name, bpm_samples


def _parse_date(line: str) -> datetime:
    """
    Parse a date timestamp line to a datetime object.
    If parsing of the default format fails, returns a filler datetime for today.
    """
    date_str = line.replace(ACQ_DATE_PREFIX, "").replace(ASCII_COMMENT, "").strip()
    try:
        return datetime.strptime(date_str, ACQ_DATE_FORMAT)
    except ValueError:
        LOGGER.error("Could not parse date in file, defaulting to: Today, UTC")
        return datetime.today().replace(tzinfo=tz.tzutc())


def _parse_bunch_id(file_path: Path) -> int:
    """Parse the bunch_id from the filename."""
    bunch_id_match = re.match(r".*_(?P<bunch_id>\d+)(.sdds)?$", file_path.name)
    if bunch_id_match:
        try:
            return int(bunch_id_match.group("bunch_id"))
        except ValueError:
            pass
    return 0


# For backwards compatibility <0.4.2:
write_ascii = write_tbt
read_ascii = read_tbt

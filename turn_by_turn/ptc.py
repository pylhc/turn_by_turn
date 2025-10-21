"""
PTC
---

Data handling for turn-by-turn measurement files from the ``PTC`` code, which can be obtained by performing
particle tracking of your machine through the ``MAD-X PTC`` interface. The files are very close in
structure to **TFS** files, with the difference that the data part is split into "segments" relating
containing data for a given observation point.
"""

from __future__ import annotations

import copy
import logging
from collections import namedtuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from dateutil import tz

from turn_by_turn.constants import PLANES, MetaDict
from turn_by_turn.errors import PTCFormatError
from turn_by_turn.structures import TbtData, TransverseData

if TYPE_CHECKING:
    from collections.abc import Sequence

LOGGER = logging.getLogger(__name__)
Segment = namedtuple("Segment", ["number", "turns", "particles", "element", "name"])

# IDs ---

HEADER: str = "@"
NAMES: str = "*"
TYPES: str = "$"
SEGMENTS: str = "#segment"
SEGMENT_MARKER: tuple[str, str] = ("start", "end")
COLX: str = "X"
COLY: str = "Y"
COLTURN: str = "TURN"
COLPARTICLE: str = "NUMBER"
DATE: str = "DATE"
TIME: str = "TIME"
TIME_FORMAT: str = "%d/%m/%y %H.%M.%S"


def read_tbt(file_path: str | Path) -> TbtData:
    """
    Reads turn-by-turn data from the ``PTC`` **trackone** format file.

    Args:
        file_path (Union[str, Path]): path to the turn-by-turn measurement file.

    Returns:
        A ``TbTData`` object with the loaded data.
    """
    file_path = Path(file_path)
    LOGGER.debug(f"Reading PTC trackone file at path: '{file_path.absolute()}'")
    lines: list[str] = file_path.read_text().splitlines()

    LOGGER.debug("Reading header from file")
    date, header_length = _read_header(lines)
    lines = lines[header_length:]

    # parameters
    params = _read_from_first_turn(lines)

    # read into dict first for speed then convert to DFs
    matrices = [
        {p: {bpm: np.zeros(params.n_turns) for bpm in params.bpms} for p in PLANES} for _ in range(params.n_particles)
    ]
    matrices = _read_data(lines, matrices, params.column_indices)
    for bunch in range(params.n_particles):
        matrices[bunch] = TransverseData(
            X=pd.DataFrame(matrices[bunch]["X"]).transpose(),
            Y=pd.DataFrame(matrices[bunch]["Y"]).transpose(),
        )

    LOGGER.debug(f"Read Tbt matrices from: '{file_path.absolute()}'")

    meta: MetaDict = {
        "date": date,
        "file": file_path,
        "source_datatype": "ptc",
    }
    return TbtData(matrices=matrices, nturns=params.n_turns, bunch_ids=params.particles, meta=meta)


def _read_header(lines: Sequence[str]) -> tuple[datetime, int]:
    """Reads header length and datetime from header."""
    idx_line = 0
    date_str = dict.fromkeys([DATE, TIME])
    for idx_line, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) == 0:
            continue

        if parts[0] != HEADER:
            break

        if parts[1] in date_str:
            date_str[parts[1]] = parts[-1].strip("'\" ")

    if any(datestring is None for datestring in date_str.values()):
        LOGGER.warning("No date found in file, defaulting to today")
        return datetime.today().replace(tzinfo=tz.tzutc()), idx_line

    return datetime.strptime(f"{date_str[DATE]} {date_str[TIME]}", TIME_FORMAT), idx_line


@dataclass(slots=True)
class TbTParams:
    """ Parameters read from the first turn of the file. """
    bpms: list[str] = field(default_factory=list)
    particles: list[int] = field(default_factory=list)
    column_indices: dict[Any, Any] | None = None
    n_turns: int = 0
    n_particles: int = 0


def _read_from_first_turn(lines: Sequence[str]) -> TbTParams:
    """
    Reads the BPMs, particles, column indices and number of turns and particles from the matrices of
    the first turn.
    """
    LOGGER.debug("Reading first turn to define boundary parameters.")
    data = TbTParams()
    first_segment = True

    for line in lines:
        parts = line.strip().split()
        if len(parts) == 0 or parts[0] in [HEADER, TYPES]:
            continue

        if parts[0] == NAMES:  # read column names
            if data.column_indices is not None:
                raise KeyError(f"{NAMES} are defined twice in tbt file!")
            data.column_indices = _parse_column_names_to_indices(parts[1:])
            continue

        if parts[0] == SEGMENTS:  # read segments, append to bunch_id
            segment = Segment(*parts[1:])
            if segment.name == SEGMENT_MARKER[0]:  # start of first segment
                data.n_turns = int(segment.turns) - 1
                data.n_particles = int(segment.particles)

            elif segment.name == SEGMENT_MARKER[1]:  # end of first segment
                break

            else:
                first_segment = False
                data.bpms.append(segment.name)

        elif first_segment:
            if data.column_indices is None:
                LOGGER.error("Columns not defined in Tbt file")
                raise PTCFormatError

            new_data = _parse_data(data.column_indices, parts)
            particle = int(float(new_data[COLPARTICLE]))
            data.particles.append(particle)

    if len(data.particles) == 0:
        msg = "No particles found in TbT file"
        LOGGER.error(msg)
        raise PTCFormatError(msg)
    return data


def _read_data(
    lines: Sequence[str], matrices: dict[str, dict[str, np.ndarray]], column_indices: dict
) -> dict[str, dict[str, np.ndarray]]:
    """Read the matrices into the matrices."""
    LOGGER.debug("Reading matrices.")
    matrices = copy.deepcopy(matrices)
    segment = None
    column_map = {"X": COLX, "Y": COLY}

    for line in lines:
        parts = line.strip().split()
        if len(parts) == 0 or parts[0] in (HEADER, TYPES, NAMES):
            continue

        if parts[0] == SEGMENTS:  # start of a new segment
            segment = Segment(*parts[1:])
            continue

        if segment is None:
            LOGGER.error("Data written before Segment definition")
            raise PTCFormatError

        if segment.name in SEGMENT_MARKER:
            continue

        data = _parse_data(column_indices, parts)
        part_id = int(float(data[COLPARTICLE])) - 1
        turn_nr = int(float(data[COLTURN])) - 1
        for plane in PLANES:
            matrices[part_id][plane][segment.name][turn_nr] = float(data[column_map[plane]])
    return matrices


def _parse_data(column_indices, parts: Sequence[str]) -> dict:
    """
    Converts the ``parts`` (split elements of a data line) into a dictionary based on the indices in
    ``column_indices``.
    """
    return {col: parts[col_idx] for col, col_idx in column_indices.items()}


def _parse_column_names_to_indices(parts: Sequence[str]) -> dict:
    """Parses the column names from the line into a dictionary with indices."""
    col_idx = dict.fromkeys([COLX, COLY, COLTURN, COLPARTICLE])
    LOGGER.debug("Setting column names.")

    for idx, column_name in enumerate(parts):
        if column_name not in col_idx:
            LOGGER.debug(f"Column '{column_name}' will be ignored.")
            continue
        if col_idx[column_name] is not None:
            raise KeyError(f"'{column_name}' is defined twice.")
        col_idx[column_name] = idx
    missing = [c for c in col_idx.values() if c is None]

    if any(missing):
        raise ValueError(f"The following columns are missing in ptc file: '{str(missing)}'")
    return col_idx

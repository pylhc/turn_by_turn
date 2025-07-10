"""
SPS
---

Data handling for turn-by-turn measurement files from the ``SPS`` (files in **SDDS** format).
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
import re
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import sdds
from dateutil import tz

from turn_by_turn.ascii import is_ascii_file, read_tbt as read_ascii
from turn_by_turn.structures import TbtData, TransverseData

if TYPE_CHECKING:
    from collections.abc import Sequence

LOGGER = logging.getLogger(__name__)


# IDs
N_TURNS: str = "nbOfTurns"
TIMESTAMP: str = "timestamp"
BPM_NAMES: str = "MonNames"
BPM_PLANES: str = "MonPlanes"


def read_tbt(file_path: str | Path, remove_trailing_bpm_plane: bool = True) -> TbtData:
    """
    Reads turn-by-turn data from the ``SPS``'s **SDDS** format file.
    Will first determine if it is in ASCII format to figure out which reading method to use.

    Args:
        file_path (str | Path): path to the turn-by-turn measurement file.
        remove_trailing_bpm_plane (bool, optional): if ``True``, will remove the trailing
            BPM plane ('.H', '.V') from the BPM-names.
            This makes the measurement data compatible with the madx-models.
            Defaults to ``True``.

    Returns:
        A ``TbTData`` object with the loaded data.
    """
    file_path = Path(file_path)
    LOGGER.debug(f"Reading SPS file at path: '{file_path.absolute()}'")

    if is_ascii_file(file_path):
        return read_ascii(file_path)

    sdds_file = sdds.read(file_path)

    nturns = sdds_file.values[N_TURNS]
    date = datetime.fromtimestamp(sdds_file.values[TIMESTAMP] / 1e9, tz=tz.tzutc())

    bpm_names_x, bpm_names_y = _split_bpm_names_to_planes(
        sdds_file.values[BPM_NAMES],
        sdds_file.values[BPM_PLANES]
    )

    tbt_data_x = [sdds_file.values[bpm] for bpm in bpm_names_x]
    tbt_data_y = [sdds_file.values[bpm] for bpm in bpm_names_y]

    if remove_trailing_bpm_plane:
        pattern = re.compile(r"\.[HV]$", flags=re.IGNORECASE)
        bpm_names_x  = [pattern.sub("", bpm) for bpm in bpm_names_x]
        bpm_names_y  = [pattern.sub("", bpm) for bpm in bpm_names_y]

    matrices = [
        TransverseData(
            X=pd.DataFrame(index=bpm_names_x, data=tbt_data_x, dtype=float),
            Y=pd.DataFrame(index=bpm_names_y, data=tbt_data_y, dtype=float),
        )
    ]

    return TbtData(matrices, date, [0], nturns)


def _split_bpm_names_to_planes(bpm_names: Sequence[str], bpm_planes: Sequence[int] = ()) -> tuple[np.ndarray, np.ndarray]:
    """ Splits BPM names into X and Y planes.

    In the past, this was done by using the ``MonPlanes`` array, but in 2025 the SPS output changed from using
    `1` for vertical and `0` for horizontal to `3` for vertical and `1` for horizontal.
    It is therefore probably better to split based on the naming of the BPMs.

    Args:
        bpm_names (Sequence[str]): BPM names
        bpm_planes (Sequence[int], optional): BPM planes

    Returns:
        tuple[np.ndarray, np.ndarray]: BPM names for X and Y planes
    """
    bpm_names = pd.Series(bpm_names)

    if bpm_names.str.match(r".+\.[HV]$", flags=re.IGNORECASE).all():
        # all names end in .V or .H -> split by name
        vertical_bpms = bpm_names.str.endswith(".V")
    else:
        LOGGER.warning(
            "Could not determine BPM planes from BPM names. "
            "Splitting by the 'MonPlanes' array, which might be subject to changes."
        )
        if 3 in bpm_planes: # 2025 format splitting: 3 for vertical, 1 for horizontal
            vertical_bpms = np.array(bpm_planes) == 3

        elif 0 in bpm_planes: # pre-2025 format splitting: 1 for vertical, 0 for horizontal
            vertical_bpms = np.array(bpm_planes).astype(bool)

        else:
            msg = "Could not determine SPS file format to split BPMs into planes."
            raise ValueError(msg)

    bpm_names_y = bpm_names[vertical_bpms].to_numpy()
    bpm_names_x = bpm_names[~vertical_bpms].to_numpy()

    return bpm_names_x, bpm_names_y


def write_tbt(output_path: str | Path, tbt_data: TbtData, add_trailing_bpm_plane: bool = True) -> None:
    """
    Write a ``TbtData`` object's data to file, in a ``SPS``'s **SDDS** format.
    The format is reduced to the minimum parameters used by the reader.

    WARNING: This writer uses ``0`` for horizontal and ``1`` for vertical BPMs
             in the  ``MonPlanes`` array, i.e. the pre-2025 format.

    Args:
        output_path (str | Path): path to a the disk location where to write the data.
        tbt_data (TbtData): the ``TbtData`` object to write to disk.
        add_trailing_bpm_plane (bool, optional): if ``True``, will add the trailing
            BPM plane ('.H', '.V') to the BPM-names. This assures that all BPM-names are unique,
            and that the measurement data is compatible with the sdds files from the FESA-class.
            WARNING: If present, these will be used to determine the plane of the BPMs,
            otherwise the ``MonPlanes`` array will be used.
            Defaults to ``True``.
    """
    output_path = Path(output_path)
    LOGGER.info(f"Writing TbTdata in binary SDDS (SPS) format at '{output_path.absolute()}'")

    df_x, df_y = tbt_data.matrices[0].X, tbt_data.matrices[0].Y

    # bpm names
    bpm_names_x, bpm_names_y = df_x.index.to_list(), df_y.index.to_list()

    if add_trailing_bpm_plane:
        bpm_names_x = [f"{bpm_name}.H" if not bpm_name.endswith(".H") else bpm_name for bpm_name in bpm_names_x]
        bpm_names_y = [f"{bpm_name}.V" if not bpm_name.endswith(".V") else bpm_name for bpm_name in bpm_names_y]

    bpm_names = bpm_names_x + bpm_names_y

    # bpm planes
    bpm_planes = np.zeros(shape=[len(bpm_names)])
    bpm_planes[-len(bpm_names_y):] = 1

    list_of_data = [a for df in (df_x, df_y) for a in df.to_numpy()]

    definitions = [
                      sdds.classes.Parameter(TIMESTAMP, "llong"),
                      sdds.classes.Parameter(N_TURNS, "long"),
                      sdds.classes.Array(BPM_NAMES, "string"),
                      sdds.classes.Array(BPM_PLANES, "long"),
                  ] + [sdds.classes.Array(bpm, "double") for bpm in bpm_names]

    values = [
                 tbt_data.date.timestamp() * 1e9,
                 tbt_data.nturns,
                 bpm_names,
                 bpm_planes,
                 ] + list_of_data
    sdds.write(sdds.SddsFile("SDDS1", None, definitions, values), output_path)

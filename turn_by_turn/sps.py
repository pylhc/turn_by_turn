"""
SPS
---

Data handling for turn-by-turn measurement files from the ``SPS`` (files in **SDDS** format).
"""
import logging
from datetime import datetime
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import sdds
from dateutil import tz

from turn_by_turn.ascii import is_ascii_file, read_ascii
from turn_by_turn.constants import PLANE_TO_NUM
from turn_by_turn.structures import TbtData, TransverseData
from turn_by_turn.utils import matrices_to_array, add_noise

LOGGER = logging.getLogger()

# IDs
N_TURNS: str = "nbOfTurns"
TIMESTAMP: str = "timestamp"
BPM_NAMES: str = "MonNames"
BPM_PLANES: str = "MonPlanes"


def read_tbt(file_path: Union[str, Path]) -> TbtData:
    """
    Reads turn-by-turn data from the ``SPS``'s **SDDS** format file.
    Will first determine if it is in ASCII format to figure out which reading method to use.

    Args:
        file_path (Union[str, Path]): path to the turn-by-turn measurement file.

    Returns:
        A ``TbTData`` object with the loaded data.
    """
    file_path = Path(file_path)
    LOGGER.debug(f"Reading SPS file at path: '{file_path.absolute()}'")

    if is_ascii_file(file_path):
        matrices, date = read_ascii(file_path)
        return TbtData(matrices, date, [0], matrices[0].X.shape[1])

    sdds_file = sdds.read(file_path)

    nturns = sdds_file.values[N_TURNS]
    date = datetime.utcfromtimestamp(sdds_file.values[TIMESTAMP] / 1e9).replace(
        tzinfo=tz.tzutc()
    )
    bpm_names = np.array(sdds_file.values[BPM_NAMES])
    bpm_planes = np.array(sdds_file.values[BPM_PLANES]).astype(bool)

    ver_bpms = bpm_names[bpm_planes]
    hor_bpms = bpm_names[~bpm_planes]

    tbt_data_x = [sdds_file.values[bpm] for bpm in hor_bpms]
    tbt_data_y = [sdds_file.values[bpm] for bpm in ver_bpms]

    matrices = [
        TransverseData(
            X=pd.DataFrame(index=hor_bpms, data=tbt_data_x, dtype=float),
            Y=pd.DataFrame(index=ver_bpms, data=tbt_data_y, dtype=float),
        )
    ]

    return TbtData(matrices, date, [0], nturns)


def write_tbt(output_path: Union[str, Path], tbt_data: TbtData) -> None:
    """
    Write a ``TbtData`` object's data to file, in a ``SPS``'s **SDDS** format.
    The format is reduced to the necessary parameters used by the reader.

    Args:
        output_path (Union[str, Path]): path to a the disk location where to write the data.
        tbt_data (TbtData): the ``TbtData`` object to write to disk.
    """
    output_path = Path(output_path)
    LOGGER.info(f"Writing TbTdata in binary SDDS (SPS) format at '{output_path.absolute()}'")

    df_x, df_y = tbt_data.matrices[0].X, tbt_data.matrices[0].Y
    bpm_names_x, bpm_names_y = df_x.index.to_list(), df_y.index.to_list()
    bpm_names = bpm_names_x + bpm_names_y
    bpm_planes = np.zeros(shape=[len(bpm_names_x)]).tolist() + np.ones(shape=[len(bpm_names_y)]).tolist()
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

"""
LHC
---

Data handling for turn-by-turn measurement files from the ``LHC`` (files in **SDDS** format).
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
import re

import numpy as np
import pandas as pd

from turn_by_turn.structures import TbtData, TransverseData

LOGGER = logging.getLogger(__name__)


def read_tbt(file_path: str | Path) -> TbtData:
    """
    Reads turn-by-turn data from the ``SuperKEKB``'s measurement file.

    Args:
        file_path (Union[str, Path]): path to the turn-by-turn measurement file.

    Returns:
        A ``TbTData`` object with the loaded data.
    """
    file_path = Path(file_path)
    LOGGER.debug(f"Reading SuperKEKB file at path: '{file_path.absolute()}'")

    with open(file_path, "r") as f:
        content = f.read().replace("\n", "").replace(" ", "").replace("\\", "")

    # Get the date, stored in the header, a simple regex is enough
    date = re.findall(r"(\d{4}-[01]\d-[0-3]\d_[0-2]\d:[0-5]\d:[0-5]\d\.\d+)", content)[0]
    date = datetime.strptime(date, "%Y-%m-%d_%H:%M:%S.%f")

    # Craft a regex to extract BPM data
    # The data is organized as follows:
    # ("BPM_NAME"->{X_VALUES}, {Y_VALUES}, {STD_ERR?})
    x_vals = []
    y_vals = []
    monitors = []
    for match in re.finditer(
        r'\("([A-Z0-9]+)"->{{([^}]+)},{([^}]+)},{([^}]+)}}\)', content
    ):
        monitors.append(match.group(1))
        x_vals.append(np.array([float(v) for v in match.group(2).split(",")]))
        y_vals.append(np.array([float(v) for v in match.group(3).split(",")]))
    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)

    # Create the TbtData object
    tbt_data = TbtData(
        nturns=len(x_vals[0]),
        matrices=[
            TransverseData(
                X=pd.DataFrame(
                    index=monitors,
                    data=x_vals,
                ),
                Y=pd.DataFrame(
                    index=monitors,
                    data=y_vals,
                ),
            )
        ],
        meta={
            "file": file_path.name,
            "source_datatype": "superkekb",
            "date": date,
        },
    )
    return tbt_data


def write_tbt(file_path: str | Path, tbt_data: TbtData) -> None:
    """
    Writes turn-by-turn data to a ``SuperKEKB``'s measurement file.

    Args:
        file_path (Union[str, Path]): path to the output turn-by-turn measurement file.
        tbt_data (TbtData): turn-by-turn data to write.
    """
    raise NotImplementedError("Writing SuperKEKB files is not implemented yet.")
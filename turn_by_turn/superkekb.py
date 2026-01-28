"""
SuperKEKB
---

Data handling for turn-by-turn measurement files from ``SuperKEKB`` taken by the application in
the control room. The file format is similar to Mathematica or some json and be parsed easily
with regex. The extension is usually `.data`.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path

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
    content = file_path.read_text().replace("\n", "").replace("\\", "").replace(" ", "")

    # Get the date, stored in the header, a simple regex is enough
    # Same for the values, with the follow pattern:
    # ("BPM_NAME"->{X_VALUES}, {Y_VALUES}, {STD_ERR?})
    DATE_PATTERN = re.compile(
        r"([01]\d\/[0-3]\d\/\d{4}[0-2]\d:[0-5]\d:[0-5]\d)"
    )  # date in header
    VALUES_PATTERN = re.compile(
        r'\("(?P<monitors>[A-Z0-9]+)"->{{(?P<x>[^}]+)},{(?P<y>[^}]+)},{(?P<std>[^}]+)}}\)'
    )

    try:
        date = datetime.strptime(
            re.findall(DATE_PATTERN, content)[0], "%m/%d/%Y%H:%M:%S"
        )
    except:
        date = None

    # Craft a regex to extract BPM data
    # The data is organized as follows:
    x_vals = []
    y_vals = []
    monitors = []
    for match in re.finditer(VALUES_PATTERN, content):
        monitors.append(match.group("monitors"))
        x_vals.append(np.array([float(v) for v in match.group("x").split(",")]))
        y_vals.append(np.array([float(v) for v in match.group("y").split(",")]))
    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)

    # Create the TbtData object
    meta_dict = {
        "file": file_path.name,
        "source_datatype": "superkekb",
    }
    if date is not None:
        meta_dict["date"] = date

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
        meta=meta_dict,
    )

    LOGGER.info(
        f"Loaded SuperKEKB TbT file with {tbt_data.nturns} turns and {len(monitors)} monitors."
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

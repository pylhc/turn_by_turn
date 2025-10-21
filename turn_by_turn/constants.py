"""
Constants
---------

Specific constants to be used in ``turn_by_turn``, to help with consistency.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path

PLANES: tuple[str, str] = ("X", "Y")
NUM_TO_PLANE: dict[str, str] = {"0": "X", "1": "Y"}
PLANE_TO_NUM: dict[str, int] = {"X": 0, "Y": 1}

# ----- Common Meta Keys ----- #

class MetaDict(TypedDict, total=False):
    """ Metadata dictionary, to type-hint known entries.
    None of the entries are required (``total=False``).

    Attributes:
        date (datetime): Date of the measurement/creation of the data
        file (Path | str): Path to the file the data was loaded from (if available).
        machine (str): Name of the machine the data was measured/simulated on.
        source_datatype (str): The datatype this data was loaded from.
        comment (str): Any comment on the measurement.
    """
    date: datetime
    file: Path | str
    machine: str
    source_datatype: str
    comment: str

# ----- Miscellaneous ----- #

PRINT_PRECISION: int = 6
FORMAT_STRING: str = f" {{:.{PRINT_PRECISION:d}f}}"

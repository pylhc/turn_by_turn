"""
Constants
---------

Specific constants to be used in ``turn_by_turn``, to help with consistency.
"""
from typing import Dict, Tuple

PLANES: Tuple[str, str] = ("X", "Y")
NUM_TO_PLANE: Dict[str, str] = {"0": "X", "1": "Y"}
PLANE_TO_NUM: Dict[str, int] = {"X": 0, "Y": 1}


# ----- Miscellaneous ----- #

PRINT_PRECISION: int = 6
FORMAT_STRING: str = f" {{:.{PRINT_PRECISION:d}f}}"

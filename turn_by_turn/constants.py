"""
Constants
---------

Specific constants to be used in ``turn_by_turn``, to help with consistency.
"""
from typing import Dict, Tuple

PLANES: Tuple[str, str] = ("X", "Y")
NUM_TO_PLANE: Dict[str, str] = {"0": "X", "1": "Y"}
PLANE_TO_NUM: Dict[str, int] = {"X": 0, "Y": 1}

# ----- LHC Format Specifics -----

# ASCII IDs
_ASCII_COMMENT: str = "#"
_ACQ_DATE_PREFIX: str = "Acquisition date:"
_ACQ_DATE_FORMAT: str = "%Y-%m-%d at %H:%M:%S"

# BINARY IDs
N_BUNCHES: str = "nbOfCapBunches"
BUNCH_ID: str = "BunchId"
HOR_BUNCH_ID: str = "horBunchId"
N_TURNS: str = "nbOfCapTurns"
ACQ_STAMP: str = "acqStamp"
BPM_NAMES: str = "bpmNames"

# ----- PTC Format Specifics ----- #

HEADER: str = "@"
NAMES: str = "*"
TYPES: str = "$"
SEGMENTS: str = "#segment"
SEGMENT_MARKER: Tuple[str, str] = ("start", "end")
COLX: str = "X"
COLY: str = "Y"
COLTURN: str = "TURN"
COLPARTICLE: str = "NUMBER"
DATE: str = "DATE"
TIME: str = "TIME"
TIME_FORMAT: str = "%d/%m/%y %H.%M.%S"

# ----- Miscellaneous ----- #

PRINT_PRECISION: int = 6
FORMAT_STRING: str = f" {{:.{PRINT_PRECISION:d}f}}"

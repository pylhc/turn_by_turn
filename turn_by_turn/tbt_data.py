"""
Generic TbT Data
---

Data handling for turn-by-turn measurement data.
Can be used with in-memory tracking data so not to have to write .sdds files.

"""
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Union

import numpy as np
import pandas as pd
import sdds
from dateutil import tz

from turn_by_turn.ascii import is_ascii_file, read_tbt as read_ascii
from turn_by_turn.constants import PLANES, PLANE_TO_NUM
from turn_by_turn.structures import TbtData, TransverseData
from turn_by_turn.utils import matrices_to_array

LOGGER = logging.getLogger()

def read_tbt(data: TbtData) -> TbtData:
    """
    Reads turn-by-turn data as a TbtData format.

    Args:
        data (Union[TbtData]): measurement data.

    Returns:
        The same ``TbTData`` object that was loaded.
    """

    return data


def write_tbt(*args, **kwargs) -> None:
    """
    Not implemented, as it would not make sense to write an object. Use other functions to write to
    the desired format.
    """
    raise NotImplementedError("This function is not implemented.")
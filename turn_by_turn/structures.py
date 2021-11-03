"""
Structures
----------

Data structures to be used in ``turn_by_turn`` to store turn-by-turn measurement data.
"""
from dataclasses import dataclass, field, fields
from datetime import datetime
from typing import Dict, List, Sequence

import pandas as pd
from dateutil import tz


@dataclass
class TransverseData:
    """
    Object holding measured turn-by-turn data for both transverse planes in the form of pandas DataFrames.
    """

    X: pd.DataFrame  # horizontal data
    Y: pd.DataFrame  # vertical data

    def fieldnames(self):
        return (f.name for f in fields(self))

    def __getitem__(self, item):  # to access X and Y like one would with a dictionary
        if item not in self.fieldnames():
            raise KeyError(f"'{item}' is not in the fields of a {self.__class__.__name__} object.")
        return getattr(self, item)


@dataclass
class TbtData:
    """
    Object holding a representation of a Turn-by-Turn data measurement. The date of the measurement,
    the transverse data, number of turns and bunches as well as the bunch IDs are encapsulated in this object.
    """

    matrices: Sequence[TransverseData]  # each entry corresponds to a bunch
    date: datetime = None  # will default in post_init
    bunch_ids: List[int] = None  # will default in post_init
    nturns: int = 0
    nbunches: int = field(init=False)

    def __post_init__(self):
        self.nbunches = len(self.bunch_ids)
        if self.date is None:
            self.date = datetime.today().replace(tzinfo=tz.tzutc())  # to today, UTC if nothing is given
        if self.bunch_ids is None:
            self.bunch_ids = []

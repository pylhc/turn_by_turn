"""
Structures
----------

Data structures to be used in ``turn_by_turn`` to store turn-by-turn measurement data.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Sequence

import pandas as pd
from dateutil import tz


@dataclass
class TbtData:
    """
    Object holding a representation of a Turn-by-Turn Data.
    """

    matrices: Sequence[Dict[str, pd.DataFrame]]
    date: datetime = datetime.today().replace(tzinfo=tz.tzutc())  # defaults to today, UTC
    bunch_ids: List[int] = None
    nturns: int = 0
    nbunches: List[int] = field(init=False)

    def __post_init__(self):
        self.nbunches = len(self.bunch_ids)

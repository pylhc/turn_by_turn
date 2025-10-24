"""
Structures
----------

Data structures to be used in ``turn_by_turn`` to store turn-by-turn measurement data.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd

    from turn_by_turn.constants import MetaDict


@dataclass(slots=True)
class TransverseData:
    """
    Object holding measured turn-by-turn data for both transverse planes in the form of pandas DataFrames.

    The DataFrames should be N_(observation-points) x M_(turns) matrices, and usually contain
    the BPM/observation-point names as index, while the columns are simply numbered starting from ``0``.
    All DataFrames should have the same N x M size.

    Attributes:
        X (pd.DataFrame): Horizontal position data
        Y (pd.DataFrame): Vertical position data

    """

    X: pd.DataFrame
    Y: pd.DataFrame

    @classmethod
    def fieldnames(cls) -> list[str]:
        """Return a list of the fields of this dataclass."""
        return [f.name for f in fields(cls)]

    def __getitem__(self, item):  # to access X and Y like one would with a dictionary
        if item not in self.fieldnames():
            raise KeyError(f"'{item}' is not in the fields of a {self.__class__.__name__} object.")
        return getattr(self, item)


@dataclass(slots=True)
class TrackingData:
    """
    Object holding multidimensional turn-by-turn simulation data in the form of pandas DataFrames.

    The DataFrames should be N_observation-points x M_turns matrices, and usually contain
    the BPM/observation-point names as index, while the columns are simply numbered starting from ``0``.
    All DataFrames should have the same N x M size.

    Attributes:
        X (pd.DataFrame): Horizontal position data
        PX (pd.DataFrame): Horizontal momentum data
        Y (pd.DataFrame): Vertical position data
        PY (pd.DataFrame): Vertical momentum data
        T (pd.DataFrame): Negative time difference wrt. the reference particle (multiplied by c)
        PT (pd.DataFrame): Energy difference wrt. the reference particle divided by the ref. momentum (multiplied by c)
        S (pd.DataFrame): Longitudinal position data
        E (pd.DataFrame): Energy data

    """

    X: pd.DataFrame
    PX: pd.DataFrame
    Y: pd.DataFrame
    PY: pd.DataFrame
    T: pd.DataFrame
    PT: pd.DataFrame
    S: pd.DataFrame
    E: pd.DataFrame

    @classmethod
    def fieldnames(cls) -> list[str]:
        """Return a list of the fields of this dataclass."""
        return [f.name for f in fields(cls)]

    def __getitem__(self, item):  # to access fields like one would with a dictionary
        if item not in self.fieldnames():
            raise KeyError(f"'{item}' is not in the fields of a {self.__class__.__name__} object.")
        return getattr(self, item)


DataType = TransverseData | TrackingData


@dataclass(slots=True)
class TbtData:
    """
    Object holding a representation of a Turn-by-Turn data measurement. The date of the measurement,
    the transverse data, number of turns and bunches as well as the bunch IDs are encapsulated in this object.

    Attributes:
        matrices (Sequence[DataType]): Sequence of ``TransverseData`` or ``TrackingData`` objects.
                                       Each entry corresponds to a "bunch" or "particle" (tracking).
        nturns (int): Number of turns. Needs to be a positive integer.
                      It is assumed all bunches (and observation points in all entries therein)
                      have the same length in the turn-dimension (columns).
        bunch_ids (list[int] | None): List of bunch/particle IDs.
                                      Will default to ``[0, 1, ..., nbunches-1]`` if not set.
        meta (MetaDict): Dictionary of metadata.
        nbunches (int): Number of bunches/particles.
                        Automatically set (i.e. cannot be set in the initialization of this object).

    """

    matrices: Sequence[DataType]
    nturns: int
    bunch_ids: list[int] | None  = None  # will default in post_init
    meta: MetaDict = field(default_factory=dict)
    nbunches: int = field(init=False)

    def __post_init__(self):
        self.nbunches = len(self.matrices)

        if self.nturns is None or self.nturns < 1:
            raise ValueError("Number of turns needs to be larger than zero.")

        if self.bunch_ids is None:
            self.bunch_ids = list(range(self.nbunches))  # we always need bunch-ids

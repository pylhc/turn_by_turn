"""
Iota
----

Data handling for turn-by-turn measurement files from ``Iota`` (files in **hdf5** format).
"""

from __future__ import annotations

import abc
import logging
from enum import Enum
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import pandas as pd

from turn_by_turn.errors import HDF5VersionError
from turn_by_turn.structures import TbtData, TransverseData

LOGGER = logging.getLogger(__name__)


class Version(int, Enum):
    one: int = 1
    two: int = 2


def read_tbt(file_path: str | Path, version: Version = Version.two) -> TbtData:
    """
    Reads turn-by-turn data from ``IOTA``'s **hdf5** format file.
    Beware, that there are two possible versions of the iota-HDF5 format.

    Args:
        file_path (Union[str, Path]): path to the turn-by-turn measurement file.
        version (int): the format version to use when reading the written file.
                       Defaults to the latest one, currently ``2``.

    Returns:
        A ``TbTData`` object with the loaded data.
    """
    file_path = Path(file_path)
    LOGGER.debug(f"Reading Iota file at path: '{file_path.absolute()}'")

    match version:
        case Version.one:
            read_data = VersionOneReader(file_path)
        case Version.two:
            read_data = VersionTwoReader(file_path)
        case _:
            raise ValueError(f"Version {version} unknown for IOTA reader.")

    return read_data.tbt_data


class AbstractIotaReader(abc.ABC):
    """Class that reads the IOTA turn-by-turn data.

    This abstract class implements the whole reading in its `__init__`,
    but cannot run by itself, as the version specific functions (see below)
    need to be implemented first.

    The read data is stored as TbtData-object in the `tbt_data` attribute.
    """

    def __init__(self, path: Path):
        self.path: Path = path

        with h5py.File(path, "r") as hdf5_file:
            self.hdf5_file: h5py.File = hdf5_file

            self._prepare()
            self.tbt_data = self._read_turn_by_turn_data()

    def _prepare(self):
        """Prepare attributes and check that the correct version is used."""
        bpm_names = [self.map_bpm_name(key) for key in self.hdf5_file if self.is_bpm_key(key)]
        if not bpm_names:
            msg = f"Wrong version of the IOTA-HDF5 format was used for file {self.path!s}!"
            LOGGER.error(msg)
            raise HDF5VersionError(msg)

        self.bpm_names: list[str] = list(dict.fromkeys(bpm_names))  # unique and keep order
        self.nbpms: int = len(self.bpm_names)
        self.nturns: int = self._get_number_of_turns()

    def _read_turn_by_turn_data(self) -> TbtData:
        """Read data and create the turn-by-turn data object."""
        return TbtData(
            bunch_ids=[1],
            nturns=self.nturns,
            matrices=[
                TransverseData(
                    X=pd.DataFrame(
                        index=self.bpm_names,
                        data=self._get_data_for_plane("X"),
                        dtype=float,
                    ),
                    Y=pd.DataFrame(
                        index=self.bpm_names,
                        data=self._get_data_for_plane("Y"),
                        dtype=float,
                    ),
                )
            ],
            meta={
                "file": self.path,
                "source_datatype": "iota",
            },
        )

    def _get_data_for_plane(self, plane: str) -> np.ndarray:
        """Extract the turn-by-turn data for the given plane as numpy array,
        truncated to the maximum common number of turns."""
        data = np.zeros((self.nbpms, self.nturns))
        bpm_keys = [key for key in self.hdf5_file if self.is_bpm_key(key, plane)]

        for i, key in enumerate(bpm_keys):
            data[i, :] = self._get_data_for_key(key, plane)[:self.nturns]

        return data

    def _get_number_of_turns(self) -> int:
        """Get the maximum common number of tuns over all BPMs,
        such that the arrays can be trimmed to be of equal lengths."""
        return min(
            len(self._get_data_for_key(key, plane))
            for plane in ("X", "Y")
            for key in self.hdf5_file
            if self.is_bpm_key(key, plane)
        )

    def _get_data_for_key(self, key: str, plane: Literal["X", "Y"]) -> np.ndarray:
        """Extract the turn-by-turn data for the given key and plane as numpy array."""
        ...

    @staticmethod
    def map_bpm_name(key: str) -> str:
        """Convert the given key to a BPM name."""
        ...

    @staticmethod
    def is_bpm_key(key: str, plane: Literal["X", "Y"] | None = None) -> bool:
        """Check if the entry of the file contains BPM data."""
        ...


class VersionOneReader(AbstractIotaReader):
    """Version 1 contains three keys per BPM: X, Y and Intensity."""

    planes: dict[str, str] = {"X": "H", "Y": "V"}

    def _get_data_for_key(self, key: str, plane: Literal["X", "Y"]) -> np.ndarray:
        """Extract the turn-by-turn data for the given key and plane as numpy array."""
        return self.hdf5_file[key]  # assumes plane is already in key name

    @staticmethod
    def map_bpm_name(key: str) -> str:
        """Convert the given key to a BPM name."""
        return f"IBPM{key[4:-1]}"

    @staticmethod
    def is_bpm_key(key: str, plane: Literal["X", "Y"] | None = None) -> bool:
        """Check if the entry of the file contains BPM data."""
        is_bpm = ("state" not in key) or key.startswith("N:")
        if plane is None:
            return is_bpm and (key.endswith(VersionOneReader.planes["X"]) or key.endswith(VersionOneReader.planes["Y"]))
        return is_bpm and key.endswith(VersionOneReader.planes[plane])


class VersionTwoReader(AbstractIotaReader):
    """Version 2 contains a single key per BPM, which contains data for both planes
    (and possibly more which we ignore)."""

    planes: dict[str, str] = {"X": "Horizontal", "Y": "Vertical"}

    def _get_data_for_key(self, key: str, plane: Literal["X", "Y"]) -> np.ndarray:
        """Extract the turn-by-turn data for the given key and plane as numpy array."""
        return self.hdf5_file[key][self.planes[plane]]

    @staticmethod
    def map_bpm_name(key: str) -> str:
        """Convert the given key to a BPM name."""
        return f"IBPM{key}"

    @staticmethod
    def is_bpm_key(key: str, plane: Literal["X", "Y"] | None = None) -> bool:
        """Check if the entry of the file contains BPM data."""
        return "NL" not in key and not key.startswith("N:")  # latter: filter v1 data to be safe

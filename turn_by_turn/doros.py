"""
DOROS
-----

Data handling for turn-by-turn measurement files from the 
``DOROS`` BPMs of the ``LHC``  (files in **hdf5** format).

The file contains entries for ``METADATA``, ``TIMESTAMPS_INDEX`` and
``TIMESTAMPS_TABLE``, which we do not use and 
then the actual data per BPM.

These entries are as follows:

- The timetamps in microseconds.

  - ``bstTimestamp``: timestamp of the trigger
  - ``acqStamp``: tiemstamp of the actual acquisition


- Position/Orbit entries are the average position of the beam per turn,
  i.e. the turn-by-turn data averaged over all bunches, as DOROS cannot
  distinguish between the bunches.

  - ``nbOrbitSamplesRead``: number of orbit samples read
  - ``horPositions``: horizontal position of the beam per turn
  - ``verPositions``: vertical position of the beam per turn


- Oscillation entries are the frequencies of change in position.

- ``nbOscillationSamplesRead``: number of oscillation samples read
- ``horOscillationData``: horizontal oscillation data
- ``verOscillationData``: vertical oscillation data
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
import h5py

import pandas as pd
from dateutil import tz

from turn_by_turn.structures import TbtData, TransverseData
from turn_by_turn.utils import all_elements_equal

LOGGER = logging.getLogger(__name__)

DEFAULT_BUNCH_ID: int = 0  # bunch ID not saved in the DOROS file 

METADATA: str = "METADATA"

# tiemstamps
BST_TIMESTAMP: str = "bstTimestamp"   # microseconds
ACQ_STAMP: str = "acqStamp"           # microseconds


class DataKeys:
    """ Class to handle the different entry keys for oscillations and positions. """
    OSCILLATIONS: str = "oscillations"
    POSITIONS: str = "positions"

    def __init__(self, default_value: float, n_samples: str, names: dict[str, str]):
        """Create an object containing the keys to use in the DOROS file,
        depending on the data to extract.

        Args:
            default_value (int): Default value when the data is not present. 
            n_samples (str): Key for the number of samples present in file entry. 
            names (dict[str, str]): Keys per plane for the actual tbt data.
        """
        self.default_value: float = default_value
        self.n_samples: str = n_samples
        self.data: dict[str, str] = names

    @classmethod
    def types(cls) -> tuple[str]:
        return (cls.OSCILLATIONS, cls.POSITIONS)

    @classmethod
    def get_data_keys(cls, data_type: str) -> DataKeys:
        if data_type == cls.OSCILLATIONS:
            return cls(
                default_value=-1,  # from FESA class
                n_samples="nbOscillationSamplesRead",
                names={
                    "X": "horOscillationData",
                    "Y": "verOscillationData",
                }
            )
        
        if data_type == cls.POSITIONS:
            return cls(
                default_value=-1,  # from FESA class
                n_samples="nbOrbitSamplesRead",
                names={
                    "X": "horPositions",
                    "Y": "verPositions",
                }
            )
        
        else: 
            msg = f"Unkown datatype '{data_type}'. Use one of {cls.types()}."
            raise ValueError(msg)
    
    @classmethod
    def get_other_data_keys(cls, data_type: str) -> DataKeys:
        if data_type not in cls.types():
            msg = f"Unkown datatype '{data_type}'. Use one of {cls.types()}."
            raise ValueError(msg)

        other_type = cls.POSITIONS if data_type == cls.OSCILLATIONS else cls.OSCILLATIONS
        return cls.get_data_keys(other_type)


def read_tbt(file_path: str|Path, bunch_id: int = DEFAULT_BUNCH_ID, data_type: str = DataKeys.OSCILLATIONS) -> TbtData:
    """
    Reads turn-by-turn data from the ``DOROS``'s **SDDS** format file.

    Args:
        file_path (Union[str, Path]): path to the turn-by-turn measurement file.
        bunch_id (int, optional): the ID of the bunch in the file. Defaults to 0.
        data_type(str): Datatype to load. Defaults to "oscillations".

    Returns:
        A ``TbTData`` object with the loaded data.
    """
    file_path = Path(file_path)
    LOGGER.debug(f"Reading DOROS {data_type} data at path: '{file_path.absolute()}'")
    data_keys = DataKeys.get_data_keys(data_type)
    
    with h5py.File(file_path, "r") as hdf_file:
        # use "/" to keep track of bpm order, see https://github.com/h5py/h5py/issues/1471
        bpm_names = [name for name in hdf_file["/"].keys() if data_keys.n_samples in hdf_file[f"/{name}"].keys()]
        LOGGER.debug(f"Found BPMs in DOROS-type file: {bpm_names}")

        _check_data_lengths(hdf_file, data_keys, bpm_names)

        time_stamps = [hdf_file[bpm][ACQ_STAMP][0] for bpm in bpm_names]
        date = datetime.fromtimestamp(min(time_stamps) / 1e6, tz=tz.tzutc())

        nturns = hdf_file[bpm_names[0]][data_keys.n_samples][0]  # equal lengths checked before
        matrices = [
            TransverseData(
                X=_create_dataframe(hdf_file, data_keys, bpm_names, plane="X"),
                Y=_create_dataframe(hdf_file, data_keys, bpm_names, plane="Y"),
            )
        ]
    return TbtData(matrices, date, [bunch_id], nturns)


def write_tbt(file_path: str|Path, tbt_data: TbtData, data_type: str = DataKeys.OSCILLATIONS) -> None:
    """
    Writes turn-by-turn data to the ``DOROS``'s **SDDS** format file.

    Args:
        tbt_data (TbtData): data to be written
        file_path (Union[str, Path]): path to the turn-by-turn measurement file.
    """
    if len(tbt_data.matrices) != 1:
        msg = "DOROS only supports one bunch."
        raise ValueError(msg)

    file_path = Path(file_path)
    LOGGER.debug(f"Writing DOROS file at path: '{file_path.absolute()}'")
    data_keys = DataKeys.get_data_keys(data_type)
    other_keys = DataKeys.get_other_data_keys(data_type)

    data = tbt_data.matrices[0]
    with h5py.File(file_path, "w", track_order=True) as hdf_file:
        hdf_file.create_group(METADATA)
        for bpm in data.X.index:
            hdf_file.create_group(bpm)
            hdf_file[bpm].create_dataset(ACQ_STAMP, data=[tbt_data.date.timestamp() * 1e6])
            hdf_file[bpm].create_dataset(BST_TIMESTAMP, data=[tbt_data.date.timestamp() * 1e6])

            hdf_file[bpm].create_dataset(data_keys.n_samples, data=[tbt_data.nturns])
            hdf_file[bpm].create_dataset(data_keys.data["X"], data=data.X.loc[bpm, :].values)
            hdf_file[bpm].create_dataset(data_keys.data["Y"], data=data.Y.loc[bpm, :].values)

            hdf_file[bpm].create_dataset(other_keys.n_samples, data=0)
            hdf_file[bpm].create_dataset(other_keys.data["X"], data=[other_keys.default_value])
            hdf_file[bpm].create_dataset(other_keys.data["Y"], data=[other_keys.default_value])


def _create_dataframe(hdf_file: h5py.File, data_keys: DataKeys, bpm_names: str, plane: str) -> pd.DataFrame:
    data = [hdf_file[bpm][data_keys.data[plane]] for bpm in bpm_names]
    return pd.DataFrame(index=bpm_names, data=data, dtype=float)


def _check_data_lengths(hdf_file: h5py.File, data_keys: DataKeys, bpm_names: str) -> None:
    """Confirm that the data lengths are as defined and same for all BPMs."""
    suspicious_bpms = []
    for bpm in bpm_names:
        n_turns = hdf_file[bpm][data_keys.n_samples][0]
        if n_turns != len(hdf_file[bpm][data_keys.data["X"]]) or n_turns != len(hdf_file[bpm][data_keys.data["Y"]]):
            suspicious_bpms.append(bpm)
    
    if suspicious_bpms:
        msg = f"Found BPMs with different data lengths than defined in '{data_keys.n_samples}': {suspicious_bpms}"
        raise ValueError(msg)

    if not all_elements_equal(hdf_file[bpm][data_keys.n_samples][0] for bpm in bpm_names):
        msg = "Not all BPMs have the same number of turns!"
        raise ValueError(msg)

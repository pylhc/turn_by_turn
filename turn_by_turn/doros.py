"""
DOROS
-----

Data handling for turn-by-turn measurement files from the 
``DOROS`` BPMs of the ``LHC``  (files in **hdf5** format).

The file contains an unused entry for ``METADATA`` and 
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


- Oscillation entries are the frequencies of change in position
  these are not used in the turn-by-turn format.

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

LOGGER = logging.getLogger()

DEFAULT_BUNCH_ID: int = 0  # bunch ID not saved in the DOROS file 

METADATA: str = "METADATA"
BPM_NAME_END: str = "_DOROS"

# tiemstamps
BST_TIMESTAMP: str = "bstTimestamp"   # microseconds
ACQ_STAMP: str = "acqStamp"           # microseconds

# Position data
N_ORBIT_SAMPLES: str = "nbOrbitSamplesRead"
POSITIONS: dict[str, str] = {
    "X": "horPositions",
    "Y": "verPositions",
}

# Oscillation data 
DEFAULT_OSCILLATION_DATA: int = -1  # from FESA class
N_OSCILLATION_SAMPLES: str = "nbOscillationSamplesRead"
OSCILLATIONS: dict[str, str] = {
    "X": "horOscillationData",
    "Y": "verOscillationData",
}

def read_tbt(file_path: str|Path, bunch_id: int = DEFAULT_BUNCH_ID) -> TbtData:
    """
    Reads turn-by-turn data from the ``DOROS``'s **SDDS** format file.

    Args:
        file_path (Union[str, Path]): path to the turn-by-turn measurement file.
        bunch_id (int, optional): the ID of the bunch in the file. Defaults to 0

    Returns:
        A ``TbTData`` object with the loaded data.
    """
    file_path = Path(file_path)
    LOGGER.debug(f"Reading DOROS file at path: '{file_path.absolute()}'")
    with h5py.File(file_path, "r") as hdf_file:
        # use "/" to keep track of bpm order, see https://github.com/h5py/h5py/issues/1471
        bpm_names = [name for name in hdf_file["/"].keys() if N_ORBIT_SAMPLES in hdf_file[f"/{name}"].keys()]
        LOGGER.debug(f"Found BPMs in DOROS-type file: {bpm_names}")

        _check_data_lengths(hdf_file, bpm_names)

        time_stamps = [hdf_file[bpm][ACQ_STAMP][0] for bpm in bpm_names]
        date = datetime.fromtimestamp(min(time_stamps) / 1e6, tz=tz.tzutc())

        nturns = hdf_file[bpm_names[0]][N_ORBIT_SAMPLES][0]  # equal lengths checked before
        matrices = [
            TransverseData(
                X=_create_dataframe(hdf_file, bpm_names, "X"),
                Y=_create_dataframe(hdf_file, bpm_names, "Y"),
            )
        ]
    return TbtData(matrices, date, [bunch_id], nturns)


def write_tbt(tbt_data: TbtData, file_path: str|Path) -> None:
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

    data = tbt_data.matrices[0]
    with h5py.File(file_path, "w", track_order=True) as hdf_file:
        hdf_file.create_group(METADATA)
        for bpm in data.X.index:
            hdf_file.create_group(bpm)
            hdf_file[bpm].create_dataset(ACQ_STAMP, data=[tbt_data.date.timestamp() * 1e6])
            hdf_file[bpm].create_dataset(BST_TIMESTAMP, data=[tbt_data.date.timestamp() * 1e6])

            hdf_file[bpm].create_dataset(N_ORBIT_SAMPLES, data=[tbt_data.nturns])
            hdf_file[bpm].create_dataset(POSITIONS["X"], data=data.X.loc[bpm, :].values)
            hdf_file[bpm].create_dataset(POSITIONS["Y"], data=data.Y.loc[bpm, :].values)

            hdf_file[bpm].create_dataset(N_OSCILLATION_SAMPLES, data=0)
            hdf_file[bpm].create_dataset(OSCILLATIONS["X"], data=[DEFAULT_OSCILLATION_DATA])
            hdf_file[bpm].create_dataset(OSCILLATIONS["Y"], data=[DEFAULT_OSCILLATION_DATA])


def _create_dataframe(hdf_file: h5py.File, bpm_names: str, plane: str) -> pd.DataFrame:
    data = [hdf_file[bpm][POSITIONS[plane]] for bpm in bpm_names]
    return pd.DataFrame(index=bpm_names, data=data, dtype=float)


def _check_data_lengths(hdf_file: h5py.File, bpm_names: str) -> None:
    """Confirm that the data lengths are as defined and same for all BPMs."""
    suspicious_bpms = []
    for bpm in bpm_names:
        n_turns = hdf_file[bpm][N_ORBIT_SAMPLES][0]
        if n_turns != len(hdf_file[bpm][POSITIONS["X"]]) or n_turns != len(hdf_file[bpm][POSITIONS["Y"]]):
            suspicious_bpms.append(bpm)
    
    if suspicious_bpms:
        msg = f"Found BPMs with different data lengths than defined in '{N_ORBIT_SAMPLES}': {suspicious_bpms}"
        raise ValueError(msg)

    if not all_elements_equal(hdf_file[bpm][N_ORBIT_SAMPLES][0] for bpm in bpm_names):
        msg = "Not all BPMs have the same number of turns!"
        raise ValueError(msg)

"""
Trackone
--------

Data handling for turn-by-turn measurement files from the ``MAD-X`` code, which can be obtained by performing
particle tracking of your machine through in ``MAD-X``. The files are very close in structure to **TFS**
files, with the difference that the data part is split into "segments" relating containing data for a given
observation point.
"""
import logging
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np

from turn_by_turn.structures import TbtData, TrackingData, TransverseData
from turn_by_turn.utils import numpy_to_tbt

LOGGER = logging.getLogger(__name__)


def read_tbt(file_path: Union[str, Path], is_tracking_data: bool = False) -> TbtData:
    """
    Reads turn-by-turn data from the ``MAD-X`` **trackone** format file.

    Args:
        file_path (Union[str, Path]): path to the turn-by-turn measurement file.
        is_tracking_data (bool): if ``True``, all (``X``, ``PX``, ``Y``, ``PY``,
            ``T``, ``PT``, ``S``, ``E``) fields are expected in the file as it
            is considered a full tracking simulation output. Those are then read
            into ``TrackingData`` objects. Defaults to ``False``.

    Returns:
        A ``TbTData`` object with the loaded data.
    """
    nturns, npart = get_trackone_stats(file_path)
    names, matrix = get_structure_from_trackone(nturns, npart, file_path)
    if is_tracking_data:
        # Converts full tracking output to TbTData.
        return numpy_to_tbt(names, matrix, datatype=TrackingData)
    else:
        # matrix[0, 2] contains just (x, y) samples.
        return numpy_to_tbt(names, matrix[[0, 2]], datatype=TransverseData)


def get_trackone_stats(file_path: Union[str, Path], write_out: bool = False) -> Tuple[int, int]:
    """
    Determines the number of particles and turns in the matrices from the provided ``MAD-X``
    **trackone** file.

    Args:
        file_path (Union[str, Path]): path to the turn-by-turn measurement file.
        write_out (bool): if ``True``, write out the determined stats to a **stats.txt** file.

    Returns:
        A tuple with the number of turns and particles.
    """
    stats_string = ""
    nturns, nparticles = 0, 0
    first_seg = True
    with Path(file_path).open("r") as input_file:
        for line in input_file:
            if len(line.strip()) == 0:
                continue
            if line.strip()[0] in ["@", "*", "$"]:
                stats_string = stats_string + line
                continue
            parts = line.split()
            if parts[0] == "#segment":
                if not first_seg:
                    break
                nturns = int(parts[2])
                nparticles = int(parts[3])
                first_seg = False
            if parts[0] == "-1":
                nparticles = 1
            stats_string = stats_string + line

    if write_out:
        LOGGER.debug(f"Writing tbt stats for file '{file_path.absolute()}' at 'stats.txt'")
        with Path("stats.txt").open("w") as stats_file:
            stats_file.write(stats_string)

    return nturns - 1, nparticles


def get_structure_from_trackone(
    nturns: int = 0, npart: int = 0, file_path: Union[str, Path] = "trackone"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts BPM names and particle coordinates in the **trackone** file produced by ``MAD-X``.

    Args:
        nturns (int): Number of turns tracked in the **trackone**, i.e. obtained from
            ``get_trackone_stats()``.
        npart (int):  Number of particles tracked in the **trackone**, i.e. obtained from
            ``get_trackone_stats()``.
        file_path (Union[str, Path]): path to the turn-by-turn measurement file.

    Returns:
        A numpy array of BPM names and a 4D Numpy array [quantity, BPM, particle/bunch No.,
        turn No.] quantities in order [x, px, y, py, t, pt, s, E].
    """
    bpms: Dict[str, np.ndarray] = dict()
    with Path(file_path).open("r") as input_file:
        for line in input_file:
            if len(line.strip()) == 0:
                continue
            if line.strip()[0] in ["@", "*", "$"]:
                continue
            parts = line.split()
            if parts[0] == "#segment":
                bpm_name = parts[-1].upper()
                if (np.all([k not in bpm_name.lower() for k in ["start", "end"]])) and (
                    bpm_name not in bpms.keys()
                ):
                    bpms[bpm_name] = np.empty([npart, nturns, 8], dtype=float)
            elif np.all([k not in bpm_name.lower() for k in ["start", "end"]]):
                bpms[bpm_name][np.abs(int(float(parts[0]))) - 1, int(float(parts[1])) - 1, :] = np.array(
                    parts[2:]
                )
    return np.array(list(bpms.keys())), np.transpose(np.array(list(bpms.values())), axes=[3, 0, 1, 2])

"""
Utils
-----

Utility functions for convenience operations on turn-by-turn data objects in this package.
"""
import logging

from typing import Dict, Sequence

import numpy as np
import pandas as pd

from turn_by_turn.constants import PLANES, PLANE_TO_NUM
from turn_by_turn.errors import ExclusiveArgumentsError
from turn_by_turn.structures import TbtData, TransverseData

LOGGER = logging.getLogger(__name__)


def generate_average_tbtdata(tbtdata: TbtData) -> TbtData:
    """
    Takes a ``TbtData`` object and returns another containing the averaged matrices over all
    bunches/particles at all used BPMs.

    Args:
        tbtdata (TbtData): entry TbtData object from measurements.

    Returns:
        A new TbtData object with the averaged matrices.
    """
    data = tbtdata.matrices
    bpm_names = data[0].X.index

    new_matrices = [
        TransverseData(
            X=pd.DataFrame(
                index=bpm_names,
                data=get_averaged_data(bpm_names, data, "X", tbtdata.nturns),
                dtype=float,
            ),
            Y=pd.DataFrame(
                index=bpm_names,
                data=get_averaged_data(bpm_names, data, "Y", tbtdata.nturns),
                dtype=float,
            ),
        )
    ]
    return TbtData(new_matrices, tbtdata.date, [1], tbtdata.nturns)


def get_averaged_data(
    bpm_names: Sequence[str], matrices: Sequence[TransverseData], plane: str, turns: int
) -> np.ndarray:
    """
    Average data from a given plane from the matrices of a ``TbtData``.

    Args:
        bpm_names (Sequence[str]):
        matrices (Sequence[TransverseData]): matrices from a ``TbtData`` object.
        plane (str): name of the given plane to average in.
        turns (int): number of turns in the provided data.

    Returns:
        A numpy array with the averaged data for the given bpms.
    """
    bpm_data: np.ndarray = np.empty((len(bpm_names), len(matrices), turns))
    bpm_data.fill(np.nan)

    for index, bpm in enumerate(bpm_names):
        for i, _ in enumerate(matrices):
            bpm_data[index, i, : len(matrices[i][plane].loc[bpm])] = matrices[i][plane].loc[bpm]

    return np.nanmean(bpm_data, axis=1)


def matrices_to_array(tbt_data: TbtData) -> np.ndarray:
    """
    Convert  the matrices of a ``TbtData`` object to a numpy array.

    Args:
        tbt_data (TbtData): ``TbtData`` object to convert the data from.

    Returns:
        A numpy array with the matrices data.
    """
    LOGGER.debug("Getting number of BPMs from the measurement data.")
    n_bpms = tbt_data.matrices[0].X.index.size
    data = np.empty((2, n_bpms, tbt_data.nbunches, tbt_data.nturns), dtype=float)

    LOGGER.debug("Converting matrices data ")
    for index in range(tbt_data.nbunches):
        for plane in PLANES:
            data[PLANE_TO_NUM[plane], :, index, :] = tbt_data.matrices[index][plane].to_numpy()
    return data


def add_noise(data: np.ndarray, noise: float = None, sigma: float = None, seed: int = None) -> np.ndarray:
    """
    Returns the given data with added noise. Noise is generated as a standard normal distribution (mean=0,
    standard_deviation=1) with the size of the input data, and scaled by the a factor before being added to
    the provided data. Said factor can either be provided, or calculated from the input data's own standard
    deviation.

    Args:
        data (np.ndarray): your input data.
        noise (float): the scaling factor applied to the generated noise.
        sigma (float): if provided, then that number times the standard deviation of the input data will
            be used as scaling factor for the generated noise.
        seed(int): a given seed to initialise the RNG.

    Returns:
        A new numpy array with added noise to the provided data.
    """
    if (noise is None and sigma is None) or (noise is not None and sigma is not None):
        raise ExclusiveArgumentsError("noise", "sigma")

    if sigma is None:
        scaling = noise
    else:
        scaling = sigma * np.std(data, dtype=np.float64)

    return np.array(data + scaling * np.random.default_rng(seed).standard_normal(data.shape))


def add_noise_to_tbt(data: TbtData, noise: float = None, sigma: float = None, seed: int = None) -> TbtData:
    """
    Returns a new copy of the given TbT data with added noise.
    The noise is generated by :func:`turn_by_turn.utils.add_noise` from a single rng,
    i.e. the noise is not repeated on each dataframe.

    Args:
        data (TbtData): your input TbT-data.
        noise (float): the scaling factor applied to the generated noise.
        sigma (float): if provided, then that number times the standard deviation of the input data will
            be used as scaling factor for the generated noise.
        seed(int): a given seed to initialise the RNG.

    Returns:
        A copy of the TbtData with noised data on all matrices.
    """
    rng = np.random.default_rng(seed)  # as we don't want to use the same noise on all dataframes
    return TbtData(
        matrices=[TransverseData(
            X=pd.DataFrame(index=m.X.index, data=add_noise(m.X.to_numpy(), noise, sigma, seed=rng)),
            Y=pd.DataFrame(index=m.Y.index, data=add_noise(m.Y.to_numpy(), noise, sigma, seed=rng))
        ) for m in data.matrices],
        date=data.date,
        bunch_ids=data.bunch_ids,
        nturns=data.nturns,
    )


def numpy_to_tbt(names: np.ndarray, matrix: np.ndarray) -> TbtData:
    """
    Converts turn by turn matrices and names into a ``TbTData`` object.

    Args:
        names (np.ndarray): Numpy array of BPM names.
        matrix (np.ndarray): 4D Numpy array [quantity, BPM, particle/bunch No., turn No.]
            quantities in order [x, y].

    Returns:
        A ``TbtData`` object loaded with the matrices in the provided numpy arrays.
    """
    # get list of TbTFile from 4D matrix ...
    _, _, nbunches, nturns = matrix.shape
    matrices = []
    indices = []
    for index in range(nbunches):
        matrices.append(
            TransverseData(
                X=pd.DataFrame(index=names, data=matrix[0, :, index, :]),
                Y=pd.DataFrame(index=names, data=matrix[1, :, index, :]),
            )
        )
        indices.append(index)
    return TbtData(matrices=matrices, bunch_ids=indices, nturns=nturns)

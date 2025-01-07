"""
MAD-NG
------

This module provides functions to read and write ``MAD-NG`` turn-by-turn measurement files. These files
are in the **TFS** format.

"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import tfs

from turn_by_turn.structures import TbtData, TransverseData

LOGGER = logging.getLogger()

# Define the column names in the TFS file
NAME = "name"
ELEMENT_INDEX = "eidx"
TURN = "turn"
PARTICLE_ID = "id"

def read_tbt(file_path: str | Path) -> TbtData:
    """
    Reads turn-by-turn data from the ``MAD-NG`` **TFS** format file.

    Args:
        file_path (str | Path): path to the turn-by-turn measurement file.

    Returns:
        A ``TbTData`` object with the loaded data.
    """
    LOGGER.debug("Starting to read TBT data from dataframe")
    df = tfs.read(file_path)

    nturns = int(df.iloc[-1].loc[TURN])
    npart = int(df.iloc[-1].loc[PARTICLE_ID])
    LOGGER.info(f"Number of turns: {nturns}, Number of particles: {npart}")

    # Set the order of the columns
    df = df.set_index([PARTICLE_ID, TURN, ELEMENT_INDEX])
    df = df.sort_index()
    
    # Get the names of the observed points (BPMs) from the first particle's first turn
    observe_points = df.loc[(1, 1)][NAME].to_numpy()
    num_observables = len(observe_points)

    # Check if the number of observed points is consistent for all particles/turns
    if len(df[NAME]) / nturns / npart != num_observables:
        raise ValueError(
            "The number of BPMs (or observed points) is not consistent for all particles/turns. Simulation may have lost particles."
        )

    matrices = []
    bunch_ids = range(1, npart + 1) # Particle IDs start from 1 (not 0)

    for particle_id in bunch_ids:
        LOGGER.info(f"Processing particle ID: {particle_id}")

        # Filter the dataframe for the current particle
        df_particle = df.loc[particle_id]

        # Create a dictionary of the TransverseData fields
        tracking_data_dict = {
            plane: pd.DataFrame(
                index=observe_points,
                data=df_particle[plane.lower()]  # MAD-NG uses lower case for the planes
                .to_numpy()
                .reshape(num_observables, nturns, order="F"),
                # ^ Number of Observables x Number of turns, Fortran order (So that the observables are the rows)
            )
            for plane in TransverseData.fieldnames() # X, Y
        }

        # Append the TransverseData object to the matrices list
        # We don't use TrackingData, as MAD-NG does not provide energy
        matrices.append(TransverseData(**tracking_data_dict))

    LOGGER.debug("Finished reading TBT data")
    return TbtData(matrices=matrices, bunch_ids=list(bunch_ids), nturns=nturns)

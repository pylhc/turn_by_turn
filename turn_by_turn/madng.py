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
    df = tfs.read(file_path)
    LOGGER.debug("Starting to read TBT data from dataframe")

    nturns = int(df.iloc[-1].loc[TURN])
    npart = int(df.iloc[-1].loc[PARTICLE_ID])
    LOGGER.info(f"Number of turns: {nturns}, Number of particles: {npart}")

    # Get the unique BPMs and number of BPMs
    bpms = df[NAME].unique()
    nbpms = len(bpms)

    # Set the index to the particle ID
    df.set_index([PARTICLE_ID], inplace=True)

    matrices = []
    bunch_ids = range(1, npart + 1) # Particle IDs start from 1 (not 0)
    for particle_id in bunch_ids:
        LOGGER.info(f"Processing particle ID: {particle_id}")

        # Filter the dataframe for the current particle
        df_particle = df.loc[particle_id]

        # Check if the number of BPMs is consistent for all particles/turns (i.e. no lost particles)
        if len(df_particle[NAME]) / nturns != nbpms:
            raise ValueError(
                "The number of BPMs is not consistent for all particles/turns. Simulation may have lost particles."
            )

        # Set the index to the element index, which are unique for every BPM and turn
        df_particle.set_index([ELEMENT_INDEX], inplace=True)

        # Create a dictionary of the TransverseData fields
        tracking_data_dict = {
            plane: pd.DataFrame(
                index=bpms,
                data=df_particle[plane.lower()]  # MAD-NG uses lower case field names
                .to_numpy()
                .reshape(nbpms, nturns, order="F"),
                # ^ Number of BPMs x Number of turns, Fortran order (So that the BPMs are the rows)
            )
            for plane in TransverseData.fieldnames() # X, Y
        }

        # Append the TransverseData object to the matrices list
        # We don't use TrackingData, as MAD-NG does not provide energy
        matrices.append(TransverseData(**tracking_data_dict))

    LOGGER.debug("Finished reading TBT data")
    # Should we also provide date? (jgray 2024)
    return TbtData(matrices=matrices, bunch_ids=list(bunch_ids), nturns=nturns)

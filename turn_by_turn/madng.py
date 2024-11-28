"""
MAD-NG
------

This module provides functions to read and write ``MAD-NG`` turn-by-turn measurement files. These files
are in the **TFS** format.

"""

from __future__ import annotations

import logging

import pandas as pd
import tfs

from turn_by_turn.structures import TbtData, TransverseData

LOGGER = logging.getLogger()


# def read_tbt(file_path: str | Path) -> TbtData:
def read_tbt(df: tfs.TfsDataFrame) -> TbtData:
    LOGGER.info("Starting to read TBT data")
    """
    Reads turn-by-turn data from the ``MAD-NG`` **TFS** format file.

    Args:
        file_path (str | Path): path to the turn-by-turn measurement file.

    Returns:
        A ``TbTData`` object with the loaded data.
    """
    # df = tfs.read(file_path)
    LOGGER.info("Starting to read TBT data from dataframe")
    
    nturns = int(df.iloc[-1].loc["turn"])
    npart = int(df.iloc[-1].loc["id"])
    LOGGER.info(f"Number of turns: {nturns}, Number of particles: {npart}")

    # Get the unique BPMs and number of BPMs
    bpms = df["name"].unique()
    nbpms = len(bpms)

    # Set the index to the particle ID
    df.set_index(["id"], inplace=True)

    matrices = []
    for particle_id in range(npart):
        LOGGER.info(f"Processing particle ID: {particle_id + 1}")
        
        # Filter the dataframe for the current particle and set index to the matrix dims
        subdf = df.loc[particle_id + 1]  # Particle ID starts from 1 (not 0)

        # Check if the number of BPMs is consistent for all particles/turns (i.e. no lost particles)
        assert (
            len(subdf["name"]) / nturns == nbpms
        ), "The number of BPMs is not consistent for all particles/turns. Simulation may have lost particles."

        # Set the index to the element index, which are unique for every BPM and turn
        subdf.set_index(["eidx"], inplace=True)

        # Create a dictionary of the TransverseData fields
        tracking_data_dict = {
            field: pd.DataFrame(
                index=bpms,
                data=subdf[field.lower()] # MAD-NG uses lower case field names
                .to_numpy()
                .reshape(nbpms, nturns, order="F"),  
                #^ Number of BPMs x Number of turns, Fortran order (So that the BPMs are the rows)
            )
            for field in TransverseData.fieldnames()
        }

        # Append the TransverseData object to the matrices list
        # We don't use TrackingData, as MAD-NG does not provide energy
        matrices.append(TransverseData(**tracking_data_dict))

    LOGGER.info("Finished reading TBT data")
    # Should we also provide date? (jgray 2024)
    return TbtData(matrices=matrices, bunch_ids=list(range(npart)), nturns=nturns)

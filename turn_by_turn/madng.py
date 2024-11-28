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
        print(len(subdf["name"]), nbpms, nturns)
        assert (
            len(subdf["name"]) / nturns == nbpms
        ), "The number of BPMs is not consistent for all particles/turns. Simulation may have lost particles."
        subdf.set_index(
            ["eidx"], inplace=True
        )  # Must set index after getting unique BPMs

        # Create a dictionary of the TrackingData fields
        tracking_data_dict = {
            field: pd.DataFrame(
                index=bpms,
                data=subdf[field.lower()]
                .to_numpy()
                .reshape(nbpms, nturns, order="F"),  # Number of BPMs x Number of turns
            )
            for field in TransverseData.fieldnames()
        }

        # Append the TrackingData object to the matrices list
        matrices.append(TransverseData(**tracking_data_dict))

    LOGGER.info("Finished reading TBT data")
    return TbtData(matrices=matrices, bunch_ids=list(range(npart)), nturns=nturns)

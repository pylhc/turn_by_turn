"""
MAD-NG
------

This module provides functions to read and write ``MAD-NG`` turn-by-turn measurement files. These files
are in the **TFS** format.

"""

from __future__ import annotations

from datetime import datetime
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

# Define the header names in the TFS file
HNAME = "name"
ORIGIN = "origin"
DATE = "date"
TIME = "time"
REFCOL = "refcol"


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

    # Get the date and time from the headers (return None if not found)
    date_str = df.headers.get(DATE)
    time_str = df.headers.get(TIME)
    
    # Combine the date and time into a datetime object
    date = None
    if date_str and time_str:
        date = datetime.strptime(f"{date_str} {time_str}", "%d/%m/%y %H:%M:%S")
    elif date_str:
        date = datetime.strptime(date_str, "%d/%m/%y")

    nturns = int(df.iloc[-1].loc[TURN])
    npart = int(df.iloc[-1].loc[PARTICLE_ID])
    LOGGER.info(f"Number of turns: {nturns}, Number of particles: {npart}")

    # Get the names of the observed points (BPMs) from the first particle's first turn
    df = df.set_index([PARTICLE_ID, TURN]).sort_index()
    observe_points = df.loc[(1, 1)][NAME].to_numpy()
    num_observables = len(observe_points)

    # Check if the number of observed points is consistent for all particles/turns
    if len(df[NAME]) / nturns / npart != num_observables:
        raise ValueError(
            "The number of BPMs (or observed points) is not consistent for all particles/turns. Simulation may have lost particles."
        )

    matrices = []
    bunch_ids = range(1, npart + 1)  # Particle IDs start from 1 (not 0)

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
            for plane in TransverseData.fieldnames()  # X, Y
        }

        # Append the TransverseData object to the matrices list
        # We don't use TrackingData, as MAD-NG does not provide energy
        matrices.append(TransverseData(**tracking_data_dict))

    LOGGER.debug("Finished reading TBT data")
    return TbtData(matrices=matrices, bunch_ids=list(bunch_ids), nturns=nturns, date=date)


def write_tbt(output_path: str | Path, tbt_data: TbtData) -> None:
    """
    Writes turn-by-turn data to a TFS file for MAD-NG.

    Args:
        tbt_data (TbtData): Turn-by-turn data to write.
        file_path (str | Path): Target file path.
    """
    planes = [plane.lower() for plane in TransverseData.fieldnames()]  # x, y
    plane_dfs = {plane: [] for plane in planes}

    for particle_id, transverse_data in zip(tbt_data.bunch_ids, tbt_data.matrices):
        for plane in planes:
            # Create a dataframe for the current plane and particle
            particle_df: pd.DataFrame = transverse_data[plane.upper()].copy()

            # Create the name column from the index
            particle_df.index.name = NAME
            particle_df = particle_df.reset_index()

            # Add the element index column (to be used for merging)
            particle_df[ELEMENT_INDEX] = particle_df.index

            # Melt the dataframe to have columns: name, element index, turn, plane
            particle_df = pd.melt(
                particle_df,
                id_vars=[NAME, ELEMENT_INDEX],
                var_name=TURN,
                value_name=plane,
            )

            # Add the particle ID column
            particle_df[PARTICLE_ID] = particle_id

            # Convert the turn column to integer and increment by 1 (MAD-NG uses 1-based indexing)
            particle_df[TURN] = particle_df[TURN].astype(int) + 1

            # Append the dataframe to the list
            plane_dfs[plane].append(particle_df)

    # Merge the dataframes on name, turn, particle ID and element index for both planes
    df_x = pd.concat(plane_dfs[planes[0]])
    df_y = pd.concat(plane_dfs[planes[1]])
    merged_df = pd.merge(df_x, df_y, on=[NAME, TURN, PARTICLE_ID, ELEMENT_INDEX])
    merged_df = merged_df.set_index([NAME])

    # Sort the dataframe by turn, element index and particle ID (so the format is consistent with MAD-NG)
    merged_df = merged_df.sort_values(by=[TURN, ELEMENT_INDEX, PARTICLE_ID])

    # Drop the element index column (this is not the real element index, but a temporary one for merging)
    merged_df = merged_df.drop(columns=[ELEMENT_INDEX])

    # Set the columns to x, y, turn, id, for consistency.
    merged_df = merged_df[[planes[0], planes[1], TURN, PARTICLE_ID]]

    # Write the dataframe to a TFS file
    headers = {
        HNAME: "TbtData",
        ORIGIN: "Python",
        DATE: tbt_data.date.strftime("%d/%m/%y"),
        TIME: tbt_data.date.strftime("%H:%M:%S"),
        REFCOL: NAME,
    }
    tfs.write(output_path, merged_df, headers_dict=headers, save_index=NAME)

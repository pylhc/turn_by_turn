"""
MAD-NG
------

This module provides functions to read and write turn-by-turn measurement data
produced by the ``MAD-NG`` code. MAD-NG stores its tracking data in the **TFS**
(Table File System) file format.

Data is loaded into the standardized ``TbtData`` structure used by ``turn_by_turn``,
allowing easy post-processing and conversion between formats.

Dependencies:
    - Requires the ``tfs-pandas >= 4.0.0`` package for compatibility with MAD-NG
      features. Earlier versions does not support MAD-NG TFS files.

"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path  # Only used for type hinting

    import tfs

from turn_by_turn.structures import TbtData, TransverseData

LOGGER = logging.getLogger(__name__)

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
    Read turn-by-turn data from a MAD-NG TFS file.

    Loads the TFS file using ``tfs-pandas`` and converts its contents into a
    ``TbtData`` object for use with the ``turn_by_turn`` toolkit.

    Args:
        file_path (str | Path): Path to the MAD-NG TFS measurement file.

    Returns:
        TbtData: The loaded turn-by-turn data.

    Raises:
        ImportError: If the ``tfs-pandas`` package is not installed.
    """
    try:
        import tfs
    except ImportError as e:
        raise ImportError(
            "The 'tfs' package is required to read MAD-NG TFS files. "
            "Install it with: pip install 'turn_by_turn[madng]'"
        ) from e

    LOGGER.debug("Starting to read TBT data from dataframe")
    df = tfs.read(file_path)
    return convert_to_tbt(df)


def convert_to_tbt(df: pd.DataFrame | tfs.TfsDataFrame) -> TbtData:
    """
    Convert a TFS or pandas DataFrame to a ``TbtData`` object.

    This function parses the required turn-by-turn columns, reconstructs the
    particle-by-particle tracking data, and returns a ``TbtData`` instance
    that can be written or converted to other formats.

    Args:
        df (pd.DataFrame | TfsDataFrame):
            DataFrame containing MAD-NG turn-by-turn tracking data.

    Returns:
        TbtData: The extracted and structured turn-by-turn data.

    Raises:
        TypeError: If the input is not a recognized DataFrame type.
        ValueError: If the data structure is inconsistent (e.g., lost particles).
    """

    # Get the date and time from the headers (return None if not found)
    try:
        import tfs

        is_tfs_df = isinstance(df, tfs.TfsDataFrame)
    except ImportError:
        LOGGER.debug("The 'tfs' package is not installed. Assuming a pandas DataFrame.")
        is_tfs_df = False

    if is_tfs_df:
        date_str = df.headers.get(DATE)
        time_str = df.headers.get(TIME)
    else:
        date_str = df.attrs.get(DATE)
        time_str = df.attrs.get(TIME)

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
            "The number of observed points is not consistent for all particles/turns. "
            "Simulation may have lost particles."
        )

    matrices = []

    # Particle IDs start from 1, but we use 0-based indexing in Python
    particle_ids = range(npart)
    for particle_id in particle_ids:
        LOGGER.info(f"Processing particle ID: {particle_id}")

        # Filter the dataframe for the current particle
        df_particle = df.loc[particle_id + 1]

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
    return TbtData(
        matrices=matrices, bunch_ids=list(particle_ids), nturns=nturns, date=date
    )


def write_tbt(output_path: str | Path, tbt_data: TbtData) -> None:
    """
    Write turn-by-turn data to a MAD-NG TFS file.

    Takes a ``TbtData`` object and writes its contents to disk in the standard
    TFS format used by MAD-NG, including relevant headers (date, time, origin).

    Args:
        output_path (str | Path): Destination file path for the TFS file.
        tbt_data (TbtData): The turn-by-turn data to write.

    Raises:
        ImportError: If the ``tfs-pandas`` package is not installed.
    """
    try:
        import tfs
    except ImportError as e:
        raise ImportError(
            "The 'tfs' package is required to write MAD-NG TFS files. Install it with: pip install 'turn_by_turn[madng]'"
        ) from e

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
            particle_df[PARTICLE_ID] = particle_id + 1

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

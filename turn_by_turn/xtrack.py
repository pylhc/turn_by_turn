"""
XTRACK
------

This module provides functions to convert tracking results from an ``xtrack`` Line
(or its internal ParticlesMonitor elements) into the standardized ``TbtData`` format
used throughout ``turn_by_turn``.
"""


from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from turn_by_turn.structures import TbtData, TransverseData

try:
    from xtrack import Line
    from xtrack.monitors import ParticlesMonitor

    HAS_XTRACK = True
except ImportError:
    HAS_XTRACK = False

LOGGER = logging.getLogger()


def convert_to_tbt(xline: Line) -> TbtData:
    """
    Convert tracking results from an ``xtrack`` Line into a ``TbtData`` object.

    This function extracts all ``ParticlesMonitor`` elements found in the Line,
    verifies they contain consistent turn-by-turn data, and assembles the results
    into the standard ``TbtData`` format. One ``TransverseData`` matrix is created
    per tracked particle.

    Args:
        xline (Line): An ``xtrack`` Line containing at least one ``ParticlesMonitor``.

    Returns:
        TbtData: The extracted turn-by-turn data for all particles and monitors.

    Raises:
        ImportError: If the ``xtrack`` library is not installed.
        TypeError: If the input is not a valid ``xtrack.Line``.
        ValueError: If no monitors are found or data is inconsistent.
    """
    if not HAS_XTRACK:
        raise ImportError(
            "The 'xtrack' package is required to convert xtrack Line objects. Install it with: pip install 'xtrack>=0.84.7'"
        )
    if not isinstance(xline, Line):
        raise TypeError(f"Expected an xtrack Line object, got {type(xline)} instead.")

    # Collect monitor names and monitor objects in order from the line
    monitor_names, monitors = zip(
        *[
            (name, elem)
            for name, elem in zip(xline.element_names, xline.elements)
            if isinstance(elem, ParticlesMonitor)
        ]
    )
    # Check that we have at least one monitor
    if not monitors:
        raise ValueError(
            "No ParticlesMonitor found in the Line. Please add a ParticlesMonitor to the Line."
        )

    # Check that all monitors have the same number of turns
    nturns_set = {mon.data.at_turn.max() + 1 for mon in monitors}
    if len(nturns_set) != 1:
        raise ValueError(
            "Monitors have different number of turns, maybe some lost particles?"
        )
    nturns = nturns_set.pop()

    # Check that all monitors have the same number of particles
    npart_set = {len(set(mon.data.particle_id)) for mon in monitors}
    if len(npart_set) != 1:
        raise ValueError(
            "Monitors have different number of particles, maybe some lost particles?"
        )
    npart = npart_set.pop()

    # Precompute masks for each monitor and particle_id (Half the time to compute this bit)
    monitor_particle_masks = [
        [mon.data.particle_id == particle_id for particle_id in range(npart)]
        for mon in monitors
    ]

    matrices = []
    # Loop over each particle ID
    for particle_id in range(npart):
        # For each plane (e.g., 'X', 'Y'), build a DataFrame: rows=BPMs, cols=turns
        tracking_data_dict = {
            plane: pd.DataFrame(
                np.vstack(
                    [
                        getattr(mon.data, plane.lower())[
                            monitor_particle_masks[i][particle_id]
                        ]
                        for i, mon in enumerate(monitors)
                    ]
                ),
                index=monitor_names,
            )
            for plane in TransverseData.fieldnames()
        }
        # Create a TransverseData object for this particle and add to the list
        matrices.append(TransverseData(**tracking_data_dict))

    # Return the TbtData object containing all particles' data
    return TbtData(
        matrices=matrices, bunch_ids=list(range(npart)), nturns=nturns, date=None
    )

# Added this function to match the interface, but it is not implemented - should I not include it?
def read_tbt(path: str | Path) -> None:
    """
    Not implemented.

    Reading TBT data directly from files is not supported for xtrack.
    Use ``convert_to_tbt`` to convert an in-memory ``xtrack.Line`` instead.
    """
    raise NotImplementedError(
        "Reading TBT data from xtrack Line files is not implemented."
    )

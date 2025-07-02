"""
XTRACK
------

This module provides functions to convert tracking results from the ``xtrack`` library into the
``turn_by_turn`` format. 

"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from xtrack import Line
from xtrack.monitors import ParticlesMonitor
from turn_by_turn.structures import TbtData, TransverseData

LOGGER = logging.getLogger()



def convert_to_tbt(xline: Line) -> TbtData:
    """
    Converts the results from an ``xtrack`` Line object into a ``TbtData`` object.

    Args:
        xline (Line): An ``xtrack`` Line object containing the tracking results.

    Returns:
        TbtData: A ``TbtData`` object containing the turn-by-turn data for all particles.
    """
    # Collect monitor names and monitor objects in order from the line
    monitor_names, monitors = zip(*[
        (name, elem)
        for name, elem in zip(xline.element_names, xline.elements)
        if isinstance(elem, ParticlesMonitor)
    ])

    # Check that all monitors have the same number of turns
    nturns_set = {mon.data.at_turn.max() + 1 for mon in monitors}
    if len(nturns_set) != 1:
        raise ValueError("Monitors have different number of turns, maybe some lost particles?")
    nturns = nturns_set.pop()

    # Check that all monitors have the same number of particles
    npart_set = {len(set(mon.data.particle_id)) for mon in monitors}
    if len(npart_set) != 1:
        raise ValueError("Monitors have different number of particles, maybe some lost particles?")
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
                np.vstack([
                    getattr(mon.data, plane.lower())[monitor_particle_masks[i][particle_id]]
                    for i, mon in enumerate(monitors)
                ]),
                index=monitor_names
            )
            for plane in TransverseData.fieldnames()
        }
        # Create a TransverseData object for this particle and add to the list
        matrices.append(TransverseData(**tracking_data_dict))

    # Return the TbtData object containing all particles' data
    return TbtData(matrices=matrices, bunch_ids=list(range(npart)), nturns=nturns, date=None)
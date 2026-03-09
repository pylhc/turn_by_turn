"""
XTrack TbT Conversion from Particle Monitors
--------------------------------------------

Convert tracking results produced by one or more ``xtrack.ParticlesMonitor``
elements into the standardised ``TbtData`` format used by ``turn_by_turn``.

Usage
=====

    from turn_by_turn.xtrack import convert_to_tbt

    # after building particles and tracking a line containing
    # xt.ParticlesMonitor elements, convert to TbtData:
    tbt = convert_to_tbt(line)

Prerequisites for using ``convert_to_tbt``:

1. The input ``Line`` must contain one or more ``ParticlesMonitor`` elements
     positioned at each location where turn-by-turn data is required (e.g., all BPMs).

     A valid monitor setup involves:

     - Placing a ``xt.ParticlesMonitor`` instance in the line's element sequence
         at all the places you would like to observe.
     - Configuring each monitor with identical settings:

             * ``start_at_turn`` (first turn to record, usually 0)
             * ``stop_at_turn`` (The total number of turns to record, e.g., 100)
             * ``num_particles`` (number of tracked particles)

     If any monitor is configured with different parameters, ``convert_to_tbt``
     will either find no data or raise an inconsistency error.

     Also, if you specify more turns than were actually tracked, the resulting
     TBT data will include all turns up to the monitor's configured limit.
     This may result in extra rows filled with zeros for turns where no real
     data was recorded, which might not be desirable for your analysis.

2. Before conversion, you must:

     - Build particles with the desired initial coordinates
         (using ``line.build_particles(...)``).
     - Track those particles through the line for the intended number of turns
         (using ``line.track(..., num_turns=num_turns)``).

Once these conditions are met, pass the tracked ``Line`` to ``convert_to_tbt`` to
extract the data from each particle monitor into a ``TbtData`` object.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from turn_by_turn.structures import TbtData, TransverseData

if TYPE_CHECKING:
    import xtrack as xt  # noqa: TC004

LOGGER = logging.getLogger(__name__)


def is_line_suitable_for_conversion(xline: xt.Line) -> bool:
    """
    Check if the given xtrack Line is suitable for conversion to TbtData.

    This function verifies that the Line contains at least one ParticlesMonitor
    and that all monitors have consistent tracking data (same number of turns and particles).

    Args:
        xline (xt.Line): The xtrack Line to check.

    Returns:
        bool: True if the Line is suitable for conversion, False otherwise.
    """
    try:
        import xtrack as xt
    except ImportError as e:
        raise ImportError(
            "The 'xtrack' package is required to convert from xtrack monitors. "
            "Install it with: python -m pip install 'turn_by_turn[xtrack]'"
        ) from e

    return any(isinstance(elem, xt.ParticlesMonitor) for elem in xline.elements)


def convert_to_tbt(xline: xt.Line) -> TbtData:
    """
    Convert tracking results from an ``xtrack`` Line into a ``TbtData`` object.

    This function extracts all ``ParticlesMonitor`` elements found in the Line,
    verifies they contain consistent turn-by-turn data, and assembles the results
    into the standard ``TbtData`` format. One ``TransverseData`` matrix is created
    per tracked particle.

    Args:
        xline (Line): An ``xtrack.Line`` containing at least one ``ParticlesMonitor``.

    Returns:
        TbtData: The extracted turn-by-turn data for all particles and monitors.

    Raises:
        ValueError: If no monitors are found or data is inconsistent.
    """
    try:
        import xtrack as xt
    except ImportError as e:
        raise ImportError(
            "The 'xtrack' package is required to convert from xtrack monitors. "
            "Install it with: python -m pip install 'turn_by_turn[xtrack]'"
        ) from e

    # Collect monitor names and monitor objects in order from the line
    monitor_pairs = [
        (name, elem)
        for name, elem in zip(xline.element_names, xline.elements)
        if isinstance(elem, xt.ParticlesMonitor)
    ]
    # Check that we have at least one monitor
    if not monitor_pairs:
        raise ValueError(
            "No ParticlesMonitor found in the Line. Please add a ParticlesMonitor to the Line."
        )
    monitor_names, monitors = zip(*monitor_pairs)

    # First check that no particles were lost during tracking. There will be trailing
    # zeros in the data if particles were lost. This might be difficult to detect.
    if not all(mon.data.particle_id[-1] == mon.data.particle_id.max() for mon in monitors):
        raise ValueError(
            "Some particles were lost during tracking, which is not supported by this function. "
            "Ensure that all particles are tracked through the entire line without loss."
        )

    # Check that all monitors have the same number of turns
    nturns_set = {mon.data.at_turn.max() + 1 for mon in monitors}
    if len(nturns_set) != 1:
        raise ValueError(
            "Monitors have different number of turns, have you set the monitors with different 'start_at_turn' or 'stop_at_turn' parameters?"
        )
    nturns = nturns_set.pop()

    # Check that all monitors have the same number of particles
    npart_set = {len(set(mon.data.particle_id)) for mon in monitors}
    if len(npart_set) != 1:
        raise ValueError("Monitors have different number of particles, maybe some lost particles?")
    npart = npart_set.pop()

    # Precompute masks for each monitor and particle_id
    monitor_pid_masks = [
        mon.data.particle_id[:, None] == np.arange(npart)[None, :] for mon in monitors
    ]

    matrices = []
    # Loop over each particle ID (pid)
    for pid in range(npart):
        # For each plane (e.g., 'X', 'Y'), build a DataFrame: rows=BPMs, cols=turns
        tracking_data_dict = {}
        for plane in TransverseData.fieldnames():
            # fmt: off
            stacked = np.vstack([
                getattr(mon.data, plane.lower())[monitor_pid_masks[i][:, pid]]
                for i, mon in enumerate(monitors)
            ])
            # fmt: on
            tracking_data_dict[plane] = pd.DataFrame(
                stacked,
                index=monitor_names,
            )
        # Create a TransverseData object for this particle and add to the list
        matrices.append(TransverseData(**tracking_data_dict))

    # Return the TbtData object containing all particles' data
    return TbtData(
        matrices=matrices,
        bunch_ids=list(range(npart)),
        nturns=nturns,
        meta={"source_datatype": "xtrack_particle_monitors", "date": pd.Timestamp.now()},
    )

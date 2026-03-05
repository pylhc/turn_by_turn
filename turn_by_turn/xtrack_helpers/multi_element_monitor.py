"""
XTrack Multi-Element Monitor Conversion
---------------------------------------

Convert Xsuite ``MultiElementMonitor`` output into ``TbtData``.
Reference: https://xsuite.readthedocs.io/en/latest/track.html#multi-element-monitor

Usage
=====
1. Track with ``multi_element_monitor_at``::

    import xtrack as xt
    # place a MultiElementMonitor in the line via the xtrack API
    monitor_names = ["BPM1", "BPM2", "BPM3"]
    line.track(
         particles,
         num_turns=1024,
         multi_element_monitor_at=monitor_names,
    )

2. Convert to ``turn_by_turn`` data::

    from turn_by_turn.xtrack_helpers import multi_element_monitor

    tbt = multi_element_monitor.convert_to_tbt(line)

Notes
=====
- Data is read from ``line.record_multi_element_last_track``.
- The converter expects arrays with shape ``(turn, particle, obs)`` from
  ``monitor.get("x")`` and ``monitor.get("y")``.
- Observation order in the output follows ``monitor.obs_names``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from turn_by_turn.structures import TbtData, TransverseData

if TYPE_CHECKING:
    from pathlib import Path

    import xtrack as xt

def is_line_suitable_for_conversion(xline: xt.Line) -> bool:
    """
    Check if the given xtrack Line is suitable for conversion to TbtData.

    This function verifies that the Line contains a non None multi-element monitor data.

    Args:
        xline (xt.Line): The xtrack Line to check.
    Returns:
        bool: True if the Line is suitable for conversion, False otherwise.
    """
    return hasattr(xline, "record_multi_element_last_track") and xline.record_multi_element_last_track is not None


def convert_to_tbt(xline: xt.Line) -> TbtData:
    """
    Convert ``xtrack`` multi-element monitor data to ``TbtData``.

    Args:
        xline (xt.Line): Tracked line containing ``record_multi_element_last_track``.

    Returns:
        TbtData: Turn-by-turn data for each particle and observed element.

    Raises:
        ImportError: If ``xtrack`` is not installed.
        TypeError: If ``xline`` is not an ``xtrack.Line``.
        ValueError: If monitor data is missing, or has unexpected shape.
    """
    monitor: xt.MultiElementMonitor = xline.record_multi_element_last_track
    nturns = int(monitor.stop_at_turn - monitor.start_at_turn)
    npart = int(monitor.part_id_end - monitor.part_id_start)
    nobs = len(monitor.obs_names)
    expected_shape = (nturns, npart, nobs)

    # monitor.get("x"/"y") returns (turn, particle, obs).
    x_all = np.asarray(monitor.get("x"))
    y_all = np.asarray(monitor.get("y"))

    for arr in (x_all, y_all):
        if arr.shape != expected_shape:
            raise ValueError(
                f"Unexpected monitor array shape {arr.shape}; expected {expected_shape}."
            )

    bunch_ids = list(range(npart))
    x_stack = np.transpose(x_all, (2, 0, 1))  # (obs, turn, particle)
    y_stack = np.transpose(y_all, (2, 0, 1))  # (obs, turn, particle)

    matrices = [
        TransverseData(
            X=pd.DataFrame(x_stack[:, :, pid], index=monitor.obs_names),
            Y=pd.DataFrame(y_stack[:, :, pid], index=monitor.obs_names),
        )
        for pid in bunch_ids
    ]

    return TbtData(
        matrices=matrices,
        bunch_ids=bunch_ids,
        nturns=nturns,
        meta={"source_datatype": "xtrack_multi_element_monitor", "date": pd.Timestamp.now()},
    )


def read_tbt(path: str | Path) -> None:
    """
    Not implemented.

    Reading ``xtrack`` multi-element monitor data directly from files is not supported.
    Use ``convert_to_tbt`` on an in-memory tracked ``xtrack.Line`` instead.
    """
    raise NotImplementedError("Reading TBT data from xtrack multi-element monitor files is not implemented.")

"""
XTrack TbT Conversion from Lines
--------------------------------

Helpers to convert data produced by the ``xtrack`` tracking framework into the
``turn_by_turn`` ``TbtData`` format.

This module converts monitor data from ``xtrack`` into the corresponding
``TbtData`` representation. It currently supports data produced by:

- ``xtrack.MultiElementMonitor`` recorded via ``xtrack.Line.record_multi_element_last_track``.
- One or more ``xtrack.ParticlesMonitor`` elements placed in the line.

The appropriate converter is selected based on which monitor data are present
on the provided ``xtrack.Line``. If multi-element monitor data are available, they
are preferred; otherwise particle monitor data are used.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from turn_by_turn.xtrack import _multi_element_monitor as multi_element_monitor
from turn_by_turn.xtrack import _particle_monitors as particle_monitors

if TYPE_CHECKING:
    from pathlib import Path

    from xtrack import Line

    from turn_by_turn.structures import TbtData


# Added this function to match the interface, but it is not implemented.
def read_tbt(path: str | Path) -> None:
    """
    Not implemented.

    Reading TBT data directly from files is not supported for xtrack.
    Use ``convert_to_tbt`` to convert an in-memory ``xtrack.Line`` instead.
    """
    raise NotImplementedError("Reading TBT data from xtrack Line files is not implemented.")


def convert_to_tbt(xline: Line) -> TbtData:
    """
    Convert tracking results from an ``xtrack.Line`` into a ``TbtData`` object.

    Dispatches to one of the specific converters in ``turn_by_turn.xtrack``
    depending on which monitor data is available in the provided ``Line``.

    Supported source datatypes and resulting ``TbtData.meta['source_datatype']``
    values:

    - ``xtrack_multi_element_monitor``: when converting from an
        ``xtrack.MultiElementMonitor`` (via ``record_multi_element_last_track``).
    - ``xtrack_particle_monitors``: when converting from one or more
        ``xtrack.ParticlesMonitor`` elements.

    Args:
        xline (Line): An ``xtrack.Line`` containing monitor data.

    Returns:
        TbtData: The extracted turn-by-turn data. The ``meta`` mapping contains a
        ``source_datatype`` key describing the origin of the data.

    Raises:
        ImportError: If the ``xtrack`` package is not installed.
        TypeError: If the input is not an ``xtrack.Line``.
        ValueError: If no suitable monitor data is found on the provided Line.
    """
    try:
        import xtrack as xt
    except ImportError as e:
        raise ImportError(
            "The 'xtrack' package is required to convert from xtrack monitors. "
            "Install it with: python -m pip install 'turn_by_turn[xtrack]'"
        ) from e

    if not isinstance(xline, xt.Line):
        raise TypeError(f"Expected an xtrack Line object, got {type(xline)} instead.")

    if multi_element_monitor.is_line_suitable_for_conversion(xline):
        return multi_element_monitor.convert_to_tbt(xline)

    if particle_monitors.is_line_suitable_for_conversion(xline):
        return particle_monitors.convert_to_tbt(xline)

    raise ValueError(
        "No suitable monitor data found on the provided xtrack.Line. "
        "The line must contain either recorded multi-element monitor data "
        "(from xtrack.MultiElementMonitor via Line.record_multi_element_last_track) "
        "or one or more xtrack.ParticlesMonitor elements with completed tracking data "
        "that are still present and accessible."
    )

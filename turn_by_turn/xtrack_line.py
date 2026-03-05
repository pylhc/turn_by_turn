"""
XTrack Line Conversion
----------------------

Helpers to convert data produced by the ``xtrack`` tracking framework into the
``turn_by_turn`` ``TbtData`` format.

This module dispatches conversion to the implementations in
``turn_by_turn.xtrack_helpers``:

- ``multi_element_monitor``: converts data produced by xtrack's
    ``MultiElementMonitor`` (recorded on ``line.record_multi_element_last_track``).
- ``particle_monitors``: converts data produced by one or more
    ``ParticlesMonitor`` elements placed in the line.

The dispatch order prefers the multi-element monitor converter when
``record_multi_element_last_track`` is present; otherwise it falls back to
the particle monitors converter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from turn_by_turn.xtrack_helpers import multi_element_monitor, particle_monitors

if TYPE_CHECKING:
    from xtrack import Line

    from turn_by_turn.structures import TbtData


def convert_to_tbt(xline: Line) -> TbtData:
    """
    Convert tracking results from an ``xtrack.Line`` into a ``TbtData`` object.

    Dispatches to one of the helper converters in ``turn_by_turn.xtrack_helpers``
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
    """
    try:
        import xtrack as xt
    except ImportError as e:
        raise ImportError(
            "The 'xtrack' package is required to convert xtrack Line objects. "
            "Install it with: pip install 'turn_by_turn[xtrack]'"
        ) from e

    if not isinstance(xline, xt.Line):
        raise TypeError(f"Expected an xtrack Line object, got {type(xline)} instead.")

    if multi_element_monitor.is_line_suitable_for_conversion(xline):
        return multi_element_monitor.convert_to_tbt(xline)

    if particle_monitors.is_line_suitable_for_conversion(xline):
        return particle_monitors.convert_to_tbt(xline)

    raise ValueError(
        "No suitable monitor data found in line. "
        "Ensure you tracked with either particle monitors or multi-element monitors, and that tracking completed successfully."
    )

import sys

import numpy as np
import pytest
import xtrack as xt

from tests.test_lhc_and_general import compare_tbt
from turn_by_turn.io import convert_to_tbt as io_convert_to_tbt
from turn_by_turn.structures import TbtData
from turn_by_turn.xtrack import _multi_element_monitor, _particle_monitors
from turn_by_turn.xtrack.converter import convert_to_tbt as xtrack_convert_to_tbt
from turn_by_turn.xtrack.converter import read_tbt


@pytest.mark.skipif(sys.platform == "win32", reason="xtrack not supported on Windows")
def test_convert_xsuite(example_line: xt.Line, example_fake_tbt: TbtData):
    # Build the particles
    particles = example_line.build_particles(x=[1e-3, -1e-3], y=[-1e-3, 1e-3])

    # Track the particles through the line
    example_line.track(particles, num_turns=3)

    # Convert to TbtData using xtrack
    tbt_data = _particle_monitors.convert_to_tbt(example_line)
    compare_tbt(example_fake_tbt, tbt_data, no_binary=True)
    assert tbt_data.meta["source_datatype"] == "xtrack_particle_monitors"

    # Now convert using the generic function
    tbt_data = io_convert_to_tbt(example_line, datatype="xtrack")
    compare_tbt(example_fake_tbt, tbt_data, no_binary=True)
    assert tbt_data.meta["source_datatype"] == "xtrack_particle_monitors"


@pytest.mark.skipif(sys.platform == "win32", reason="xtrack not supported on Windows")
@pytest.mark.skipif(
    xt.__version__ < "0.99.0", reason="xtrack version does not support multi-element monitor"
)
def test_convert_xsuite_multi_element_monitor(example_line: xt.Line, example_fake_tbt: TbtData):
    particles = example_line.build_particles(x=[1e-3, -1e-3], y=[-1e-3, 1e-3])
    monitor_names = ["BPM1", "BPM3", "BPM2"]
    example_line.track(particles, num_turns=3, multi_element_monitor_at=monitor_names)

    tbt_data = _multi_element_monitor.convert_to_tbt(example_line)
    compare_tbt(example_fake_tbt, tbt_data, no_binary=True)
    assert tbt_data.meta["source_datatype"] == "xtrack_multi_element_monitor"

    tbt_data = io_convert_to_tbt(example_line, datatype="xtrack")
    compare_tbt(example_fake_tbt, tbt_data, no_binary=True)
    assert tbt_data.meta["source_datatype"] == "xtrack_multi_element_monitor"


@pytest.mark.skipif(sys.platform == "win32", reason="xtrack not supported on Windows")
def test_read_tbt_raises_not_implemented():
    with pytest.raises(
        NotImplementedError, match="Reading TBT data from xtrack Line files is not implemented"
    ):
        read_tbt("dummy_path")


@pytest.mark.skipif(sys.platform == "win32", reason="xtrack not supported on Windows")
def test_convert_to_tbt_invalid_type():
    with pytest.raises(TypeError, match="Expected an xtrack Line object"):
        xtrack_convert_to_tbt("not a line")  # ty:ignore[invalid-argument-type]


@pytest.mark.skipif(sys.platform == "win32", reason="xtrack not supported on Windows")
def test_convert_to_tbt_no_monitors():
    # Create a line without monitors
    line = xt.Line(elements=[xt.Drift(length=1.0)], element_names=["drift"])
    line.particle_ref = xt.Particles(p0c=1e9, q0=1.0, mass0=xt.ELECTRON_MASS_EV)

    # First try with no tracker, xtrack raises a RuntimeError if the line doesn't have a suitable tracker, but we want to catch that and raise a ValueError instead.
    with pytest.raises(
        ValueError, match="No suitable monitor data found on the provided xtrack.Line."
    ):
        xtrack_convert_to_tbt(line)

    # Now try with a tracker but no monitors, which should also raise the same ValueError.
    line.build_particles(x=[1e-3], y=[-1e-3])
    with pytest.raises(
        ValueError, match="No suitable monitor data found on the provided xtrack.Line."
    ):
        xtrack_convert_to_tbt(line)


# --- Fixtures ---- #
@pytest.fixture(scope="module")
def example_line():
    """
    Creates a simple xtrack Line with three BPMs and two quadrupoles.
    This replicates the MAD-NG example used in the tests exactly, changes
    will likely break the tests.
    See tests/inputs/madng/fodo_track.mad for the original MAD-NG file.
    """
    lcell = 20.0
    f = lcell / np.sin(np.pi / 4.0) / 4.0
    k = 1 / f
    nturns = 3
    qf = xt.Multipole(knl=[0.0, k], ksl=[0.0, 0.0])
    qd = xt.Multipole(knl=[0.0, -k], ksl=[0.0, 0.0])
    drift = xt.Drift(length=10.0)
    # fmt: off
    line = xt.Line(
        elements=[
            xt.ParticlesMonitor(start_at_turn=0, stop_at_turn=nturns, num_particles=2),
            qf,            drift,
            qd,            drift,
            qf,            drift,
            xt.ParticlesMonitor(start_at_turn=0, stop_at_turn=nturns, num_particles=2),
            qd,            drift,
            qf,            drift,
            qd,            drift,
            xt.ParticlesMonitor(start_at_turn=0, stop_at_turn=nturns, num_particles=2),
            ],
        element_names=[
            'BPM1',
            'qf_0', 'drift_0',
            'qd_0', 'drift_1',
            'qf_1', 'drift_2',
            'BPM3',  # Deliberately not in order to test the conversion
            'qd_1', 'drift_3',
            'qf_2', 'drift_4',
            'qd_2', 'drift_5',
            'BPM2'
            ]
        )
    # fmt: on
    line.particle_ref = xt.Particles(p0c=1e9, q0=1.0, mass0=xt.ELECTRON_MASS_EV)
    return line

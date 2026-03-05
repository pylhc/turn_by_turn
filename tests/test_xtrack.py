import sys

import numpy as np
import pytest
import xtrack as xt

from tests.test_lhc_and_general import compare_tbt
from turn_by_turn import xtrack_helpers
from turn_by_turn.io import convert_to_tbt
from turn_by_turn.structures import TbtData


@pytest.mark.skipif(sys.platform == "win32", reason="xtrack not supported on Windows")
def test_convert_xsuite(example_line: xt.Line, example_fake_tbt: TbtData):
    # Build the particles
    particles = example_line.build_particles(x=[1e-3, -1e-3], y=[-1e-3, 1e-3])

    # Track the particles through the line
    example_line.track(particles, num_turns=3)

    # Convert to TbtData using xtrack
    tbt_data = xtrack_helpers.particle_monitors.convert_to_tbt(example_line)
    compare_tbt(example_fake_tbt, tbt_data, no_binary=True)
    assert tbt_data.meta["source_datatype"] == "xtrack_particle_monitors"

    # Now convert using the generic function
    tbt_data = convert_to_tbt(example_line, datatype="xtrack")
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

    tbt_data = xtrack_helpers.multi_element_monitor.convert_to_tbt(example_line)
    compare_tbt(example_fake_tbt, tbt_data, no_binary=True)
    assert tbt_data.meta["source_datatype"] == "xtrack_multi_element_monitor"

    tbt_data = convert_to_tbt(example_line, datatype="xtrack")
    compare_tbt(example_fake_tbt, tbt_data, no_binary=True)
    assert tbt_data.meta["source_datatype"] == "xtrack_multi_element_monitor"


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

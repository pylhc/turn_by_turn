import numpy as np
import xtrack as xt
import pytest

from tests.test_lhc_and_general import compare_tbt
from turn_by_turn.structures import TbtData
from turn_by_turn import xtrack
from turn_by_turn.io import convert_to_tbt

def test_convert_xsuite(example_line: xt.Line, example_tbt: TbtData):
    # Build the particles
    particles = example_line.build_particles(x=[1e-3,-1e-3], y=[-1e-3, 1e-3])

    # Track the particles through the line
    example_line.track(particles, num_turns=3)
    
    # Convert to TbtData using xtrack
    tbt_data = xtrack.convert_to_tbt(example_line)
    compare_tbt(example_tbt, tbt_data, no_binary=True)

    # Now convert using the generic function
    tbt_data = convert_to_tbt(example_line, data_type="xtrack")
    compare_tbt(example_tbt, tbt_data, no_binary=True)

# --- Fixtures ---- #
@pytest.fixture(scope="module")
def example_line():
    lcell = 20
    f = lcell/np.sin(np.pi/4)/4
    k = 1/f
    nturns = 3
    qf = xt.Multipole(knl=[0, k], ksl=[0,0])
    qd = xt.Multipole(knl=[0,-k], ksl=[0,0])
    drift = xt.Drift(length=10.)
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
    line.particle_ref = xt.Particles(p0c=1e9, q0=1, mass0=xt.ELECTRON_MASS_EV)
    return line
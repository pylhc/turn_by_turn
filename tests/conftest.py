import numpy as np
import pandas as pd
import pytest
from turn_by_turn.structures import TbtData, TransverseData

@pytest.fixture(scope="session")
def example_fake_tbt():
    """
    Returns a TbtData object using simulation data taken from MAD-NG. 
    This data is also used for the tests in xtrack, so change the numbers
    at your own risk.

    It is possible to run the MAD-NG in the inputs folder to regenerate the data.
    Also, xtrack produces the same data, so you can use the xtrack test fixture 
    `example_line`.
    """
    names = np.array(["BPM1", "BPM3", "BPM2"])
    # First BPM
    bpm1_p1_x = np.array([ 1e-3, 0.002414213831,-0.0009999991309])
    bpm1_p1_y = np.array([-1e-3, 0.0004142133507, 0.001000000149])
    bpm1_p2_x = np.array([-1e-3,-0.002414213831, 0.0009999991309])
    bpm1_p2_y = np.array([ 1e-3,-0.0004142133507,-0.001000000149])

    # Second BPM
    bpm3_p1_x = np.array([ 0.002414213831,-0.0009999991309,-0.002414214191])
    bpm3_p1_y = np.array([ 0.0004142133507, 0.001000000149,-0.0004142129907])
    bpm3_p2_x = np.array([-0.002414213831, 0.0009999991309, 0.002414214191])
    bpm3_p2_y = np.array([-0.0004142133507,-0.001000000149, 0.0004142129907])

    # Third BPM
    bpm2_p1_x = np.array([-0.0009999999503,-0.0004142138307, 0.0009999998012])
    bpm2_p1_y = np.array([ 0.00100000029,-0.002414213351,-0.001000001159])
    bpm2_p2_x = np.array([ 0.0009999999503, 0.0004142138307,-0.0009999998012])
    bpm2_p2_y = np.array([-0.00100000029, 0.002414213351, 0.001000001159])

    matrix = [
        TransverseData(  # first particle
            X=pd.DataFrame(index=names, data=[bpm1_p1_x, bpm2_p1_x, bpm3_p1_x]),
            Y=pd.DataFrame(index=names, data=[bpm1_p1_y, bpm2_p1_y, bpm3_p1_y]),
        ),
        TransverseData(  # second particle
            X=pd.DataFrame(index=names, data=[bpm1_p2_x, bpm2_p2_x, bpm3_p2_x]),
            Y=pd.DataFrame(index=names, data=[bpm1_p2_y, bpm2_p2_y, bpm3_p2_y]),
        ),
    ]
    return TbtData(matrices=matrix, bunch_ids=[0, 1], nturns=3)

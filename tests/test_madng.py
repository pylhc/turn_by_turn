
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from tests.test_lhc_and_general import INPUTS_DIR, compare_tbt
from turn_by_turn import madng, read_tbt, write_tbt
from turn_by_turn.structures import TbtData, TransverseData


def test_read_ng(_ng_file):
    original = _original_simulation_data()
    
    # Check directly from the module
    new = madng.read_tbt(_ng_file)
    compare_tbt(original, new, no_binary=True)
    
    # Check from the main function
    new = read_tbt(_ng_file, datatype="madng")
    compare_tbt(original, new, no_binary=True)

def test_write_ng(_ng_file, tmp_path):
    original_tbt = _original_simulation_data()
    
    # Write the data
    from_tbt = tmp_path / "from_tbt.tfs"
    madng.write_tbt(from_tbt, original_tbt)
    
    # Read the written data
    new_tbt = madng.read_tbt(from_tbt)
    compare_tbt(original_tbt, new_tbt, no_binary=True)

    # Check from the main function
    original_tbt = read_tbt(_ng_file, datatype="madng")
    original_tbt.date = datetime.now() # Not tested, but checks that it runs
    write_tbt(from_tbt, original_tbt, datatype="madng")

    new_tbt = read_tbt(from_tbt, datatype="madng")
    compare_tbt(original_tbt, new_tbt, no_binary=True)    
    
def test_error_ng(_error_file):
    with pytest.raises(ValueError):
        read_tbt(_error_file, datatype="madng")

# ---- Helpers ---- #
def _original_simulation_data() -> TbtData:
    # Create a TbTData object with the original data
    names = np.array(["BPM1", "BPM3", "BPM2"])
    bpm1_p1_x = np.array([ 1e-3, 0.002414213831,-0.0009999991309])
    bpm1_p1_y = np.array([-1e-3, 0.0004142133507, 0.001000000149])
    bpm1_p2_x = np.array([-1e-3,-0.002414213831, 0.0009999991309])
    bpm1_p2_y = np.array([ 1e-3,-0.0004142133507,-0.001000000149])

    bpm2_p1_x = np.array([-0.0009999999503,-0.0004142138307, 0.0009999998012])
    bpm2_p1_y = np.array([ 0.00100000029,-0.002414213351,-0.001000001159])
    bpm2_p2_x = np.array([ 0.0009999999503, 0.0004142138307,-0.0009999998012])
    bpm2_p2_y = np.array([-0.00100000029, 0.002414213351, 0.001000001159])

    bpm3_p1_x = np.array([ 0.002414213831,-0.0009999991309,-0.002414214191])
    bpm3_p1_y = np.array([ 0.0004142133507, 0.001000000149,-0.0004142129907])
    bpm3_p2_x = np.array([-0.002414213831, 0.0009999991309, 0.002414214191])
    bpm3_p2_y = np.array([-0.0004142133507,-0.001000000149, 0.0004142129907])

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
    return TbtData(matrices=matrix, bunch_ids=[1, 2], nturns=3)


# ---- Fixtures ---- #
@pytest.fixture
def _ng_file(tmp_path):
    return INPUTS_DIR / "madng" / "fodo_track.tfs"

@pytest.fixture
def _error_file(tmp_path):
    return INPUTS_DIR / "madng" / "fodo_track_error.tfs"
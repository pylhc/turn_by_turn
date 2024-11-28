
import numpy as np
import pandas as pd
import pytest

from tests.test_lhc_and_general import INPUTS_DIR, compare_tbt
from turn_by_turn.structures import TbtData, TransverseData
from turn_by_turn.madng import read_tbt
from pymadng import MAD


def test_read_ng(_ng_file):
    with MAD() as mad:
        file_as_string = mad.quote_strings(str(_ng_file))
        df = mad.mtable.read(file_as_string).eval().to_df()
    new = read_tbt(df)
    origin = _original_simulation_data()
    compare_tbt(origin, new, True)

# ---- Helpers ---- #
def _original_simulation_data() -> TbtData:
    # Create a TbTData object with the original data
    names = np.array(["BPM1", "BPM2", "BPM3"])
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

    print(pd.DataFrame(index=names, data=[bpm1_p1_x, bpm2_p1_x, bpm3_p1_x]))
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


# ---- Fixtures ---- #
@pytest.fixture
def _ng_file(tmp_path):
    return INPUTS_DIR / "madng" / "fodo_track.tfs"
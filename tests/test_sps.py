from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.test_lhc_and_general import compare_tbt, INPUTS_DIR, create_data
from turn_by_turn import sps, TbtData, TransverseData


def test_read_write_real_data(_sps_file, tmp_path):
    input_sdds = sps.read_tbt(_sps_file)
    tmp_sdds = tmp_path / "sps.sdds"
    sps.write_tbt(tmp_sdds, input_sdds)
    read_sdds = sps.read_tbt(tmp_sdds)
    compare_tbt(input_sdds, read_sdds, no_binary=True)


def test_write_read(tmp_path):
    nturns = 1324
    nbpms_x = 350
    nbpms_y = 353
    original = TbtData(
        nturns=nturns,
        matrices=[
            TransverseData(
                X=pd.DataFrame(
                    index=[f"BPM{i}.H" for i in range(nbpms_x)],
                    data=create_data(np.linspace(-np.pi, np.pi, nturns, endpoint=False), nbpms_x, np.sin)
                ),
                Y=pd.DataFrame(
                    index=[f"BPM{i}.V" for i in range(nbpms_y)],
                    data=create_data(np.linspace(-np.pi, np.pi, nturns, endpoint=False), nbpms_y, np.cos)
                ),
            )
        ],
    )
    tmp_sdds = tmp_path / "sps_fake_data.sdds"
    sps.write_tbt(tmp_sdds, original)
    read_sdds = sps.read_tbt(tmp_sdds)
    compare_tbt(original, read_sdds, no_binary=True)


@pytest.fixture()
def _sps_file() -> Path:
    return INPUTS_DIR / "test_sps.sdds"



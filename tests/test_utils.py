from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from tests.test_lhc_and_general import create_data, compare_tbt
from turn_by_turn.constants import PLANES
from turn_by_turn.errors import ExclusiveArgumentsError
from turn_by_turn.structures import TbtData, TransverseData
from turn_by_turn.utils import add_noise, add_noise_to_tbt, generate_average_tbtdata


def test_noise_addition():
    array = create_data(np.linspace(-np.pi, np.pi, 2000, endpoint=False), 1, np.sin).flatten()
    noised = add_noise(array, noise=0)
    np.testing.assert_array_equal(array, noised)

    noised = add_noise(array, sigma=0)
    np.testing.assert_array_equal(array, noised)

    noised = add_noise(array, noise=5)
    assert np.std(array) != np.std(noised)
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(array, noised)

    noised = add_noise(array, sigma=1)
    assert np.std(array) != np.std(noised)
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(array, noised)


def test_noise_addition_to_tbt():
    nturns = 1000
    nbpms = 500
    nbunches = 10

    original = TbtData(
        nturns=nturns,
        matrices=[
            TransverseData(
                X=pd.DataFrame(
                    index=[f"BPM{i}" for i in range(nbpms)],
                    data=create_data(np.linspace(-np.pi, np.pi, nturns, endpoint=False), nbpms, np.sin)
                ),
                Y=pd.DataFrame(
                    index=[f"BPM{i}" for i in range(nbpms)],
                    data=create_data(np.linspace(-np.pi, np.pi, nturns, endpoint=False), nbpms, np.cos)
                ),
            )
            for _ in range(nbunches)
        ],
    )

    noised = add_noise_to_tbt(original, noise=0)
    compare_tbt(original, noised, no_binary=False)

    noise = 5
    noised = add_noise_to_tbt(original, noise=noise, seed=14783)
    for m_original, m_noised in zip(original.matrices, noised.matrices):
        for df_original, df_noised in ((m_original.X, m_noised.X), (m_original.Y, m_noised.Y)):
            with pytest.raises(AssertionError):
                np.testing.assert_array_equal(df_original.to_numpy(), df_noised.to_numpy())
            df_noise = df_original - df_noised
            av_deviation_std = (df_noise.std() - noise).mean()
            av_mean = df_noise.mean().mean()
            assert np.abs(av_deviation_std) < 0.02
            assert np.abs(av_mean) < 0.02
            # debugging:
            # print(av_deviation_std)
            # print(av_mean)


@pytest.mark.parametrize("seed", [1236, 6749, 23495564])
def test_noise_addition_with_seed(seed):
    array = create_data(np.linspace(-np.pi, np.pi, 2000, endpoint=False), 1, np.sin).flatten()

    noised_1 = add_noise(array, sigma=5, seed=seed)
    noised_2 = add_noise(array, sigma=5, seed=seed)
    np.testing.assert_array_equal(noised_1, noised_2)  # should be equal with same noise seed

    noised_3 = add_noise(array, noise=5, seed=seed * 5)
    with pytest.raises(AssertionError):  # Should be different with different seeds
        np.testing.assert_array_equal(noised_1, noised_3)


def test_add_noise_raises_on_both_arguments():
    array = create_data(np.linspace(-np.pi, np.pi, 2000, endpoint=False), 1, np.sin).flatten()
    with pytest.raises(ExclusiveArgumentsError):
        _ = add_noise(array, noise=5, sigma=1)


def test_compare_average_Tbtdata():
    npart = 10
    data = {
        plane: np.concatenate(
            [
                [
                    create_data(
                        np.linspace(1, 10, 10, endpoint=False, dtype=int),
                        2,
                        (lambda x: np.random.randn(len(x))),
                    )
                ]
                for _ in range(npart)
            ],
            axis=0,
        )
        for plane in PLANES
    }

    origin = TbtData(
        matrices=[
            TransverseData(
                X=pd.DataFrame(index=["IBPMA1C", "IBPME2R"], data=data["X"][i], dtype=float),
                Y=pd.DataFrame(index=["IBPMA1C", "IBPME2R"], data=data["Y"][i], dtype=float),
            )
            for i in range(npart)
        ],
        date=datetime.now(),
        bunch_ids=range(npart),
        nturns=10,
    )

    new = TbtData(
        matrices=[
            TransverseData(
                X=pd.DataFrame(
                    index=["IBPMA1C", "IBPME2R"],
                    data=np.mean(data["X"], axis=0),
                    dtype=float,
                ),
                Y=pd.DataFrame(
                    index=["IBPMA1C", "IBPME2R"],
                    data=np.mean(data["Y"], axis=0),
                    dtype=float,
                ),
            )
        ],
        date=datetime.now(),
        bunch_ids=[1],
        nturns=10,
    )

    compare_tbt(generate_average_tbtdata(origin), new, False)

import pytest
import numpy as np
import pandas as pd
from pandemic.model import pandemic


def test_pandemic_runs():

    trade = np.array([[0, 500, 15], [50, 0, 10], [20, 30, 0]])
    distances = np.array([[1, 5000, 105000], [5000, 1, 7500], [10500, 7500, 1]])
    locations = pd.DataFrame(
        {
            "name": ["United States", "China", "Brazil"],
            "phytosanitary_compliance": [0.00, 0.00, 0.00],
            "Presence": [True, False, True],
            "Host Percent Area": [0.25, 0.50, 0.35],
        }
    )

    e = pandemic(
        trade=trade,
        distances=distances,
        locations=locations,
        alpha=0.2,
        beta=1,
        mu=0.0002,
        lamda_c=1,
        phi=5,
        sigma_epsilon=0.5,
        sigma_h=0.5,
        sigma_kappa=0.5,
        sigma_phi=2,
        sigma_T=20,
    )
    assert (e[0] >= 0).all() and (e[0] <= 1).all()
    assert (e[1] >= 0).all() and (e[1] <= 1).all()

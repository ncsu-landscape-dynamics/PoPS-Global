import numpy as np
import pandas as pd

from pandemic.model_equations import pandemic_single_time_step


def test_pandemic_runs():

    trade = np.array([[0, 500, 15], [50, 0, 10], [20, 30, 0]])
    min_Tc = np.min(trade)
    max_Tc = np.max(trade)
    distances = np.array([[1, 5000, 105000], [5000, 1, 7500], [10500, 7500, 1]])
    climate_similarities = np.array([[1, 0.95, 0.4], [0.95, 1, 0.5], [0.4, 0.5, 1]])
    time_step = "2015"
    season_dict = {
        "NH_season": ["09", "10", "11", "12", "01", "02", "03", "04"],
        "SH_season": ["04", "05", "06", "07", "08", "09", "10"],
    }
    time_infect = 3
    locations = pd.DataFrame(
        {
            "NAME": ["United States", "China", "Brazil"],
            "ISO3": ["USA", "CHN", "BRA"],
            "Phytosanitary Capacity": [0.00, 0.00, 0.00],
            "Presence": [True, False, True],
            "Infective": ["2010", None, "2014"],
            "Host Percent Area": [0.25, 0.50, 0.35],
        }
    )

    locations_list = [("USA", "CHN"), ("USA", "BRA"), ("BRA", "USA"), ("BRA", "CHN")]

    e = pandemic_single_time_step(
        trade=trade,
        distances=distances,
        locations=locations,
        locations_list=locations_list,
        climate_similarities=climate_similarities,
        alpha=0.2,
        beta=1,
        mu=0.0002,
        lamda_c=1,
        phi=5,
        sigma_h=0.5,
        sigma_kappa=0.5,
        w_phi=1,
        min_Tc=min_Tc,
        max_Tc=max_Tc,
        time_step=time_step,
        season_dict=season_dict,
        time_infect=time_infect,
        transmission_lag_type="static",
        time_infect_units="year",
        gamma_shape=None,
        gamma_scale=None,
        scenario_list=None,
    )
    assert (e[0] >= 0).all() and (e[0] <= 1).all()
    assert (e[1] >= 0).all() and (e[1] <= 1).all()

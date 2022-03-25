# PoPS Global - Network model of global pest introductions and spread over time.
# Copyright (C) 2019-2021 by the authors.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation; either version 2 of the License, or (at your option) any later
# version.

# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.

# You should have received a copy of the GNU General Public License along with
# this program; if not, see https://www.gnu.org/licenses/gpl-2.0.html

"""Creates a JSON containing a list of model configuration parameters for the
PoPS Global simulation.
"""

import json
import os

import os
import json


def create_config_args(
    config_out_path,
    commodity_list,
    commodity_path,
    native_countries_list,
    alpha,
    beta,
    mu,
    lamda_c_list,
    phi,
    w_phi,
    start_year,
    stop_year=None,
    save_main_output=True,
    save_metadata=True,
    save_entry=False,
    save_estab=False,
    save_intro=False,
    save_country_intros=False,
    commodity_forecast_path=None,
    season_dict=None,
    transmission_lag_type=None,
    time_to_infectivity=None,
    gamma_shape=None,
    gamma_scale=None,
    random_seed=None,
    cols_to_drop=None,
    lamda_weights_path=None,
    scenario_list=None,
):
    """
    Writes the configuration parameters to a JSON file.

    Parameters
    ----------
    config_out_path : str
        Path to directory for saving configuration JSON file.
    commodity_list : list (str)
        List of commodity codes for which the model will run.
    commodity_path : str
        Path to directory of trade data to use as model input.
    native_countries_list : list (str)
        List of countries where the pest is native or present at first time
        step of the model run.
    alpha : float
        A parameter that allows the equation to be adapated to various discrete
        time steps
    mu : float
        The mortality rate of the pest or pathogen during transport
    lamda_c_list : list (int)
        List of the commodity importance [0,1] of commodity (c) in transporting the
        pest or pathogen
    phi : int
        The degree of polyphagy of the pest of interest described as the number
        of host families
    w_phi : float
        The degree of polyphagy weight
    start_year : int
        Year of first time step of model run
    save_main_output : bool
        Indicates if main output should be saved as output
        (Default is True)
    save_metadata : bool
        Indicates if metadata should be saved as output
        (Default is True)
    save_entry : bool
        Indicates if probabilities of entry should be saved as output
        (Default is False)
    save_estab : bool
        Indicates if probabilities of establishment should be saved as output
        (Default is False)
    save_intro : bool
        Indicates if probabilities of introduction should be saved as output
        (Default is False)
    save_country_intros : bool
        Indicates if node introductions should be saved as output
        (Default is False)
    stop_year : int
        Year of last time step of model run
        (Default is None)
    commodity_forecast_path : str
        Path to directory of forecasted trade data to use as model input
        (Default is None)
    season_dict : dict (str)
        Dictionary of list of months when pest can be transported by hemisphere
        (Default is None)
    transmission_lag_type : str
        Indicates the type of transmission lag to use (e.g., None, static, stochastic)
        (Default is None)
    time_to_infectivity : int
        Number of years to delay country from becoming an origin when using static
        lag type. If lag type is none or stochastic, set to None.
        (Default is None)
    gamma_shape : int
        Shape parameter of gamma distribution when using stochastic lag type.
        If lag type is none or static, ste to None.
        (Default is None)
    gamma_scale : int
        Scale parameter of gamma distribution when using stochastic lag type.
        If lag type is none or static, ste to None.
        (Default is None)
    random_seed : int
        Random seed used to initialize the random number generator
        (Default is None)
    cols_to_drop : list (str)
        Columns to drop from model output dataframe
        (Default is None)
    lamda_weights_path : str
        Lambda weights (by country) to apply to trade when a commodity is not specific
        (Default is None)
    scenario_list : list (str)
        List of scenario model runs
        (Default is None)

    Returns
    -------
    args : dict
        Dictionary of all model arguments
    config_out_path : str
        Path to directory where model configuration JSON file was saved.

    """
    args = {}

    # Directory and file paths
    args["commodity_path"] = commodity_path
    args["commodity_list"] = commodity_list
    args["commodity_forecast_path"] = commodity_forecast_path
    # List of countries where pest is present at time T0
    args["native_countries_list"] = native_countries_list
    # List of months when pest can be transported
    args["season_dict"] = season_dict
    # pandemic parameter values
    args["alpha"] = alpha
    args["beta"] = beta
    args["mu"] = mu
    args["lamda_c_list"] = lamda_c_list
    args["lamda_weights_path"] = lamda_weights_path
    args["phi"] = phi
    args["w_phi"] = w_phi
    args["start_year"] = start_year
    args["stop_year"] = stop_year
    args["transmission_lag_unit"] = "year"
    # Transmission lag type can be static, stochastic or none
    args["transmission_lag_type"] = transmission_lag_type
    args["time_to_infectivity"] = time_to_infectivity  # only lag == static
    args["transmission_lag_shape"] = gamma_shape  # only lag == stochastic
    args["transmission_lag_scale"] = gamma_scale  # only lag == stochastic
    args["random_seed"] = random_seed
    args["save_main_output"] = save_main_output
    args["save_metadata"] = save_metadata
    args["save_entry"] = save_entry
    args["save_estab"] = save_estab
    args["save_intro"] = save_intro
    args["save_country_intros"] = save_country_intros
    args["columns_to_drop"] = cols_to_drop
    args["scenario_list"] = scenario_list

    # Write arguments to json file
    config_json_path = config_out_path
    if not os.path.exists(os.path.split(config_json_path)[0]):
        os.makedirs(os.path.split(config_json_path)[0])

    with open(config_json_path, "w") as file:
        json.dump(args, file, indent=4)

    print("\tSaved ", config_json_path)

    return args, config_out_path

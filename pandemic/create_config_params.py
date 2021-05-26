import json
import os


def create_config_args(
    config_out_path,
    commodity_path,
    native_countries_list,
    alpha,
    mu,
    lamda_c_list,
    phi,
    w_phi,
    start_year,
    save_main_output=True,
    save_metadata=True,
    save_entry=False,
    save_estab=False,
    save_intro=False,
    save_country_intros=False,
    stop_year=None,
    commodity_forecast_path=None,
    season_dict=None,
    transmission_lag_type=None,
    time_to_infectivity=None,
    gamma_shape=None,
    gamma_scale=None,
    random_seed=None,
    cols_to_drop=None,
    scenario_list=None,
):

    args = {}

    # Directory and file paths
    args["commodity_path"] = commodity_path
    args["commodity_forecast_path"] = commodity_forecast_path
    # List of countries where pest is present at time T0
    args["native_countries_list"] = native_countries_list
    # List of months when pest can be transported
    args["season_dict"] = season_dict
    # model parameter values
    args["alpha"] = alpha
    args["mu"] = mu
    args["lamda_c_list"] = lamda_c_list
    args["phi"] = phi
    args["w_phi"] = w_phi
    args["start_year"] = start_year
    args["stop_year"] = stop_year
    args["transmission_lag_unit"] = "year"
    # Transmission lag type can be static, stochastic or none
    args["transmission_lag_type"] = transmission_lag_type
    args["time_to_infectivity"] = time_to_infectivity  # only for lag type static
    args["transmission_lag_shape"] = gamma_shape  # only for lag type stochastic
    args["transmission_lag_scale"] = gamma_scale  # only for lag type stochastic
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

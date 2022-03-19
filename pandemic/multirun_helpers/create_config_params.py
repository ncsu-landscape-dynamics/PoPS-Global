import json


def create_global_config_args(
    model_files,
    quiet_time,
    project_loc,
    sim_name,
    start_run,
    end_run,
    commodity_list,
    trade_type,
    country_of_interest,
    native_countries_list,
    start_years,
    alphas,
    betas,
    lamdas,
    transmission_lag_type,
    gamma_shape,
    gamma_scale,
    threshold_val,
    scaled_min,
    scaled_max,
    season_dict,
    timestep,
    years_before_firstRecord,
    years_after_firstRecord,
    end_valid_year,
    sim_years,
    lamda_weights_path=None,
    scenario_list=None,
):

    args = {}

    # Directory and file paths
    args["model_files"] = model_files
    args["quiet_time"] = quiet_time
    args["project_loc"] = project_loc
    args["sim_name"] = sim_name
    args["start_run"] = start_run
    args["end_run"] = end_run
    args["commodity_list"] = commodity_list
    args["trade_type"] = trade_type
    args["country_of_interest"] = country_of_interest
    args["native_countries_list"] = native_countries_list
    args["start_years"] = start_years
    args["alphas"] = alphas
    args["betas"] = betas
    args["lamdas"] = lamdas
    args["lamda_weights_path"] = lamda_weights_path
    args["transmission_lag_type"] = transmission_lag_type
    args["gamma_shape"] = gamma_shape
    args["gamma_scale"] = gamma_scale
    args["threshold_val"] = threshold_val
    args["scaled_min"] = scaled_min
    args["scaled_max"] = scaled_max
    args["season_dict"] = season_dict
    args["timestep"] = timestep
    args["years_before_firstRecord"] = years_before_firstRecord
    args["years_after_firstRecord"] = years_after_firstRecord
    args["end_valid_year"] = end_valid_year
    args["sim_years"] = sim_years
    args["scenario_list"] = scenario_list

    # Write arguments to json file
    config_json_path = "config.json"
    with open(config_json_path, "w") as file:
        json.dump(args, file, indent=4)
    print("\tSaved ", config_json_path)

    return args, config_json_path

import sys
import os
import multiprocessing
from dotenv import load_dotenv
import json

from pandemic.multirun_helpers.multiple_runs import create_params, execute_model_runs
from pandemic.create_config_params import create_config_args

if __name__ == "__main__":
    alpha, beta, lamda_c_list, start_year, start_run, num_runs, run_type = [
        float(sys.argv[1]),
        float(sys.argv[2]),
        [float(sys.argv[3])],
        int(sys.argv[4]),
        int(sys.argv[5]),
        int(sys.argv[6]),
        sys.argv[7]
    ]

    with open("config.json") as json_file:
        config = json.load(json_file)

    try:
        model_files = config["model_files"]
    except: 
        model_files = "Keep"
        
    sim_name = config["sim_name"]
    native_countries_list = config["native_countries_list"]

    transmission_lag_type = config["transmission_lag_type"]
    gamma_shape = config["gamma_shape"]
    gamma_scale = config["gamma_scale"]
    threshold_val = config["threshold_val"]
    scaled_min = config["scaled_min"]
    scaled_max = config["scaled_max"]
    timestep = config["timestep"]
    season_dict = config["season_dict"]
    lamda_weights_path = config["lamda_weights_path"]
    commodity_list = config["commodity_list"]
    trade_type = config["trade_type"]

    commodity = "-".join(str(elem) for elem in commodity_list)

    load_dotenv(os.path.join(".env"))
    input_dir = os.getenv("INPUT_PATH")
    temp_dir = os.getenv("TEMP_OUTPATH")

    if model_files == "Temp":
        out_dir = f"{temp_dir}/samp{alpha}_{lamda_c_list[0]}_{start_year}"
    else:
        out_dir = f'{os.getenv("OUTPUT_PATH")}/{sim_name}_{run_type}'

    config_out_path = (
        rf"{out_dir}/config/"
        rf"year{start_year}_alpha{alpha}"
        rf"_beta{beta}"
        rf"_lamda{lamda_c_list[0]}"
        rf"_{commodity}/config.json"
    )

    if run_type == "calibrate":
        commodity_forecast_path = None
    else: 
        commodity_forecast_path=input_dir + f"/comtrade/trade_forecast/{timestep}_{trade_type}/",

    param_vals, config_file_path = create_config_args(
        config_out_path=config_out_path,
        commodity_list=commodity_list,
        commodity_path=input_dir + f"/comtrade/{timestep}_{trade_type}/",
        native_countries_list=native_countries_list,
        alpha=alpha,
        beta=beta,
        mu=0,
        lamda_c_list=lamda_c_list,
        phi=1,
        w_phi=1,
        start_year=start_year,
        stop_year=None,
        save_entry=False,
        save_estab=False,
        save_intro=False,
        save_country_intros=False,
        commodity_forecast_path=commodity_forecast_path,
        season_dict=season_dict,
        transmission_lag_type=transmission_lag_type,
        time_to_infectivity=None,
        gamma_shape=gamma_shape,
        gamma_scale=gamma_scale,
        random_seed=None,
        cols_to_drop=None,
        scenario_list=None,
        lamda_weights_path=lamda_weights_path,
    )

    param_list = create_params(
        model_script_path=("pandemic/model.py"),
        config_file_path=config_file_path,
        sim_name=sim_name,
        add_descript=(
            rf"year{param_vals['start_year']}_"
            rf"alpha{param_vals['alpha']}_"
            rf"beta{param_vals['beta']}_"
            rf"lamda{param_vals['lamda_c_list'][0]}"
        ),
        iteration_start=start_run,
        iteration_end=num_runs,
    )

    p = multiprocessing.Pool(4)  # set this number to the cores per node to use
    results = p.starmap(execute_model_runs, param_list)
    p.close()

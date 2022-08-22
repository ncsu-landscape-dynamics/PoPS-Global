import sys
import os
import multiprocessing
from dotenv import load_dotenv
import json

sys.path.append(os.getcwd())

from pandemic.multirun_helpers import create_params, execute_model_runs  # noqa: E402
from pandemic.create_config_params import create_config_args  # noqa: E402


if __name__ == "__main__":
    alpha, beta, lamda_c_list, start_year, start_run, end_run, run_type = [
        float(sys.argv[1]),
        float(sys.argv[2]),
        [float(sys.argv[3])],
        int(sys.argv[4]),
        int(sys.argv[5]),
        int(sys.argv[6]),
        str(sys.argv[7]),
    ]

    load_dotenv(os.path.join(".env"))
    input_dir = os.getenv("INPUT_PATH")
    temp_dir = os.getenv("TEMP_OUTPATH")
    out_dir = os.getenv("OUTPUT_PATH")
    sim_name = os.getenv("SIM_NAME")

    config_json_path = f"{out_dir}/config_{sim_name}.json"

    with open(config_json_path) as json_file:
        config = json.load(json_file)
    try:
        model_files = config["model_files"]
    except KeyError:
        model_files = "Keep"
    native_countries_list = config["native_countries_list"]

    transmission_lag_type = config["transmission_lag_type"]
    gamma_shape = config["gamma_shape"]
    gamma_scale = config["gamma_scale"]
    mask = config["mask"]
    threshold_val = config["threshold_val"]
    scaled_min = config["scaled_min"]
    scaled_max = config["scaled_max"]
    timestep = config["timestep"]
    season_dict = config["season_dict"]
    lamda_weights_path = config["lamda_weights_path"]
    commodity_list = config["commodity_list"]
    trade_type = config["trade_type"]

    cores_to_use = config["cores_to_use"]

    commodity = "-".join(str(elem) for elem in commodity_list)

    if model_files == "Temp":
        out_path = f"{temp_dir}/samp{alpha}_{lamda_c_list[0]}_{start_year}"
    else:
        out_path = f"{out_dir}/{sim_name}_{run_type}"
    config_out_path = (
        rf"{out_path}/config/"
        rf"year{start_year}_alpha{alpha}"
        rf"_beta{beta}"
        rf"_lamda{lamda_c_list[0]}"
        rf"_{commodity}/config.json"
    )

    if run_type == "calibrate":
        commodity_forecast_path = None
    else:
        commodity_forecast_path = (
            input_dir + f"/comtrade/trade_forecast/{timestep}_{trade_type}/",
        )
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
        mask=mask,
        threshold=threshold_val,
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
        iteration_end=end_run,
        run_type=run_type,
    )

    p = multiprocessing.Pool(
        cores_to_use
    )  # set this number to the cores per node to use
    results = p.starmap(execute_model_runs, param_list)
    p.close()

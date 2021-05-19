import sys
import os
import itertools
import numpy as np
# from dotenv import load_dotenv
import multiprocessing
import subprocess

sys.path.append('C:/Users/cawalden/Documents/GitHub/Pandemic_Model')

from pandemic.create_config_params import create_config_args

def create_params(
    model_script_path,
    config_file_path,
    sim_name,
    add_descript,
    iteration_start,
    iteration_end
):
    """
    Creates list of parameter values for running the model
    which name the outputs based on the simulation name
    iteration of that run

    Parameters
    -----------
    sim_name : str
        Name of the simulation run (e.g., 'time_lag')
    add_descript : str
        Additional description describing focus of model run
        or parameter value with suffix of commodity code or
        range of commodity codes (e.g., 3yrs_6801, 3yrs_6801-6804)
    iteration_start : int
        Integer denoting the starting stochastic run number (e.g., 1)
    iteration_end : int
        Integer denoting the last stochastic run number (e.g., 10).

    """

    model_script_path = np.repeat(
        model_script_path, ((iteration_end + 1) - iteration_start)
    )
    config_file_path = np.repeat(
        config_file_path, ((iteration_end + 1) - iteration_start)
    )
    sim_name = np.repeat(sim_name, ((iteration_end + 1) - iteration_start))
    add_descript = np.repeat(
        add_descript,
        ((iteration_end + 1) - iteration_start)
    )
    run_num = range(iteration_start, iteration_end + 1, 1)
    param_list = list(zip(
        model_script_path, config_file_path, sim_name, add_descript, run_num)
    )
    return param_list


def execute_model_runs(
    model_script_path,
    config_file_path,
    sim_name,
    add_descript,
    run_num
):

    """
    Executes a run of the model based on the model script
    location and configuration file location

    Parameters
    -----------
    sim_name : str
        Name of the simulation run (e.g., 'time_lag')
    add_descript : str
        Additional description describing focus of model run
        or parameter value (e.g., 3yrs)
    run_num : int
        Stochastic iteration of model run (e.g., 0)
    """
    print(f"Simulation: {sim_name}\t{add_descript}\trun: {run_num}")

    subprocess.call(
        [
            "python",
            str(model_script_path),
            str(config_file_path),
            str(sim_name),
            str(add_descript),
            str(run_num),
        ],
        shell=True,
    )
    return model_script_path, config_file_path, sim_name, add_descript, run_num

def create_scenario_list(
    scenario_list,
    origin, 
    destination, 
    start_scenario_ts, 
    end_scenario_ts, 
    adjustment_type, 
    percent_change
):
    for i in range(start_scenario_ts, end_scenario_ts + 1):
        start_scenario = (
            [start_scenario_ts, str(origin), str(destination), adjustment_type, float(percent_change)]
        )
        new_scenario = start_scenario
        new_scenario[0] = i
        scenario_list .append(new_scenario)
    return scenario_list


if __name__ == '__main__':
    # env_file_path = "C:/Users/cawalden/Documents/Projects/Pandemic"
    # load_dotenv(os.path.join(env_file_path, '.env'))
    # input_dir = os.getenv('INPUT_PATH')
    # out_dir = os.getenv('OUTPUT_PATH')

    input_dir = "H:/Shared drives/Pandemic Data/slf_model/inputs/noTWN/"
    out_dir = "H:/Shared drives/SLF Paper Outputs/outputs/"

    scenario1 = []
    scenario1 = create_scenario_list(scenario1, "CHN", "USA", 2014, 2029, "decrease", 1)
    scenario1 = create_scenario_list(scenario1, "VNM", "USA", 2014, 2029, "decrease", 1)
    
    scenario2 = []
    scenario2 = create_scenario_list(scenario2, "CHN", "USA", 2014, 2029, "decrease", 1)
    scenario2 = create_scenario_list(scenario2, "VNM", "USA", 2014, 2029, "decrease", 1)
    scenario2 = create_scenario_list(scenario2, "JPN", "USA", 2014, 2029, "decrease", 1)
    scenario2 = create_scenario_list(scenario2, "KOR", "USA", 2014, 2029, "decrease", 1)
    scenario2 = create_scenario_list(scenario2, "TUR", "USA", 2014, 2029, "decrease", 1)
    scenario2 = create_scenario_list(scenario2, "ITA", "USA", 2015, 2029, "decrease", 1)


    scenarios = [scenario1] + [scenario2]
    scenario_names = ['stopOrigins', 'stopBridgeheads']
    # scenarios = [scenario1]

    for i in range(0, len(scenarios)):
        scenario_list = scenarios[i]
        scenario_name = scenario_names[i]
        alpha = 0.2
        transmission_lag_type = "stochastic"
        gamma_shape = 4  
        gamma_scale = 1  
        lamda_c_list = [3.9]

        threshold_val = 16
        scaled_min = 0.3
        scaled_max = 0.8

        config_out_path = (
            rf"H:/Shared drives/Pandemic Data/slf_model/"
            rf"inputs/config_files/slf"
            rf"_scenarios_noTWN"
            rf"_6801-6804/config.json"
        )

        param_vals, config_file_path = create_config_args(
            config_out_path=config_out_path,
            commodity_path=input_dir + "/comtrade/monthly_agg/6801-6804",
            native_countries_list=["China", "Viet Nam"],
            alpha=alpha,
            mu=0,
            lamda_c_list=lamda_c_list,
            phi=1,
            w_phi=1,
            start_year=2006,
            stop_year=None,
            save_entry=False,
            save_estab=False,
            save_intro=False,
            save_country_intros=False,
            commodity_forecast_path=(
                input_dir + "/comtrade/trade_forecast/monthly_agg/6801-6804"
            ),
            season_dict={
                "NH_season": ["09", "10", "11", "12", "01", "02", "03", "04"],
                "SH_season": ["04", "05", "06", "07", "08", "09", "10"],
            },
            transmission_lag_type=transmission_lag_type,
            time_to_infectivity=None,
            gamma_shape=gamma_shape,
            gamma_scale=gamma_scale,
            random_seed=None,
            cols_to_drop=None,
            scenario_list=scenario_list,
        )

        param_list = create_params(
            model_script_path=(
                    "pandemic/model.py"
            ),
            config_file_path=config_file_path,
            sim_name=(
                    rf"slf_scenarios_noTWN_wChinaVietnam/{scenario_name}"
                ),
            add_descript=(
                    rf"alpha{param_vals['alpha']}_"
                    rf"lamda{param_vals['lamda_c_list'][0]}"
                ),
            iteration_start=0,
            iteration_end=999
        )

        p = multiprocessing.Pool(60)
        results = p.starmap(execute_model_runs, param_list)
        p.close()
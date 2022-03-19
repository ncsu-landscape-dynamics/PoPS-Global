import numpy as np
import subprocess


def create_params(
    model_script_path,
    config_file_path,
    sim_name,
    add_descript,
    iteration_start,
    iteration_end,
):
    """
    Creates list of parameter values for running the pandemic
    which name the outputs based on the simulation name
    iteration of that run

    Parameters
    -----------
    sim_name : str
        Name of the simulation run (e.g., 'time_lag')
    add_descript : str
        Additional description describing focus of pandemic run
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
        add_descript, ((iteration_end + 1) - iteration_start)
        )
    run_num = range(iteration_start, iteration_end + 1, 1)
    param_list = list(
        zip(
            model_script_path,
            config_file_path,
            sim_name,
            add_descript,
            run_num,
            )
    )
    return param_list


def execute_model_runs(
    model_script_path, config_file_path, sim_name, add_descript, run_num
):

    """
    Executes a run of the pandemic based on the pandemic script
    location and configuration file location

    Parameters
    -----------
    sim_name : str
        Name of the simulation run (e.g., 'time_lag')
    add_descript : str
        Additional description describing focus of pandemic run
        or parameter value (e.g., 3yrs)
    run_num : int
        Stochastic iteration of pandemic run (e.g., 0)
    """
    print(f"Simulation: {sim_name}\t{add_descript}\trun: {run_num}")

    subprocess.run(
        [
            "python",
            str(model_script_path),
            str(config_file_path),
            str(sim_name),
            str(add_descript),
            str(run_num),
        ],
        shell=False,
    )
    return model_script_path, config_file_path, sim_name, add_descript, run_num

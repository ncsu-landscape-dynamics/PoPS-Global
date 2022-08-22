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


"""Module containing all for helper functions running multiple PoPS
Global simulations.
"""

import itertools
import json
import numpy as np
import subprocess

import glob
import pandas as pd
import itertools
import json
import re


import os
import math
from dotenv import load_dotenv


def write_commands(params, start_run, end_run, run_type, model_files="Keep"):
    """
    Writes a command to run the multiple runs of the model the command line
    with parameters as arguments to a script.

    Parameters
    -----------
    params : list
        A list with the key model parameters, ordered: [alpha, beta, lamba, start year]
    start_run : int
        Integer denoting the starting stochastic run number(e.g., 0).
    end_run : int
        Integer denoting the ending stochastic run number (e.g., 30)
    run_type : str
        The type of run being conducted. Options are "calibrate" or "forecast".
    model_files : str
        Default is "Keep". If you are running on HPC and do not wish to save model outputs
        (only summary stats), you can use "Temp".

    """
    # Name the script to be run
    if model_files == "Temp":
        script = "./hpc/wrapper_script.csh"
    else:
        script = "python pandemic/model_run_args.py"
        # script = "python model_run_args.py"
    output = (
        " ".join(
            [
                script,
                str(params[0]),  # alpha
                str(params[1]),  # beta
                str(params[2]),  # lamda
                str(params[3]),  # start year
                str(start_run),
                str(end_run),
                run_type,
            ]
        )
        + "\n"
    )
    return output


def create_params(
    model_script_path,
    config_file_path,
    sim_name,
    add_descript,
    iteration_start,
    iteration_end,
    run_type,
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
    add_descript = np.repeat(add_descript, ((iteration_end + 1) - iteration_start))
    run_num = range(iteration_start, iteration_end + 1, 1)
    run_type = np.repeat(run_type, ((iteration_end + 1) - iteration_start))

    param_list = list(
        zip(
            model_script_path,
            config_file_path,
            sim_name,
            add_descript,
            run_num,
            run_type,
        )
    )
    return param_list


def execute_model_runs(
    model_script_path, config_file_path, sim_name, add_descript, run_num, run_type
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
    run_type: str
        Type of runs ("test", "calibrate", or "forecast")
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
            str(run_type),
        ],
        shell=False,
    )
    return (
        model_script_path,
        config_file_path,
        sim_name,
        add_descript,
        run_num,
        run_type,
    )


def complete_run_check(param_sample):
    """
    Generates a list of completed model runs by checking for their outputs. Used
    to check for run completeness when calibrating.

    Parameters
    -----------
    param_sample : list

    """

    # Empty dataframe for completed runs
    completed_runs = pd.DataFrame(
        {"start": [0], "alpha": [0], "beta": [0], "lamda": [0], "run": [0]}
    )

    # Which runs were completed
    for param in param_sample:
        sample = re.split("[\\\\/]", param)[-1]
        # "\\" to run locally, "/" on HPC or with multiprocess
        start = sample.split("year")[1].split("_")[0]
        alpha = sample.split("alpha")[1].split("_")[0]
        beta = sample.split("beta")[1].split("_")[0]
        lamda = sample.split("lamda")[1].split("_")[0]
        run_outputs = glob.glob(f"{param}/run*/origin_destination.csv")
        runs = []
        for output in run_outputs:
            indiv = re.split("[\\\\/]", output.split("run_")[1])[0]
            # "\\" to run locally, "/" on HPC or with multiprocess
            runs.append(int(indiv))
        for run in runs:
            completed_runs = completed_runs.append(
                pd.Series(
                    {
                        "start": start,
                        "alpha": alpha,
                        "beta": beta,
                        "lamda": lamda,
                        "run": run,
                    }
                ),
                ignore_index=True,
            )
    # Write it to a .csv for safe keeping
    completed_runs.drop(index=0, inplace=True)
    completed_runs.to_csv("completed_runs.csv")
    return completed_runs


# Extract the missingness to a list
def pending_run_check(completed_runs, param_sets, full_set):
    results = []
    for param_set in param_sets:
        alpha, beta, lamda, start = param_set
        complete_runs = set(
            completed_runs.loc[
                (completed_runs["start"] == start)
                & (completed_runs["lamda"] == lamda)
                & (completed_runs["alpha"] == alpha)
                & (completed_runs["beta"] == beta)
            ]["run"]
        )
        missing_runs = full_set - complete_runs
        if len(missing_runs) > 0:
            results.append([param_set, missing_runs])
    return results


def run_checker(param_sample):
    # Global parameters from config
    load_dotenv(os.path.join(".env"))
    out_dir = os.getenv("OUTPUT_PATH")
    sim_name = os.getenv("SIM_NAME")

    config_json_path = f"{out_dir}/config_{sim_name}.json"

    with open(config_json_path) as json_file:
        config = json.load(json_file)
    alphas = config["alphas"]
    betas = config["betas"]
    lamdas = config["lamdas"]
    start_years = config["start_years"]
    start_run = config["start_run"]
    end_run = config["end_run"]
    model_files = config["model_files"]

    # Recreate the full parameter set used for the runs
    param_list = [alphas, betas, lamdas, start_years]
    param_sets = list(itertools.product(*param_list))

    # Two methods to check runs:
    # 1. from folders (model_files == "Keep")
    if model_files == "Keep":
        # Create the range of runs expected, as a set
        full_set = set(range(start_run, end_run + 1))

        complete_run_check(param_sample)
        completed_runs = pd.read_csv("completed_runs.csv")
        pending_runs = pending_run_check(completed_runs, param_sets, full_set)
    # 2. from summary stats (model_files = "Temp")
    if model_files == "Temp":
        pending_runs = []
        for param_set in param_sets:
            alpha, beta, lamda, start = param_set
            completed = param_sample.loc[
                (param_sample["start_max"] == start)
                & (param_sample["lamda_max"] == lamda)
                & (param_sample["alpha_max"] == alpha)
                & (param_sample["beta_max"] == beta)
            ]
            if len(completed.index) == 0:
                pending_runs.append([param_set, set([start_run, end_run])])
    return pending_runs


def diff_metric(difference_val, threshold_val):
    if (difference_val <= threshold_val) and (difference_val > 0):
        return round(1 - (abs(difference_val) / (threshold_val + 1)), 2)
    if difference_val == 0:
        return 1
    else:
        return 0


def calc_overall_prob(probs):
    overall_prob = 1 - np.prod([1 - i for i in probs])
    return overall_prob


def compute_summary_stats(
    model_output,
    validation_df,
    years_before_firstRecord,
    years_after_firstRecord,
    end_valid_year,
    native_countries_list,
    year_probs_dict_keys,
    sim_years,
    coi,
):
    presence_cols = [
        c
        for c in model_output.columns
        if c.startswith("Presence") and len(c.split(" ")[-1]) == 4
    ]
    # First introduction year predicted
    model_output["PredFirstIntro"] = np.where(
        model_output[presence_cols].any(axis=1),
        model_output[presence_cols].idxmax(axis=1),
        "Presence 9999",
    )
    model_output["PredFirstIntro"] = (
        model_output["PredFirstIntro"].str.replace("Presence ", "")
    ).astype(int)

    model_output.set_index("ISO3", inplace=True)

    # Merge with validation data (ISO3 - First Record Year)
    model_output = model_output.merge(
        validation_df, how="left", left_index=True, right_index=True
    )
    # Difference in years between prediction and obesrvation
    model_output["pred_diff"] = np.where(
        model_output["PredFirstIntro"] != 9999,
        (model_output["ObsFirstIntro"] - model_output["PredFirstIntro"]),
        np.nan,
    )

    # Does the prediction fall within the identified time window?
    # Binary output
    model_output["temp_acc"] = model_output["PredFirstIntro"].between(
        model_output["ObsFirstIntro"] - years_before_firstRecord,
        model_output["ObsFirstIntro"] + years_after_firstRecord,
    )

    # Calculate temporal accuracy metric
    model_output["obs-pred_metric"] = model_output.apply(
        lambda x: diff_metric(x["pred_diff"], 5), axis=1
    )

    # Total introductions predicted
    total_intros_predicted = model_output[
        f"Presence {str(end_valid_year)}"
    ].sum() - len(native_countries_list)

    total_intros_diff = validation_df.shape[0] - total_intros_predicted
    total_intros_diff_sqrd = total_intros_diff**2

    # Compute prob of introduction happening at least once in
    # country of interest for each year in simulation
    year_probs_dict_values = []
    for year in sim_years:
        prob_cols = [
            c
            for c in model_output.columns
            if c.startswith("Agg Prob") and int(c.split(" ")[-1]) <= year
        ]
        model_output[f"prob_by_{year}_{coi}"] = model_output.apply(
            lambda x: calc_overall_prob(x[prob_cols]), axis=1
        )
        year_probs_dict_values.append(
            model_output.loc[f"{coi}"][f"prob_by_{year}_{coi}"]
        )
    year_probs_dict = dict(zip(year_probs_dict_keys, year_probs_dict_values))

    countries_dict = {}
    for ISO3 in validation_df.index:
        countries_dict[f"diff_obs_pred_metric_{ISO3}"] = model_output.loc[ISO3][
            "obs-pred_metric"
        ]
    # Save results in dictionary from which to build the dataframe
    summary_stats_dict = {
        "total_countries_intros_predicted": total_intros_predicted,
        "diff_total_countries": total_intros_diff,
        "diff_total_countries_sqrd": total_intros_diff_sqrd,
        "count_known_countries_predicted": (
            model_output.loc[validation_df.index]["PredFirstIntro"] != 9999
        ).sum(),
        "count_known_countries_time_window": (
            model_output.loc[validation_df.index]["temp_acc"].sum()
        ),
        "diff_obs_pred_metric_mean": model_output.loc[validation_df.index][
            "obs-pred_metric"
        ].mean(),
        "diff_obs_pred_metric_stdev": model_output.loc[validation_df.index][
            "obs-pred_metric"
        ].std(),
    }

    summary_stats_dict = {**summary_stats_dict, **year_probs_dict, **countries_dict}

    return model_output, summary_stats_dict


def compute_stat_wrapper_func(param_sample):
    load_dotenv(os.path.join(".env"))
    input_dir = os.getenv("INPUT_PATH")
    out_dir = os.getenv("OUTPUT_PATH")
    sim_name = os.getenv("SIM_NAME")

    config_json_path = f"{out_dir}/config_{sim_name}.json"

    with open(config_json_path) as json_file:
        config = json.load(json_file)
    coi = config["coi"]
    native_countries_list = config["native_countries_list"]
    years_before_firstRecord = config["years_before_firstRecord"]
    years_after_firstRecord = config["years_after_firstRecord"]
    end_valid_year = config["end_valid_year"]
    sim_years = config["sim_years"]

    validation_df = pd.read_csv(
        input_dir + "/first_records_validation.csv",
        header=0,
        index_col=0,
    )
    run_outputs = glob.glob(f"{param_sample}/run*/pandemic_output_aggregated.csv")

    # Set up probability by year dictionary keys (column names)
    year_probs_dict_keys = []
    for year in sim_years:
        year_probs_dict_keys.append(f"prob_by_{year}_{coi}")
    # Set up difference by recorded country dictionary keys (column names)
    countries_dict_keys = []
    for ISO3 in validation_df.index:
        countries_dict_keys.append(f"diff_obs_pred_metric_{ISO3}")
    summary_stat_df = pd.DataFrame(
        columns=[
            "sample",
            "run_num",
            "start",
            "alpha",
            "beta",
            "lamda",
            "total_countries_intros_predicted",
            "diff_total_countries",
            "diff_total_countries_sqrd",
            "count_known_countries_predicted",
            "count_known_countries_time_window",
            "diff_obs_pred_metric_mean",
            "diff_obs_pred_metric_stdev",
        ]
        + year_probs_dict_keys
        + countries_dict_keys
    )
    for i in range(0, len(run_outputs)):
        run_num = os.path.split(run_outputs[i])[0].split("run_")[-1]
        # "\\" to run locally, "/" on HPC
        sample = re.split("[\\\\/]", run_outputs[i])[-3]
        start = sample.split("year")[1].split("_")[0]
        alpha = sample.split("alpha")[1].split("_")[0]
        beta = sample.split("beta")[1].split("_")[0]
        lamda = sample.split("lamda")[1].split("_")[0]
        df = pd.read_csv(
            run_outputs[i], sep=",", header=0, index_col=0, encoding="latin1"
        )
        # df.set_index("ISO3", inplace=True)
        _, summary_stat_dict = compute_summary_stats(
            df,
            validation_df,
            years_before_firstRecord,
            years_after_firstRecord,
            end_valid_year,
            native_countries_list,
            year_probs_dict_keys,
            sim_years,
            coi,
        )
        summary_stat_dict["run_num"] = run_num
        summary_stat_dict["sample"] = param_sample
        summary_stat_dict["start"] = start
        summary_stat_dict["alpha"] = alpha
        summary_stat_dict["beta"] = beta
        summary_stat_dict["lamda"] = lamda
        summary_stat_df = summary_stat_df.append(summary_stat_dict, ignore_index=True)
    # summary_stat_df = pd.DataFrame(summary_stat_dict, index=[0])
    return summary_stat_df


def mse(x):
    """
    Computes the mean when aggregating across runs of a parameter sample.
    """
    return sum(x) / len(x)


def avg_std(x):
    """
    Computes average standard deviation when aggregating across runs
    of a parameter sample.
    """
    return math.sqrt(sum(x**2) / len(x))


def mape(x):
    return (1 / len(x)) * sum(abs(x / 3))


def fbeta(precision, recall, weight):
    """
    Computes the weighted harmonic mean of precision and recall (F-beta score).
    """
    if (precision != 0) and (recall != 0):
        return ((1 + (weight**2)) * precision * recall) / (
            (weight**2) * precision + recall
        )
    else:
        return 0


def f1(precision, recall):
    """
    Computes the harmonic mean of precision and recall (F1-score).
    """
    if (precision != 0) and (recall != 0):
        return (2 * precision * recall) / (precision + recall)
    else:
        return 0


# Forecast: Generating sampled parameter sets


def generate_param_samples(agg_df, n_samples):
    """
    Generates a number of parameter sets sampled from a multivariate normal distribution
    fit to the top performing samples of the calibration model runs.

    Parameters
    -----------
    agg_df : pandas dataframe
        A dataframe of summary statistics returned from the model, including the following
        named columns: "alpha" (model parameter), "beta" (model parameter), "lamba" (model parameter),
        "start" (model parameter), "top" (flag for samples above a pre-defined summary statistic threshold)/
    n_samples : int
        The number of sampled parameter sets to generate.

    Returns
    -------
    samples_to_run : pandas dataframe
        A pandas dataframe of parameter sets sampled from the multivariate normal distribution
        of the top performing parameter samples from calibration.

    """
    param_samples_df = pd.DataFrame(columns=["alpha", "beta", "lamda", "start"])

    top_sets = agg_df.loc[(agg_df["top"] == "top")][
        ["start", "alpha", "beta", "lamda"]
    ].reset_index(drop=True)
    start_years = top_sets.start.unique()
    top_count = len(top_sets.index)

    year_counts = []
    set_counts = [0]

    for year in start_years:
        year_sets = top_sets.loc[top_sets["start"] == year].reset_index(drop=True)
        year_counts.append(math.ceil(len(year_sets.index) * n_samples / top_count))

        param_mean = np.mean(top_sets[["alpha", "beta", "lamda"]].values, axis=0)
        param_cov = np.cov(top_sets[["alpha", "beta", "lamda"]].values, rowvar=0)
        param_sample = np.random.multivariate_normal(
            param_mean, param_cov, int(n_samples * 1.5)
        )
        alpha = param_sample[:, 0]
        beta = param_sample[:, 1]
        lamda = param_sample[:, 2]
        start = [year] * int(n_samples * 1.5)
        param_sample_df = pd.DataFrame(
            {"alpha": alpha, "lamda": lamda, "beta": beta, "start": start}
        )
        param_sample_df = param_sample_df.loc[
            param_sample_df["alpha"] <= 1
        ].reset_index(drop=True)
        set_counts.append(set_counts[-1] + len(param_sample_df.index))

        param_samples_df = pd.concat([param_samples_df, param_sample_df]).reset_index(
            drop=True
        )

        print(
            f"Year: {year}, Count: {year_counts[-1]},\n"
            f"Means: {param_mean},\n"
            f"Covariance Matrix: {param_cov}\n"
        )
    samp_runs = [
        item
        for sublist in [
            list(range(set_counts[i], set_counts[i] + year_count))
            for i, year_count in enumerate(year_counts)
        ]
        for item in sublist
    ]

    samples_to_run = param_samples_df.loc[samp_runs].reset_index(drop=True)

    return samples_to_run

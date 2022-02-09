import os
import glob
import math
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import json
import re


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
        model_output[presence_cols].any(axis=1) == True,
        model_output[presence_cols].idxmax(axis=1),
        "Presence 9999",
    )
    model_output["PredFirstIntro"] = (
        model_output["PredFirstIntro"].str.replace("Presence ", "")
    ).astype(int)

    # Merge with validation data (ISO3 - First Record Year)
    model_output = model_output.merge(validation_df, how="left", on="ISO3")

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
    total_intros_diff_sqrd = total_intros_diff ** 2

    # Compute prob of introduction happening at least once in country of interest
    # for each year in simulation
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
        "count_known_countries_time_window": model_output.loc[validation_df.index][
            "temp_acc"
        ].sum(),
        "diff_obs_pred_metric_mean": model_output.loc[validation_df.index][
            "obs-pred_metric"
        ].mean(),
        "diff_obs_pred_metric_stdev": model_output.loc[validation_df.index][
            "obs-pred_metric"
        ].std(),
    }

    summary_stats_dict = {**summary_stats_dict, **year_probs_dict, **countries_dict}

    return model_output, summary_stats_dict


def compute_brier_score(
    run_outputs, validation_df, native_countries_list, years_before_firstRecord
):

    model_output = pd.read_csv(run_outputs[0])

    presence_cols = [
        c
        for c in model_output.columns
        if c.startswith("Presence") and len(c.split(" ")[-1]) == 4
    ]

    years = [c.split(" ")[-1] for c in presence_cols]

    # Remove native countries
    model_output.drop(
        model_output.loc[model_output["NAME"].isin(native_countries_list)].index,
        inplace=True,
    )
    validation = model_output.merge(validation_df, how="left", on="ISO3")[
        ["ISO3", "ObsFirstIntro"]
    ]

    for year in years:
        validation[year] = 0
        validation.loc[validation["ObsFirstIntro"] <= int(year), year] = 1

    validation_w_lag = validation.copy()
    for year in years:
        validation_w_lag.loc[
            validation_w_lag["ObsFirstIntro"] <= int(year) + years_before_firstRecord,
            year,
        ] = 1

    validation.drop(columns=["ISO3", "ObsFirstIntro"], inplace=True)
    validation_w_lag.drop(columns=["ISO3", "ObsFirstIntro"], inplace=True)

    total_intros = model_output[presence_cols].values

    for run in run_outputs:
        model_output = pd.read_csv(run)
        model_output.drop(
            model_output.loc[model_output["NAME"].isin(native_countries_list)].index,
            inplace=True,
        )
        total_intros = np.dstack((total_intros, model_output[presence_cols].values))

    mean_intros = np.mean(total_intros, axis=2)

    # For each value that is in the window period, pick the score that does better (presence or absence)
    brier_scores = np.minimum(
        (mean_intros - validation.values) ** 2,
        (mean_intros - validation_w_lag.values) ** 2,
    )

    brier_score = np.mean(brier_scores)

    return brier_score


def compute_stat_wrapper_func(param_sample):
    load_dotenv(os.path.join(".env"))
    input_dir = os.getenv("INPUT_PATH")

    with open("config.json") as json_file:
        config = json.load(json_file)
    coi = config["country_of_interest"]
    native_countries_list = config["native_countries_list"]
    years_before_firstRecord = config["years_before_firstRecord"]
    years_after_firstRecord = config["years_after_firstRecord"]
    end_valid_year = config["end_valid_year"]
    sim_years = config["sim_years"]

    validation_df = pd.read_csv(
        input_dir + "/first_records_validation.csv", header=0, index_col=0,
    )
    run_outputs = glob.glob(f"{param_sample}/run*/pandemic_output_aggregated.csv")

    brier_score = compute_brier_score(
        run_outputs, validation_df, native_countries_list, years_before_firstRecord
    )

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
            "lamda",
            "samp_brier_score",
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
        sample = re.split("[\\\\/]", run_outputs[i])[
            -3
        ]  # "\\" to run locally, "/" on HPC
        start = sample.split("year")[1].split("_")[0]
        alpha = sample.split("alpha")[1].split("_")[0]
        lamda = sample.split("lamda")[1].split("_")[0]
        df = pd.read_csv(
            run_outputs[i], sep=",", header=0, index_col=0, encoding="latin1"
        )
        df.set_index("ISO3", inplace=True)
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
        summary_stat_dict["lamda"] = lamda
        summary_stat_dict["samp_brier_score"] = brier_score
        summary_stat_df = summary_stat_df.append(summary_stat_dict, ignore_index=True)
    # summary_stat_df = pd.DataFrame(summary_stat_dict, index=[0])
    return summary_stat_df


def mse(x):
    return sum(x) / len(x)


def avg_std(x):
    """
    Compute average standard deviation when aggregating across runs
    of a parameter sample
    """
    return math.sqrt(sum(x ** 2) / len(x))


def mape(x):
    return (1 / len(x)) * sum(abs(x / 3))


def fbeta(precision, recall, weight):
    if (precision != 0) and (recall != 0):
        return ((1 + (weight ** 2)) * precision * recall) / (
            (weight ** 2) * precision + recall
        )
    else:
        return 0


def f1(precision, recall):
    if (precision != 0) and (recall != 0):
        return (2 * precision * recall) / (precision + recall)
    else:
        return 0

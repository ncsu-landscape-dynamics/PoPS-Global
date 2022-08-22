if __name__ == "__main__":
    import sys
    import os
    import glob
    import pandas as pd
    import multiprocessing
    from dotenv import load_dotenv
    import json

    sys.path.append(os.getcwd())

    from pandemic.multirun_helpers import (
        compute_stat_wrapper_func,
        mse,
        f1,
        fbeta,
        avg_std,
    )

    run_type = sys.argv[1]  # Argument to get: calibrate or forecast

    # Read environmental variables
    env_file = os.path.join(".env")
    load_dotenv(env_file)

    # Path to formatted model inputs
    input_dir = os.getenv("INPUT_PATH")
    out_dir = os.getenv("OUTPUT_PATH")
    sim_name = os.getenv("SIM_NAME")

    config_json_path = f"{out_dir}/config_{sim_name}.json"

    with open(config_json_path) as json_file:
        config = json.load(json_file)
    run_name = f"{sim_name}_{run_type}"
    commodity = "-".join(str(elem) for elem in config["commodity_list"])

    coi = config["coi"]
    native_countries_list = config["native_countries_list"]

    cores_to_use = config["cores_to_use"]

    try:
        model_files = config["model_files"]
    except KeyError:
        model_files = "Keep"
    if model_files == "Temp":
        out_dir = (
            f'{os.getenv("TEMP_OUTPATH")}/samp{sys.argv[2]}_{sys.argv[3]}_{sys.argv[4]}'
        )
    else:
        out_dir = os.getenv("OUTPUT_PATH")
    param_samp = glob.glob(f"{out_dir}/{run_name}/*{commodity}*")

    validation_df = pd.read_csv(
        input_dir + "/first_records_validation.csv",
        header=0,
        index_col=0,
    )

    # Set up probability by year dictionary keys (column names)
    sim_years = config["sim_years"]
    year_probs_dict_keys = []
    for year in sim_years:
        year_probs_dict_keys.append(f"prob_by_{year}_{coi}")
    # Set up difference by recorded country dictionary keys (column names)
    countries_dict_keys = []
    for ISO3 in validation_df.index:
        countries_dict_keys.append(f"diff_obs_pred_metric_{ISO3}")
    process_pool = multiprocessing.Pool(cores_to_use)
    summary_dfs = process_pool.map(compute_stat_wrapper_func, param_samp)
    data = pd.concat(summary_dfs, ignore_index=True)
    data = data[
        [
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
    ]

    data["total_countries_intros_predicted"] = data[
        "total_countries_intros_predicted"
    ].astype(int)
    data["diff_total_countries"] = data["diff_total_countries"].astype(int)
    data["diff_total_countries_sqrd"] = data["diff_total_countries_sqrd"].astype(float)
    data["count_known_countries_predicted"] = data[
        "count_known_countries_predicted"
    ].astype(int)
    data["count_known_countries_time_window"] = data[
        "count_known_countries_time_window"
    ].astype(int)

    for ISO3 in validation_df.index:
        data[f"diff_obs_pred_metric_{ISO3}"] = data[
            f"diff_obs_pred_metric_{ISO3}"
        ].astype(float)
    # TP / (TP + FN)
    data["count_known_countries_time_window_recall"] = data[
        "count_known_countries_time_window"
    ] / (
        data["count_known_countries_time_window"]
        + (validation_df.shape[0] - data["count_known_countries_time_window"])
    )

    # TP / (TP + FP)
    data["count_known_countries_time_window_precision"] = data[
        "count_known_countries_time_window"
    ] / (
        data["count_known_countries_time_window"]
        + (
            data["total_countries_intros_predicted"]
            - data["count_known_countries_time_window"]
        )
    )

    # 2 * (precision * recall / precision + recall)
    data["count_known_countries_time_window_f1"] = data.apply(
        lambda x: f1(
            x["count_known_countries_time_window_precision"],
            x["count_known_countries_time_window_recall"],
        ),
        axis=1,
    )

    data["count_known_countries_time_window_fbeta"] = data.apply(
        lambda x: fbeta(
            x["count_known_countries_time_window_precision"],
            x["count_known_countries_time_window_recall"],
            2,
        ),
        axis=1,
    )

    summary_stat_path = f'{os.getenv("OUTPUT_PATH")}/summary_stats/{run_name}/'
    if not os.path.exists(summary_stat_path):
        os.makedirs(summary_stat_path)

    if os.path.isfile(
        summary_stat_path + "/summary_stats_wPrecisionRecallF1FBetaAggProb.csv"
    ):
        data.to_csv(
            summary_stat_path + "/summary_stats_wPrecisionRecallF1FBetaAggProb.csv",
            mode="a",
            index=False,
            header=False,
        )
    else:
        data.to_csv(
            summary_stat_path + "/summary_stats_wPrecisionRecallF1FBetaAggProb.csv",
            index=False,
        )
    process_pool.close()

    agg_dict = {
        "start": ["max"],
        "alpha": ["max"],
        "beta": ["max"],
        "lamda": ["max"],
        "total_countries_intros_predicted": ["mean", "std"],
        "diff_total_countries": ["mean", "std"],
        "diff_total_countries_sqrd": [mse],
        "count_known_countries_time_window": ["mean", "std"],
        "diff_obs_pred_metric_mean": ["mean"],
        "diff_obs_pred_metric_stdev": [avg_std],
        "count_known_countries_time_window_recall": ["mean"],
        "count_known_countries_time_window_precision": ["mean"],
        "count_known_countries_time_window_f1": ["mean"],
        "count_known_countries_time_window_fbeta": ["mean"],
    }
    prob_agg_dict = dict(
        zip(year_probs_dict_keys, ["mean" for i in range(len(year_probs_dict_keys))])
    )
    countries_agg_dict = dict(
        zip(
            countries_dict_keys,
            [["mean", "std"] for i in range(len(countries_dict_keys))],
        )
    )

    agg_dict = {**agg_dict, **prob_agg_dict, **countries_agg_dict}

    if run_type == "forecast":
        agg_df = data.groupby("run_num").agg(agg_dict)
    else:
        agg_df = data.groupby("sample").agg(agg_dict)
    agg_df.columns = ["_".join(x) for x in agg_df.columns.values]
    agg_df.to_csv(summary_stat_path + "/summary_stats_bySample.csv")

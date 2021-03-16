import os
import glob
import math
import numpy as np
import pandas as pd
import multiprocessing


def diff_metric(difference_val, threshold_val):
    if (difference_val <= threshold_val) and (difference_val > 0):
        return round(1 - (abs(difference_val) / (threshold_val + 1)), 2)
    if (difference_val == 0):
        return 1
    else:
        return 0


def compute_summary_stats(
    model_output,
    validation_df,
    years_before_firstRecord,
    years_after_firstRecord,
    end_valid_year,
    native_countries_list
):

    presence_cols = (
        [c for c in model_output.columns if c.startswith("Presence")
            and len(c.split(" ")[-1]) == 4]
    )
    # First introduction year predicted
    model_output["PredFirstIntro"] = (
        np.where(
            model_output[presence_cols].any(axis=1) is True,
            model_output[presence_cols].idxmax(axis=1),
            "Presence 9999"
        )
    )
    model_output["PredFirstIntro"] = (
        (
            model_output["PredFirstIntro"].str.replace("Presence ", "")
        ).astype(int)
    )
    # Merge with validation data (ISO3 - First Record Year)
    model_output = model_output.merge(validation_df, how="left", on="ISO3")

    # Difference in years between prediction and obesrvation
    model_output["pred_diff"] = (
        (model_output["ObsFirstIntro"] - model_output["PredFirstIntro"])
    )

    # Does the prediction fall within the identified time window?
    model_output["temp_acc"] = (
        model_output["PredFirstIntro"].between(
            model_output["ObsFirstIntro"] - years_before_firstRecord,
            model_output["ObsFirstIntro"] + years_after_firstRecord
        )
    )

    # If predicted intro is outside of identified time window, set metric to 0.
    # Otherwise, divide by years_before value and subtract from 1 so higher values
    # indicate more "accurate" predictions accorrding to first record date
    model_output['obs-pred_metric'] = (
        (np.where(
            (model_output['pred_diff'] >= (years_before_firstRecord + 1))
            | (model_output['pred_diff'] < 0),
            0,
            (1 - (abs(model_output['pred_diff'] / years_before_firstRecord)))
        )
        )
    )

    total_intros_predicted = (
        model_output[f"Presence {str(2016)}"].sum() - len(native_countries_list)
    )
    total_intros_diff = validation_df.shape[0] - total_intros_predicted
    total_intros_diff_sqrd = total_intros_diff ** 2
    total_intros_diff_metric = diff_metric(total_intros_diff, 5)

    # Save results in dictionary from which to build the dataframe
    summary_stats_dict = ({
        # "count_known_countries_predicted":
        # (model_output.loc[validation_df.index]["PredFirstIntro"] != 9999).sum(),
        # "total_countries_intros_predicted":
        # total_intros_predicted,
        'diff_total_countries':
        total_intros_diff,
        'diff_total_countries_sqrd':
        total_intros_diff_sqrd,
        "diff_total_countries_metric":
        total_intros_diff_metric,
        "count_known_countries_time_window":
        model_output.loc[validation_df.index]["temp_acc"].sum(),
        "diff_temp_acc":
        validation_df.shape[0] - model_output["temp_acc"].sum(),
        "diff_obs_pred_metric_KOR":
        model_output.loc["KOR"]["obs-pred_metric"],
        "diff_obs_pred_metric_JPN":
        model_output.loc["JPN"]["obs-pred_metric"],
        "diff_obs_pred_metric_USA":
        model_output.loc["USA"]["obs-pred_metric"],
        # "diff_pred_year_mean":
        # model_output.loc[validation_df.index]["pred_diff"].mean(),
        "diff_obs_pred_metric_mean":
        model_output.loc[validation_df.index]["obs-pred_metric"].mean(),
        "diff_obs_pred_metric_stdev":
        model_output.loc[validation_df.index]["obs-pred_metric"].std(),
    }
    )

    return model_output, summary_stats_dict


def compute_stat_wrapper_func(param_sample):
    print(param_sample)
    native_countries_list = ["China"]
    validation_df = (
        pd.read_csv(
            'H:/Shared drives/Pandemic Data/inputs/gbif_first_records_validation.csv',
            header=0,
            index_col=0
        )
    )
    run_outputs = glob.glob(f'{param_sample}/run*/pandemic_output_aggregated.csv')
    summary_stat_df = pd.DataFrame(
        columns=[
            'sample',
            'run_num',
            'alpha',
            'lamda',
            # "count_known_countries_predicted",
            # 'total_countries_intros_predicted',
            'diff_total_countries',
            'diff_total_countries_sqrd',
            'diff_total_countries_metric',
            'count_known_countries_time_window',
            "diff_temp_acc",
            # "diff_pred_year_mean",
            'diff_obs_pred_metric_KOR',
            'diff_obs_pred_metric_JPN',
            'diff_obs_pred_metric_USA',
            'diff_obs_pred_metric_mean',
            'diff_obs_pred_metric_stdev',
        ]
    )
    for i in range(0, len(run_outputs)):
        run_num = os.path.split(run_outputs[i])[0].split('run_')[-1]
        alpha = run_outputs[i].split('alpha')[1].split('_')[0]
        lamda = run_outputs[i].split('lamda')[1].split('_')[0]
        df = pd.read_csv(
            run_outputs[i], sep=",", header=0, index_col=0, encoding='latin1'
        )
        df.set_index('ISO3', inplace=True)
        _, summary_stat_dict = compute_summary_stats(
            df, validation_df, 5, 0, 2016, native_countries_list
        )
        summary_stat_dict['run_num'] = run_num
        summary_stat_dict['sample'] = param_sample
        summary_stat_dict['alpha'] = alpha
        summary_stat_dict['lamda'] = lamda
        summary_stat_df = summary_stat_df.append(summary_stat_dict, ignore_index=True)
    # summary_stat_df = pd.DataFrame(summary_stat_dict, index=[0])
    return summary_stat_df


def mse(x):
    return sum(x) / len(x)


def avg_std(x):
    '''
    Compute average standard deviation when aggregating across runs
    of a parameter sample
    '''
    return math.sqrt(sum(x ** 2) / len(x))


def mape(x):
    return (1 / len(x)) * sum(abs(x / 3))


if __name__ == '__main__':

    root = 'H:/Shared drives/Pandemic Data/outputs'
    sim = '/slf_linearTrade_gridSearch_hiiMask16_phyto0.3-0.8_wTWN'
    param_samp = glob.glob(root + sim + '/*gridSearch*')

    process_pool = multiprocessing.Pool(processes=50)
    summary_dfs = process_pool.map(compute_stat_wrapper_func, param_samp)
    data = pd.concat(summary_dfs, ignore_index=True)
    data = data[['sample', 'run_num', 'alpha', 'lamda',
                'diff_total_countries', 'diff_total_countries_sqrd',
                 'diff_total_countries_metric', 'count_known_countries_time_window',
                 "diff_temp_acc", 'diff_obs_pred_metric_mean',
                 'diff_obs_pred_metric_stdev', 'diff_obs_pred_metric_KOR',
                 'diff_obs_pred_metric_JPN', 'diff_obs_pred_metric_USA']]

    data['diff_total_countries'] = data['diff_total_countries'].astype(int)
    data['diff_total_countries_sqrd'] = data['diff_total_countries_sqrd'].astype(float)
    data['count_known_countries_time_window'] = (
        data['count_known_countries_time_window'].astype(int)
    )
    data['diff_temp_acc'] = data['diff_temp_acc'].astype(int)
    data['diff_obs_pred_metric_KOR'] = data['diff_obs_pred_metric_KOR'].astype(float)
    data['diff_obs_pred_metric_JPN'] = data['diff_obs_pred_metric_JPN'].astype(float)
    data['diff_obs_pred_metric_USA'] = data['diff_obs_pred_metric_USA'].astype(float)

    summary_stat_path = f'{root}/summary_stats/{os.path.split(sim)[-1]}/'
    if not os.path.exists(summary_stat_path):
        os.makedirs(summary_stat_path)
    data.to_csv(summary_stat_path + '/summary_stats_v5.csv')

    agg_df = (
        data.groupby('sample').agg(
            {
                'diff_total_countries': ['mean', 'std'],
                'diff_total_countries_sqrd': ['sum', mse],
                'diff_total_countries_metric': ['mean', 'std'],
                'count_known_countries_time_window': ['mean', 'std'],
                "diff_temp_acc": ['mean', 'std'],
                # "diff_pred_year_mean": ['mean', 'std'],
                'diff_obs_pred_metric_KOR': ['mean', 'std'],
                'diff_obs_pred_metric_JPN': ['mean', 'std'],
                'diff_obs_pred_metric_USA': ['mean', 'std'],
                'diff_obs_pred_metric_mean': ['mean'],
                'diff_obs_pred_metric_stdev': [avg_std],
            }
        )
    )

    agg_df.columns = ["_".join(x) for x in agg_df.columns.values]
    agg_df.to_csv(summary_stat_path + '/summary_stats_bySample_v5.csv')

import os
import sys
import glob
import fnmatch
import shutil
import random
import numpy as np
import pandas as pd


def create_date_lists(
    start_forecast_date,
    number_historical_years,
    number_forecast_years,
):
    """
    Create lists of trade file to use as historical values for trade forecasts.
    Function also returns a list to use for naming forecast files

    Parameters:
    ----------
    start_forecast_date : int
        Numeric representation of the date to start forecasting
        trade as YYYY or YYYYMM (e.g., 2020 or 202001)
    number_historical_years : int
        Years of historical data to use when forecasting
        (e.g., 5)
    number_forecast_years : int
        Number of years to forecast trade into the future
        (e.g., 10)

    Returns:
    --------
    hist_ts_list : list
        List of historical timestamps
    forecast_ts_list : list
        List of forecast timestamps
    month_list : list
        List of MM strings

    """
    # If generating an annual forecast, date lists include
    # a 4-digit date format (e.g., 2020)
    if len(str(start_forecast_date)) == 4:
        end_forecast_date = start_forecast_date + number_forecast_years - 1
        hist_ts_list = list(
            range(start_forecast_date - number_historical_years, start_forecast_date, 1)
        )
        forecast_ts_list = list(range(start_forecast_date, end_forecast_date + 1, 1))

    # If generating a monthly forecast, date lists include
    # a 6-digit date format(e.g., 202012 for December 2020)
    if len(str(start_forecast_date)) == 6:
        start_year = int(str(start_forecast_date)[:4])
        end_year = start_year + number_forecast_years - 1
        end_forecast_date = int(str(end_year) + "12")
        month_list = [f"{x:02d}" for x in range(1, 13)]
        hist_ts_list = []
        hist_year_list = list(
            range((start_year - number_historical_years), start_year, 1)
        )
        for year in hist_year_list:
            hist_year_month = [f"{year}" + str(month) for month in month_list]
            hist_ts_list.append(hist_year_month)
        hist_ts_list = [y for x in hist_ts_list for y in x]

        forecast_ts_list = []
        forecast_year_list = list(range(start_year, end_year + 1, 1))
        for year in forecast_year_list:
            forecast_year_mo = [f"{year}" + str(month) for month in month_list]
            forecast_ts_list.append(forecast_year_mo)
        forecast_ts_list = [y for x in forecast_ts_list for y in x]
    return hist_ts_list, forecast_ts_list, month_list


def create_trade_arrays(list_of_csvs, number_forecast_years):
    """
    Generate matrices of forecasted trade data based on selection
    of historical trade data.

    Parameters:
    ----------
    list_of_csvs : list
        List of historical trade csv files to use when
        generating a trade forecast
    number_forecast_years : int
        Number of years to forecast trade into the future
        (e.g., 10)

    Returns:
    --------
    hist_arr : numpy.array
        t x n x n matrix of historical trade data where t
        is the number of time steps used to create the
        forecast and n is the number of locations
    forecast_arr : numpy.array
        t x n x n matrix of forecasted trade data where t
        is the number of time steps forecasted and n is
        the number of locations

    """
    example_matrix = pd.read_csv(
        list_of_csvs[0], header=0, index_col=0, encoding="latin1"
    )
    hist_arr = np.zeros(
        shape=(len(list_of_csvs), example_matrix.shape[0], example_matrix.shape[0])
    )
    forecast_arr = np.zeros(
        shape=(number_forecast_years, example_matrix.shape[0], example_matrix.shape[0])
    )

    for i in range(len(list_of_csvs)):
        hist_arr[i] = pd.read_csv(
            list_of_csvs[i],
            sep=",",
            header=0,
            index_col=0,
            encoding="latin1",
        ).values
    # Randomly choose a value from the historical trade matrices
    # to populate the trade forecast for each destination (j) -
    # origin (i) pair and forecasted time step (k)
    for k in range(0, forecast_arr.shape[0]):
        for j in range(0, hist_arr.shape[1]):
            for i in range(0, hist_arr.shape[2]):
                forecast_arr[k, j, i] = random.choice(hist_arr[:, j, i])
    return hist_arr, forecast_arr


def write_forecast_arrays(
    list_of_csvs,
    forecast_arr,
    forecast_ts_list,
    output_dir,
):
    """
    Saves the trade forecasts files to the specified directory.

    Parameters:
    ----------
    list_of_csvs : list
        List of historical trade csv files to used when
        generating the trade forecast
    forecast_arr : numpy.array
        t x n x n matrix of forecasted trade data where t
        is the number of time steps forecasted and n is
        the number of locations
    forecast_ts_list :list
        List of forecast timestamps
    output_dir : str
        File path where forecasted trade files will be saved

    Returns:
    --------
    none

    """

    file_prefix = os.path.basename(list_of_csvs[0]).split("_")[0]
    example_matrix = pd.read_csv(
        list_of_csvs[0], header=0, index_col=0, encoding="latin1"
    )

    for i in range(0, forecast_arr.shape[0]):
        ts = forecast_ts_list[i]
        arr = forecast_arr[i, :, :]
        forecast_df = pd.DataFrame(
            data=arr, columns=example_matrix.columns, index=example_matrix.index
        )
        forecast_df.to_csv(output_dir + f"/{file_prefix}_trades_{ts}.csv")


start_forecast_date = sys.argv[1]
number_historical_years = sys.argv[2]
number_forecast_years = sys.argv[3]

root_dir = "G:/Shared drives/APHIS  Projects/Pandemic/Data/slf_model"

output_dir = root_dir + "/inputs/trade_forecast/monthly_agg/6801-6804"
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

os.makedirs(output_dir)

hist_ts_list, forecast_ts_list, month_list = create_date_lists(
    start_forecast_date, number_historical_years, number_forecast_years
)

historical_trade_dir = root_dir + "/inputs/monthly_agg/6801-6804"
historical_trade = glob.glob(historical_trade_dir + "/*")
hist_trade_to_use = [
    timestep
    for timestep in historical_trade
    if (os.path.basename(timestep)[:-4].split("_")[-1]) in (hist_ts_list)
]

# For monthly forecasts
if len(str(start_forecast_date)) == 6:
    for month in month_list:
        hist_trade_to_use_subsample = fnmatch.filter(
            hist_trade_to_use, f"*20*{month}.csv"
        )
        hist_arr, forecast_arr = create_trade_arrays(
            hist_trade_to_use_subsample, number_forecast_years
        )
        forecast_ts_list_filtered = fnmatch.filter(forecast_ts_list, f"*20*{month}")
        write_forecast_arrays(
            hist_trade_to_use_subsample,
            forecast_arr,
            forecast_ts_list_filtered,
            output_dir,
        )
# For annual forecasts
elif len(str(start_forecast_date)) == 4:
    hist_arr, forecast_arr = create_trade_arrays(
        hist_trade_to_use, number_forecast_years
    )
    write_forecast_arrays(hist_trade_to_use, forecast_arr, forecast_ts_list, output_dir)
else:
    print("format start_forecast_date as YYYY or YYYYMM")

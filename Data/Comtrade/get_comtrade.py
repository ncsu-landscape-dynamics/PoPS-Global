# Download and format data from Comtrade API
import sys
import math
import os
import json
import urllib
import pandas as pd
import numpy as np
import time
from datetime import datetime


def nested_list(original_list, list_length):
    """Split long list into nested list of lists at specified length.
    Returns nested list.
    Ex: Create short lists of years or country codes for API calls."""
    nested_lists = []
    start = 0
    for unused_i in range(math.ceil(len(original_list) / list_length)):
        x = original_list[start : start + list_length]
        nested_lists.append(x)
        start += list_length
    return nested_lists


def download_trade_data(hs_str, freq_str, year_country_dict, auth_code_str):
    """Loops over years and countries, calling Comtrade API for specified
    HS commodity code and appends downloaded data to dataframe. Prints
    messages to track progress.
    Returns dataframe of trade data, one row for each
    origin/destination/timestep combo.
    Parameters
    ----------
    hs_str : str
        HS commodity code, can be 2, 4, or 6 digits depending on desired
        aggregation
    freq_str: str
        temporal resolution of data, "A" for annual, "M" for monthly
    year_country_dict: dict
        specifies which countries to use for which years, keys are years,
        values are lists of UN country codes
    auth_code_str: str
        premium API authorization code
    """
    print(f"Downloading HS{hs_str} @ {freq_str} data from Comtrade...")
    data = pd.DataFrame()
    for key in year_country_dict.keys():
        country_codes = year_country_dict[key]
        # Set number of countries to call at once
        num_countries_per_call = 200
        nested_country_codes = nested_list(country_codes, num_countries_per_call)
        # loop over all country lists countries and call API
        for country_code_list in nested_country_codes:
            country_code_str = "%2C".join(country_code_list)
            url_str = (
                "http://comtrade.un.org/api/get?max=250000&type=C&px=HS&cc="
                + hs_str
                + "&r="
                + country_code_str
                # rg=1 (imports only)
                + "&rg=1&p=all&freq="
                + freq_str
                + "&ps="
                + str(key)
                + "&fmt=json&token="
                + auth_code_str
            )
            delay = 5
            retries = 1
            success = False
            while not success and retries <= 10:
                try:
                    url = urllib.request.urlopen(url_str)
                    success = True
                    print("\t\t!! Success...")
                except Exception as e:
                    print("\t", e)
                    if delay < 60:
                        print(f"\t\tRetry {retries} of 10 in {delay} seconds")
                    else:
                        print(f"\t\tRetry {retries} of 10 in {delay/60:.2f} minutes")
                    sys.stdout.flush()
                    time.sleep(delay)
                    delay *= 2
                    retries += 1

            if "url" in locals():
                raw = json.loads(url.read().decode())
                url.close()

                if len(raw["dataset"]) == 0:
                    print(
                        "No data downloaded for HS"
                        + hs_str
                        + ", "
                        + str(key)
                        + ", UN code:"
                        + ", ".join(map(str, country_code_list))
                        + ". Message: "
                        + str(raw["validation"]["message"])
                    )
                    continue

                data = data.append(raw["dataset"])
                data["ptCode"] = data["ptCode"].astype(str)
                print(
                    "Freq: "
                    + freq_str
                    + " HS"
                    + hs_str
                    + " "
                    + str(key)
                    + ", UN code:"
                    + ", ".join(map(str, country_code_list))
                    + ", downloaded"
                )
            else:
                print(f"***{key}: {hs_str} failed")
    return data


def save_hs_timestep_matrices(
    hs_str, timesteps_list, trade_data, un_country_df, un_to_iso_dict
):
    """Loops over timesteps and creates country x country matrix of import
    trade values. If no data were downloaded for a country pair, will add zeros and
    print a message. Changes US codes to ISO3 codes in column, row names.
    Saves a matrix for each timestep as CSVs.
    Parameters
    ----------
    hs_str : str
        HS commodity code, can be 2, 4, or 6 digits depending on desired
        aggregation
    timesteps_list : list of int
        list of timesteps downloaded, will use format YYYY or YYYYMM
    trade_data: dataframe
        dataframe of downloaded trade data
    un_country_df: dataframe
        dataframe with single column containing all UN country codes, used as template
        for HS matrix
    un_to_iso_dict: dict
        dictionary crosswalk of UN to ISO3 codes
    """
    print(f"Saving HS{hs_str} matrices for each timestep...")
    for timestep in timesteps_list:
        timestep_data = trade_data[trade_data.period.eq(timestep)]
        HS_matrix = un_country_df.copy()
        for reporter in un_to_iso_dict.keys():
            reporter_data = timestep_data[timestep_data.rtCode.eq(int(reporter))]
            reporter_data = reporter_data[["ptCode", "TradeValue"]]
            # if no data were downloaded for reporter/timestep, add column of
            # zeros to commodity/year df and move to next country
            if len(reporter_data) == 0:
                HS_matrix = HS_matrix.assign(x=0)
                HS_matrix.rename(columns={"x": reporter}, inplace=True)
                print(
                    "HS"
                    + hs_str
                    + " "
                    + str(timestep)
                    + " "
                    + str(un_to_iso_dict[reporter])
                    + ": no data"
                )
                continue

            # Merge to commodity/year df
            HS_matrix = pd.merge(
                HS_matrix, reporter_data, how="left", left_on="UN", right_on="ptCode",
            )
            HS_matrix.drop("ptCode", axis=1, inplace=True)
            HS_matrix.rename(columns={"TradeValue": reporter}, inplace=True)
            print(
                "HS"
                + hs_str
                + " "
                + str(timestep)
                + " "
                + str(un_to_iso_dict[reporter])
                + ": finished"
            )

        # Use crosswalk to change UN country codes to ISO3
        HS_matrix.fillna(0, inplace=True)
        HS_matrix.rename(columns={"UN": "ISO"}, inplace=True)
        HS_matrix.set_index("ISO", inplace=True)
        HS_matrix.rename(mapper=un_to_iso_dict, inplace=True, axis=0)
        HS_matrix.rename(mapper=un_to_iso_dict, inplace=True, axis=1)

        # Check to see if there are any duplicate ISO codes in matrix
        # Should only occur for a few historical cases - Germany, Viet Nam, Yemen.
        # If so, group and sum any rows and columns with matching ISO codes.
        if len(HS_matrix.columns.to_list()) != len(set(HS_matrix.columns.to_list())):
            HS_matrix = HS_matrix.groupby("ISO", sort=False).sum()
            HS_matrix = HS_matrix.transpose()
            HS_matrix = HS_matrix.reset_index()
            HS_matrix = HS_matrix.groupby("index", sort=False).sum().transpose()
        assert len(HS_matrix.columns.to_list()) == len(set(HS_matrix.columns.to_list()))
        # Tranpose matrix so that columns are exporting partners (origins) and
        # rows are importing reporters (destinations) to match other model input data
        HS_matrix = HS_matrix.transpose()
        # create a directory to save downloaded data
        if not os.path.exists(hs_str):
            os.makedirs(hs_str)
        HS_matrix.to_csv(
            hs_str + "/" + hs_str + "_" + str(timestep) + ".csv", index=True
        )


def query_comtrade(
    model_inputs_dir,
    auth_code,
    hs_list,
    start_year,
    end_year,
    temporal_res,
    crosswalk_path,
):
    """
    Runs trade data request and download process, including
    review of data completeness.
    Parameters
    ----------
    model_inputs_dir : str
        Directory path to where data will be saved
    auth_code_str: str
        premium API authorization code
    hs_list : list
        List of HS commodity codes, which can be 2, 4, or 6
        digits
    start_year : int
        First year (YYYY) requested
    end_year : int
        Last year (YYYY) requested
    temporal_res : str
        temporal resolution of data, "A" for annual, "M" for monthly
    crosswalk_path : str
        Location of UN code to ISO3 code crosswalk csv
    """

    # list HS commodity codes to query, will be downloaded individually
    # and aggregated (if needed) in later script

    # Set time step for trade data
    years = np.arange(start_year, end_year + 1, 1)

    # Read UN codes to ISO3 codes crosswalk to use as country list
    crosswalk = pd.read_csv(crosswalk_path)
    crosswalk["UN"] = crosswalk["UN"].astype(str)
    crosswalk = crosswalk[crosswalk.ISO3.notnull()]

    # Change "Now" end date in crosswalk to a max year value so column can be converted
    # to numeric and compared to simulation years
    crosswalk.loc[crosswalk["End"] == "Now", "End"] = "9999"
    crosswalk["End"] = pd.to_numeric(crosswalk["End"])
    crosswalk = crosswalk[crosswalk["End"] >= start_year]
    # Create dictionary to convert UN to ISO3 codes
    crosswalk_dict = pd.Series(crosswalk.ISO3.values, index=crosswalk.UN).to_dict()

    # Get data availability from Comtrade and compare to desired data
    data_availability_url = urllib.request.urlopen(
        "http://comtrade.un.org/api/refs/da/view?type=C&freq=all&ps=all&px=HS"
    )
    data_availability_raw = json.loads(data_availability_url.read().decode())
    data_availability_url.close()
    data_availability = pd.json_normalize(data_availability_raw)

    # Create summary of availability, tracking availability of annual,
    # and monthly (all 12 months, or less than 12)
    data_summary = pd.DataFrame(
        columns=[
            "country",
            "year",
            "annual_avail",
            "all_monthly_avail",
            "partial_monthly_avail",
        ]
    )

    for country in crosswalk.UN.to_list():
        country_availability = data_availability[data_availability["r"] == country]
        for year in years:
            summary_data = {
                "country": [country],
                "year": [year],
                "annual_avail": [0],
                "all_monthly_avail": [0],
                "partial_monthly_avail": [0],
            }
            year_summary = pd.DataFrame(summary_data)
            if str(year) in list(country_availability["ps"]):
                year_summary["annual_avail"] = 1
            country_monthly = list(
                country_availability[country_availability["freq"] == "MONTHLY"][
                    "ps"
                ].str[:4]
            )
            monthly_sum = sum(1 for i in country_monthly if i == str(year))
            if monthly_sum == 12:
                year_summary["all_monthly_avail"] = 1
            if 0 < monthly_sum < 12:
                year_summary["partial_monthly_avail"] = 1
            data_summary = data_summary.append(year_summary)

    # Create df specifying if annual or monthly should be download
    # based on desired temp res for each country and year
    if temporal_res == "A":
        use_annual = data_summary[data_summary["annual_avail"] == 1]
        use_monthly = data_summary[
            (data_summary["annual_avail"] == 0)
            & (
                (data_summary["all_monthly_avail"] == 1)
                | (data_summary["partial_monthly_avail"] == 1)
            )
        ]
    if temporal_res == "M":
        use_monthly = data_summary[data_summary["all_monthly_avail"] == 1]
        use_monthly = use_monthly.append(
            data_summary[
                (data_summary["partial_monthly_avail"] == 1)
                & (data_summary["annual_avail"] == 0)
            ]
        )
        use_annual = data_summary[
            (data_summary["annual_avail"] == 1)
            & (data_summary["all_monthly_avail"] == 0)
        ]
    # no_data = data_summary[
    #     (data_summary["annual_avail"] == 0)
    #     & (data_summary["all_monthly_avail"] == 0)
    #     & (data_summary["partial_monthly_avail"] == 0)
    # ]
    # Save data summary as CSV for future reference
    # create a directory to save summary if needed
    if not os.path.exists(model_inputs_dir):
        os.makedirs(model_inputs_dir)
    data_summary.to_csv(
        model_inputs_dir
        + "/comtrade_data_availability_summary_"
        + str(start_year)
        + "-"
        + str(end_year)
        + ".csv"
    )

    use_annual_dict = use_annual.groupby("year")["country"].apply(list).to_dict()
    use_monthly_dict = use_monthly.groupby("year")["country"].apply(list).to_dict()

    # create a directory to save downloaded data
    if temporal_res == "A":
        if not os.path.exists(model_inputs_dir + "/annual"):
            os.makedirs(model_inputs_dir + "/annual")
        os.chdir(model_inputs_dir + "/annual")
    if temporal_res == "M":
        if not os.path.exists(model_inputs_dir + "/monthly"):
            os.makedirs(model_inputs_dir + "/monthly")
        os.chdir(model_inputs_dir + "/monthly")

    # loop over commodities, 1 at a time (could do more at once to speed it up)
    if temporal_res == "A":
        for hs in hs_list:
            # Download either annual or monthly depending on availability
            freq = "A"
            annual_data = download_trade_data(str(hs), freq, use_annual_dict, auth_code)
            freq = "M"
            print("Checking alternative temporal resolution to augment missing data...")
            monthly_data = download_trade_data(
                str(hs), freq, use_monthly_dict, auth_code
            )
            # Sum monthly to get annual
            if not monthly_data.empty:
                monthly_data_agg = (
                    monthly_data.groupby(["yr", "rtCode", "ptCode"]).sum().reset_index()
                )
                annual_data = annual_data.append(monthly_data_agg)
            # loop over timesteps (YYYY) and save a country x country matrix
            # per timestep per HS code as csv
            timesteps = years
            if not annual_data.empty:
                save_hs_timestep_matrices(
                    str(hs), timesteps, annual_data, crosswalk[["UN"]], crosswalk_dict
                )
            else:
                print("No annual data downloaded")

    if temporal_res == "M":
        current_year = datetime.now().year
        current_month = datetime.now().month
        for hs in hs_list:
            timesteps = []
            months = [str(f"{i:02}") for i in range(1, 13)]
            for year in years:
                for month in months:
                    timesteps.append(str(year) + month)
            timesteps = list(map(int, timesteps))
            timesteps_not_complete = list(
                range(
                    (int(str(current_year) + str(f"{current_month:02}"))),
                    (int(str(current_year) + "13")),
                )
            )
            timesteps = [ts for ts in timesteps if ts not in timesteps_not_complete]
            # Download either annual or monthly depending on availability
            freq = "M"
            monthly_data = download_trade_data(
                str(hs), freq, use_monthly_dict, auth_code
            )

            if year != current_year:
                freq = "A"
                annual_data = download_trade_data(
                    str(hs), freq, use_annual_dict, auth_code
                )
                if not annual_data.empty:
                    # Split annual data into monthly by dividing by 12
                    annual_split = pd.DataFrame()
                    for month in months:
                        month_portion = annual_data.copy()
                        month_portion["TradeValue"] = month_portion["TradeValue"].apply(
                            lambda x: x / 12
                        )
                        month_portion["period"] = (
                            month_portion["period"].astype(str) + month
                        )
                        month_portion["period"] = month_portion["period"].astype(int)
                        annual_split = annual_split.append(month_portion)
                monthly_data = monthly_data.append(annual_split)
            # loop over timesteps (YYYY or YYYYMM) and save a country x country matrix
            # per timestep per HS code as csv
            if not monthly_data.empty:
                save_hs_timestep_matrices(
                    str(hs), timesteps, monthly_data, crosswalk[["UN"]], crosswalk_dict
                )
            # else:
            #     print('\tNo monthly data available to download')


# project_path = "H:/My drive/Projects/Pandemic"
# load_dotenv(os.path.join(project_path, ".env"))
# # Root project data folder
# data_path = os.getenv("DATA_PATH")
# # Path to formatted model inputs
# model_inputs_dir = data_path + "slf_model/test/"

# # Premium subscription authorization code.
# auth_code = os.getenv("COMTRADE_AUTH_KEY")

# query_comtrade(
#     model_inputs_dir=model_inputs_dir,
#     auth_code=auth_code,
#     start_code=6801,
#     end_code=6804,
#     start_year=2000,
#     end_year=2019,
#     temporal_res='M',
#     crosswalk_path="H:/Shared drives/APHIS  Projects/Pandemic/Data/un_to_iso.csv",
# )

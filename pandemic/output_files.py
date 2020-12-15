import os
import numpy as np
import pandas as pd

# import geopandas as gpd
# from shapely.geometry.polygon import Polygon
# from shapely.geometry.multipolygon import MultiPolygon


def create_model_dirs(outpath, output_dict):
    """
    Creates directory and folders for model output files.

    Parameters
    ----------
    run_num : Integer
        Number identifying the model run
    outpath : String
        Absolute path of directory where model output are saved
    output_dict : Dictionary
        Key-value pairs identifying the object name and folder name
        of model output components.

    Returns
    -------
    None

    """

    os.makedirs(outpath, exist_ok=True)

    for key in output_dict.keys():
        os.makedirs(outpath + key, exist_ok=True)


def save_model_output(
    model_output_object, columns_to_drop, example_trade_matrix, outpath, date_list
):
    """
    Saves model output, including probabilities for entry, establishment,
    and introduction. Full forecast dataframe, origin-destination pairs,
    and list of time steps formatted as YYYYMM.

    Parameters
    ----------
    model_output_object : numpy array
        List of 6 n x n arrays created by running pandemic model, ordered as
        1) full forecast dataframe; 2) probability of entry;
        3) probability of establishment; 4) probability of introduction;
        5) origin - destination pairs; and 6) list of countries where pest is
        predicted to be introduced

    columns_to_drop : list
        List of columns used or created by the model that are to be dropped
        from the final output (e.g., Koppen climate classifications)

    example_trade_matrix : numpy array
        Array of trade data from one time step as example to format
        output dataframe columns and indices

    outpath : str
        String specifying absolute path of output directory

    date_list : list
        List of unique time step values (YYYY or YYYYMM)

    Returns
    -------
    out_df : geodataframe
        Geodataframe of model outputs


    """

    model_output_df = model_output_object[0]
    prob_entry = model_output_object[1]
    prob_est = model_output_object[2]
    prob_intro = model_output_object[3]
    origin_dst = model_output_object[4]
    country_intro = model_output_object[5]

    out_df = model_output_df.drop(columns_to_drop, axis=1)
    # out_df["geometry"] = [
    #     MultiPolygon([feature]) if type(feature) == Polygon else feature
    #     for feature in out_df["geometry"]
    # ]
    # out_df.to_file(outpath + "/pandemic_output.geojson", driver="GeoJSON")
    out_pdf = pd.DataFrame(out_df.drop(columns="geometry", axis=1))
    out_pdf.to_csv(outpath + "/pandemic_output.csv")

    origin_dst.to_csv(outpath + "/origin_destination.csv")

    for i in range(0, len(date_list)):
        ts = date_list[i]

        pro_entry_pd = pd.DataFrame(prob_entry[i])
        pro_entry_pd.columns = example_trade_matrix.columns
        pro_entry_pd.index = example_trade_matrix.index
        pro_entry_pd.to_csv(
            outpath + f"/prob_entry/probability_of_entry_{str(ts)}.csv",
            float_format="%.4f",
            na_rep="NAN!",
        )

        pro_intro_pd = pd.DataFrame(prob_intro[i])
        pro_intro_pd.columns = example_trade_matrix.columns
        pro_intro_pd.index = example_trade_matrix.index
        pro_intro_pd.to_csv(
            outpath + f"/prob_intro/probability_of_introduction_{str(ts)}.csv",
            float_format="%.4f",
            na_rep="NAN!",
        )

        pro_est_pd = pd.DataFrame(prob_est[i])
        pro_est_pd.columns = example_trade_matrix.columns
        pro_est_pd.index = example_trade_matrix.index
        pro_est_pd.to_csv(
            outpath + f"/prob_est/probability_of_establishment_{str(ts)}.csv",
            float_format="%.4f",
            na_rep="NAN!",
        )

        country_int_pd = pd.DataFrame(country_intro[i])
        country_int_pd.columns = example_trade_matrix.columns
        country_int_pd.index = example_trade_matrix.index
        country_int_pd.to_csv(
            outpath + f"/country_introduction/country_introduction_{str(ts)}.csv",
            float_format="%.4f",
            na_rep="NAN!",
        )

    return out_df


def agg_prob(row, column_list):
    """
    Calculates the probability of introduction for a year
    based on the non-zero monthly probabilities of
    introduction.

    Parameters
    -----------
    row
        Row of a data frame to use for calculations

    column_list : list
        List of columns containing the probabilities
        to aggregate

    Returns
    --------
    final_prob : float
        Probablity of introduction for a given year
        based on the monthly probabilities of
        introduction

    """

    non_zero = []
    for i in range(0, len(column_list)):
        if row[column_list[i]] > 0.0:
            non_zero.append(row[column_list[i]])
    prod_out = np.prod(list(map(lambda x: 1 - x, non_zero)))
    final_prob = 1 - prod_out

    return final_prob


def get_feature_cols(geojson_obj, feature_chars):
    """
    Get list of columns that start with the identified
    characters of interest

    Parameters
    ----------
    geojson_obj : geodataframe
        A geodataframe object containing the original
        model output columns and format

    feature_chars : str
        String of characters identifying the column
        prefix of interest

    Returns
    -------
    feature_cols : list
        List of all columns starting with the identified
        string

    feature_cols_monthly
        List of all monthly time step columns starting with
        the identified string

    feature_cols_annual
        List of all annual time step columns starting with
        the identified string

    """

    feature_cols = [c for c in geojson_obj.columns if c.startswith(feature_chars)]
    feature_cols_monthly = [c for c in feature_cols if len(c.split(" ")[-1]) > 5]
    feature_cols_annual = [c for c in feature_cols if c not in feature_cols_monthly]

    return feature_cols, feature_cols_monthly, feature_cols_annual


def create_feature_dict(geojson_obj, column_list, chars_to_strip):
    """
    Create a dictionary of year and value pairs based on
    multiple columns with the same prefix

    Parameters
    ----------
    geojson_obj : geodataframe
        A geodataframe containing the original
        model output columns and format

    column_list : list
        List of columns to use

    chars_to_strip: str
        Characters to remove from the dictionary key


    Returns
    --------
    d : iterable object
        Dictionary of year and value pairs

    """
    d = geojson_obj[column_list].to_dict("index")
    for key in d.keys():
        d[key] = {k.strip(chars_to_strip): v for k, v in d[key].items()}

    return d


def add_dict_to_geojson(geojson_obj, new_col_name, dictionary_obj):
    """
    Add dictionary of year and value pairs as a new feature
    to a geojson

    Parameters
    ----------
    geojson_obj : geodataframe
        A geodataframe containing the original
        model output columns and format

    new_col_name : str
        Name of new column to be added to the
        geodataframe

    dictionary_obj
        Dictionary of year and value pairs to
        add as new column to the geodataframe

    Returns
    -------
        geojson_obj : geodataframe
            Geodataframe with new column added

    """

    geojson_obj[new_col_name] = geojson_obj.index.map(dictionary_obj)

    return geojson_obj


def aggregate_monthly_output_to_annual(formatted_geojson, outpath):
    """
    Aggregate monthly time step predictions from the model to annual
    predictions of presence and probability of introduction

    Parameters
    ----------
    formatted_geojson : geodataframe
        Geodataframe containing original model output as well as
        additional columns with year: value dictionaries.

    outpath : str
        Directory path to save output (geojson and csv)

    Returns
    -------
    none

    """
    prob_intro_cols = [
        c
        for c in formatted_geojson.columns
        if c.startswith("Probability of introduction")
    ]
    annual_ts_list = sorted(set([y.split(" ")[-1][:4] for y in prob_intro_cols]))
    for year in annual_ts_list:
        prob_cols = [c for c in prob_intro_cols if str(year) in c]
        formatted_geojson[f"Agg Prob Intro {year}"] = formatted_geojson.apply(
            lambda row: agg_prob(row=row, column_list=prob_cols), axis=1
        )
        formatted_geojson[f"Presence {year}"] = formatted_geojson[f"Presence {year}12"]

    presence_cols = [c for c in formatted_geojson.columns if c.startswith("Presence")]
    formatted_geojson.to_file(
        outpath + "/pandemic_output_aggregated.geojson", driver="GeoJSON"
    )
    out_csv = pd.DataFrame(formatted_geojson)
    out_csv.drop(["geometry"], axis=1, inplace=True)
    out_csv.to_csv(
        outpath + "/pandemic_output_aggregated.csv", float_format="%.2f", na_rep="NAN!"
    )
    presence_cols_monthly = [c for c in presence_cols if len(c.split(" ")[-1]) > 5]
    presence_cols_annual = [c for c in presence_cols if c not in presence_cols_monthly]
    agg_prob_cols_annual = [c for c in formatted_geojson.columns if c.startswith("Agg")]

    presence_d = create_feature_dict(
        geojson_obj=formatted_geojson,
        column_list=presence_cols_annual,
        chars_to_strip="Presence ",
    )

    agg_prob_d = create_feature_dict(
        geojson_obj=formatted_geojson,
        column_list=agg_prob_cols_annual,
        chars_to_strip="Agg Prob Intro ",
    )
    new_gdf = add_dict_to_geojson(
        geojson_obj=formatted_geojson,
        new_col_name="Presence",
        dictionary_obj=presence_d,
    )
    new_gdf = add_dict_to_geojson(
        geojson_obj=new_gdf, new_col_name="Agg Prob Intro", dictionary_obj=agg_prob_d
    )
    cols_to_drop = [
        c
        for c in new_gdf.columns
        if c in presence_cols_monthly or c.startswith("Probability")
    ]

    # sm_gdf = new_gdf.drop(cols_to_drop, axis=1)
    # sm_gdf.to_file(
    #     outpath + f"/pandemic_output_aggregated_select.geojson", driver="GeoJSON"
    # )
    sm_csv = pd.DataFrame(new_gdf.drop(cols_to_drop + ["geometry"], axis=1))
    # sm_csv.drop(["geometry"], axis=1, inplace=True)
    sm_csv.to_csv(
        outpath + "/pandemic_output_aggregated_select.csv",
        float_format="%.2f",
        na_rep="NAN!",
    )

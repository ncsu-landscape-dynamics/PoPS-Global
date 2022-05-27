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

"""Module containing all calculations for helper functions for use in the PoPS
Global simulation.
"""

import os
import glob
import pandas as pd
import numpy as np
from haversine import haversine

# from shapely.geometry.polygon import Polygon
# from shapely.geometry.multipolygon import MultiPolygon


def distance_between(array_template, shapefile):
    """
    Returns a n x n numpy array with the the distance from each element in a
    shapefile to all other elements in that shapefile.

    Parameters
    ----------
    array_template : array (float)
        n x n template matrix where n is number of locations
    shapefile : geodataframe
        A geopandas dataframe of countries with crs(epsg = 4326)

    Returns
    -------
    distance_array : numpy array
        An n x n numpy array of Haversine distances (great circle) from each
        location to every other location in kilometer

    """

    distance_array = np.zeros_like(array_template, dtype=float)
    centroids = shapefile.centroid.geometry
    shapefile["centroid_lon"] = centroids.x
    shapefile["centroid_lat"] = centroids.y
    centroids_array = shapefile.loc[:, ["centroid_lat", "centroid_lon"]].values
    for j in range(len(shapefile)):
        destination = centroids_array[j]
        for i in range(len(shapefile)):
            origin = centroids_array[i]
            distance = haversine(origin, destination)
            distance_array[j, i] = distance

    return distance_array


def location_pairs_with_host(locations):
    """
    Returns a list of countries that have host species
    presence > 0%

    Parameters
    ----------
    locations : data_frame
        data frame of nodes with species presence, phytosanitry capacity,
        koppen climate classifications % of total area for each class,
        and host percent area.

    Returns
    --------
    locations_list : list
        list of nodes with their corresponding attributes as a series
        for nodes with host species presence greater than 0%

    """

    locations_with_host_df = locations.loc[locations["Host Percent Area"] > 0]
    origins = list(
        locations_with_host_df.loc[locations_with_host_df["Presence"]]["ISO3"]
    )
    destinations = list(locations_with_host_df["ISO3"])
    origins_list = [
        country for country in origins for i in range(locations_with_host_df.shape[0])
    ]
    destinations_list = destinations * len(origins)
    location_tuples = list(zip(origins_list, destinations_list))
    # remove location tuples where origin and destination are the same country
    location_tuples = [i for i in location_tuples if i[0] != i[1]]

    return location_tuples


def filter_trades_list(file_list, start_year, stop_year=None):
    """
    Returns filtered list of trade data based on start
    year

    Parameters:
    -----------
    file_list : list
        List of all trade data files in a directory
    start_year : int
        Simulation start year (YYYY)
    stop_year : int (optional)
        Simulation end year (YYYY) used to filter
        trade data if files are after that year

    Returns:
    --------
    file_list_filtered: list
        List of trade data starting with simulation
        start year

    """

    for i, f in enumerate(file_list):
        date_tag = str.split(os.path.splitext(os.path.split(f)[1])[0], "_")[-1][:4]
        # File time step before start year
        if int(date_tag) < int(start_year):
            file_list[i] = None
        # File time step after stop year if specified
        if stop_year is not None and (int(date_tag) > int(stop_year)):
            file_list[i] = None
    file_list_filtered = [f for f in file_list if f is not None]

    return file_list_filtered


def create_trades_list(
    commodity_path,
    commodity_forecast_path,
    commodity_list,
    start_year,
    distances,
    stop_year=None,
):
    """
    Returns list (c) of n x n x t matrices, filtered by start year, where c is
    the number of commodities, n is the number of locations, t is the number
    of time steps,

    Parameters:
    -----------
    commodity_path : str
        path to all historical commodity trade data
    commodity_forecast_path : str
        path to forecasted commodity trade data
    commodity_list: list
        List of strings, each with a commodity code or aggregate
    start_year : int
        Simulation start year (YYYY) used to filter
        trade data files that are prior to that year
    distances : numpy.array
        n x n matrix of distances from one location to another where n is
        number of locations
    stop_year : int (optional)
        Simulation end year (YYYY) used to filter
        trade data if files are after that year

    Returns:
    --------
    trades_list: list
        list (c) of n x n x t matrices where c is the # of commoditites,
        n is the # of locations, and t is # of time steps
    file_list_filtered : list
        list of filtered commodity (historical and forecast) file paths
    code_list : list
        list of commodity codes available in commodity directory,
        that match the commodity list
    commodities_available : list
        list of all commodity file paths

    """
    commodities_available = glob.glob(commodity_path + "*")
    commodities_available.sort()

    codes_available = [os.path.split(f)[1] for f in commodities_available]
    code_list = list(set(codes_available).intersection(set(commodity_list)))
    
    trades_list = []
    print("Loading and formatting trade data...")
    # If trade data are aggregated (i.e., summed across
    # multiple commodity codes)
    if len(code_list) == 1:
        print("\t", code_list[0])
        file_list_historical = glob.glob(f"{commodity_path}/{code_list[0]}/*.csv")
        file_list_historical.sort()
        if commodity_forecast_path is not None:
            file_list_forecast = glob.glob(f"{commodity_forecast_path}/{code_list[0]}/*.csv")
            file_list_forecast.sort()
            file_list = file_list_historical + file_list_forecast
        else:
            file_list = file_list_historical

        file_list_filtered = filter_trades_list(
            file_list=file_list,
            start_year=start_year,
            stop_year=stop_year,
        )
        trades = np.zeros(
            shape=(len(file_list_filtered), distances.shape[0], distances.shape[0])
        )
        for i in range(len(file_list_filtered)):
            trades[i] = pd.read_csv(
                file_list_filtered[i], sep=",", header=0, index_col=0, encoding="latin1"
            ).values
        trades_list.append(trades)
    # If trade data are stored by HS code
    else:
        for i in range(len(code_list)):
            code = code_list[i]
            print("\t", code)
            file_list_historical = glob.glob(f"{commodity_path}/{code}/*.csv")
            file_list_historical.sort()

            if commodity_forecast_path is not None:
                file_list_forecast = glob.glob(
                    f"{commodity_forecast_path}/{code}/*.csv"
                )
                file_list_forecast.sort()
                file_list = file_list_historical + file_list_forecast
            else:
                file_list = file_list_historical

            file_list_filtered = filter_trades_list(
                file_list=file_list, start_year=start_year
            )
            trades = np.zeros(
                shape=(len(file_list_filtered), distances.shape[0], distances.shape[0])
            )
            for i in range(len(file_list_filtered)):
                trades[i] = pd.read_csv(
                    file_list_filtered[i],
                    sep=",",
                    header=0,
                    index_col=0,
                    encoding="latin1",
                ).values
            trades_list.append(trades)

    return trades_list, file_list_filtered, code_list, commodities_available


def adjust_trade_scenario(T_ijct, scenario):
    """
    Returns adjusted trade value for an origin-destination pair based on a
    adjustment type (e.g., increase, decrease) and percentage.

    Parameters:
    -----------
    T_ijct : float
        Original value/volume of trade between origin and destination.
        of commodity c at time t.
    scenario : list
        Nested list of scenario elements, with elements ordered as: year (YYYY),
        origin ISO3 code, destination ISO3 code, adjustment type (e.g.,
        "increase", "decrease"), and adjustment percent.

    Returns:
    --------
    Adjusted trade value/volume for origin(i) - destination(j) pair at time (t)
    for commodity (c) based on scenario.

    """

    adjustment_type = scenario[0][3]
    adjustment_pct = scenario[0][4]
    if adjustment_type == "decrease":
        return T_ijct * (1 - adjustment_pct)
    if adjustment_type == "increase":
        return T_ijct * (1 + adjustment_pct)


def convert_to_binary(raster, threshold):
    """
    Returns a raster with values converted to a binary output
    based on specified threshold.

    Parameters:
    -----------
    raster : raster
        Input raster with unique values.
    threshold : float
        The value at which original values are converted to either
        zero (if below the value) or 1 (if above the value).

    Returns:
    --------
    Binary raster.

    """
    raster[raster < threshold] = 0
    raster[raster >= threshold] = 1
    return raster

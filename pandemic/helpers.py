"""
PoPS Pandemic - Simulation

Module containing all calcualtions for helper functions

Copyright (C) 2019-2020 by the authors.

Authors: Chris Jones (cmjone25 ncsu edu)

The code contained herein is licensed under the GNU General Public
License. You may obtain a copy of the GNU General Public License
Version 3 or later at the following locations:

http://www.opensource.org/licenses/gpl-license.html
http://www.gnu.org/copyleft/gpl.html
"""
import os
import glob
import pandas as pd
import numpy as np
from scipy.spatial import distance
from shapely.geometry.polygon import Polygon
from shapely.geometry.multipolygon import MultiPolygon


def distance_between(shapefile):
    """
    Returns a n x n numpy array with the the distance from each element in a
    shapefile to all other elements in that shapefile.

    Parameters
    ----------
    shapefile : geodataframe
        A geopandas dataframe of countries with crs(epsg = 4326)

    Returns
    -------
    distance : numpy array
        An n x n numpy array of distances from each location to every other
        location in kilometer

    """

    centroids = shapefile.centroid.geometry
    centroids = centroids.to_crs(epsg=3395)
    shapefile["centroid_lon"] = centroids.x
    shapefile["centroid_lat"] = centroids.y
    centroids_array = shapefile.loc[:, ["centroid_lon", "centroid_lat"]].values
    distance_array = distance.cdist(centroids_array, centroids_array, "euclidean")

    return distance_array


def locations_with_hosts(locations):
    """
    Returns a list of countries that have host species
    presence > 0%

    Parameters
    ----------
    locations : data_frame
        data frame of countries, species presence, phytosanitry capacity,
        koppen climate classifications % of total area for each class.

    Returns
    --------
    locations_list : list
        list of countries with their corresponding attributes as a series
        for countries with host species presence greater than 0%
    """

    locations_list = []
    for i in range(len(locations)):
        location = locations.iloc[i, :]
        host_pct = location["Host Percent Area"]
        if host_pct > 0:
            locations_list.append(location)

    return locations_list


def filter_trades_list(file_list, start_year):
    """
    Returns filtered list of trade data based on start
    year

    Parameters:
    -----------
    file_list : list
        List of all trade data files in a directory
    start_year : str
        Simulation start year (YYYY) used to filter
        trade data files that are prior to that year

    Returns:
    --------
    file_list_filtered: list
        List of trade data starting with simulation
        start year

    """
    for i, f in enumerate(file_list):
        date_tag = str.split(os.path.splitext(os.path.split(f)[1])[0], "_")[-1][:4]
        if int(date_tag) < int(start_year):
            file_list[i] = None
    file_list_filtered = [f for f in file_list if f is not None]

    return file_list_filtered


def create_trades_list(commodity_path, commodity_forecast_path, start_year, distances):
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
    start_year : str
        Simulation start year (YYYY) used to filter
        trade data files that are prior to that year
    distances : numpy.array
        n x n matrix of distances from one location to another where n is
        number of locations

    Returns:
    --------
    trades_list: list
        list (c) of n x n x t matrices where c is the # of commoditites,
        n is the # of locations, and t is # of time steps
    file_list_filtered : list
        list of filtered commodity (historical and forecast) file paths
    code_list : list
        list of commodity codes available in commodity directory
    commodities_available : list
        list of all commodity file paths

    """
    commodities_available = glob.glob(commodity_path + "*")
    commodities_available.sort()
    trades_list = []
    print("Loading and formatting trade data...")
    # If trade data are aggregated (i.e., summed across
    # multiple commodity codes)
    if len(commodities_available) == 1:
        print("\t", commodities_available)
        file_list_historical = glob.glob(commodity_path + "/*.csv")
        file_list_historical.sort()
        if commodity_forecast_path != None:
            file_list_forecast = glob.glob(commodity_forecast_path + "/*.csv")
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
                file_list_filtered[i], sep=",", header=0, index_col=0, encoding="latin1"
            ).values
        trades_list.append(trades)
    # If trade data are stored by HS code
    else:
        for i in range(len(commodities_available)):
            code_list = [os.path.split(f)[1] for f in commodities_available]
            code = code_list[i]
            print("\t", commodities_available[i])
            file_list_historical = glob.glob(commodity_path + f"/{code}/*.csv")
            file_list_historical.sort()

            if commodity_forecast_path != None:
                file_list_forecast = glob.glob(
                    commodity_forecast_path + f"/{code}/*.csv"
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

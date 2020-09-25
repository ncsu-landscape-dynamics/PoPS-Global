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

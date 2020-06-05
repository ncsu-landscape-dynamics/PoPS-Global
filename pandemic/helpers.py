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

import numpy as np
import geopandas as gp
from scipy.spatial import distance
import itertools
from math import cos


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

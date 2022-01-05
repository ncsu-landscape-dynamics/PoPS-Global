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

"""Module containing functions to compute climatic similarity between all
nodes used in the PoPS Global simulation.
"""

import numpy as np


def climate_similarity(origin_climates, destination_climates):
    """
    Returns the climate similarity between origin (i) and destination (j) by
    simply checking whether or not the climate type is present in both the
    origin (i) and destination (j) and summing the total area of climate classes
    in the destination (j) that are also in the origin (i).

    Parameters
    ----------
    origin_climates : array (float)
        An array with percent area for each of the Koppen climate zones for the
        origin (i)
    destination_climates : array (float)
        An array with percent area for each of the Koppen climate zones for the
        destination (j)

    Returns
    -------
    similarity : float
        Percentage of area in destination that falls within a climate class found
        in the origin

    """

    similarity = 0.00
    for clim in range(len(origin_climates)):
        if origin_climates[clim] > 0 and destination_climates[clim] > 0:
            similarity += destination_climates[clim]
    return similarity


def create_climate_similarities_matrix(array_template, countries):
    """
    Returns the climate similarities between all origins (i) and
    destinations (j)

    Parameters
    ----------
    array_template : array (float)
        n x n template matrix where n is number of locations
    countries : data frame
        data frame of nodes with species presence, phytosanitry capacity,
        % of total area for each koppen climate class

    Returns
    -------
    climate_similarities : numpy.array (float)
        n x n array of climate similarities between all
        origins (i) and destinations (j)

    """

    climate_similarities = np.zeros_like(array_template, dtype=float)

    for j in range(len(countries)):
        destination = countries.iloc[j, :]
        for i in range(len(countries)):
            origin = countries.iloc[i, :]

            origin_climates = origin.loc[
                [
                    "Af",
                    "Am",
                    "Aw",
                    "BWh",
                    "BWk",
                    "BSh",
                    "BSk",
                    "Csa",
                    "Csb",
                    "Csc",
                    "Cwa",
                    "Cwb",
                    "Cwc",
                    "Cfa",
                    "Cfb",
                    "Cfc",
                    "Dsa",
                    "Dsb",
                    "Dsc",
                    "Dsd",
                    "Dwa",
                    "Dwb",
                    "Dwc",
                    "Dwd",
                    "Dfa",
                    "Dfb",
                    "Dfc",
                    "Dfd",
                    "ET",
                    "EF",
                ]
            ]

            destination_climates = destination.loc[
                [
                    "Af",
                    "Am",
                    "Aw",
                    "BWh",
                    "BWk",
                    "BSh",
                    "BSk",
                    "Csa",
                    "Csb",
                    "Csc",
                    "Cwa",
                    "Cwb",
                    "Cwc",
                    "Cfa",
                    "Cfb",
                    "Cfc",
                    "Dsa",
                    "Dsb",
                    "Dsc",
                    "Dsd",
                    "Dwa",
                    "Dwb",
                    "Dwc",
                    "Dwd",
                    "Dfa",
                    "Dfb",
                    "Dfc",
                    "Dfd",
                    "ET",
                    "EF",
                ]
            ]

            delta_kappa_ij = climate_similarity(origin_climates, destination_climates)

            climate_similarities[j, i] = delta_kappa_ij
    return climate_similarities


def climate_similarity_origins(origins_climate_list, destination_climates):
    """
    Returns the climate similarity between the destination (j) and the initial pest
    range by summing the total area in the destination (j) with climate types that
    are present in origin nodes (e.g., native countries) at the start of simulation.

    Parameters
    ----------
    origins_climate_list : list (str)
        A list of the Koppen climate zones present in the origins of timestep 1.
    destination_climates : array (float)
        An array with percent area for each of the Koppen climate zones for the
        destination (j)

    Returns
    -------
    similarity : float
        Percentage of the total area of the destination country has climates that
        are similar to the initial pest range

    """

    similarity = 0.00
    for clim in range(len(origins_climate_list)):
        if destination_climates[clim] > 0:
            similarity += destination_climates[clim]

    return similarity


def create_climate_similarities_matrix_origins(countries, origins_climate_list):
    """
    Returns the climate similarities between all nodes (i) and origin nodes
    (e.g., native countries) at the start of simulation.

    Parameters
    ----------
    countries : data frame data frame of countries, species presence,
        phytosanitry capacity, koppen climate classifications % of total area
        for each class origins_climate_list : list (str) list of climate
        categories in areas where pest is present at timestep 1

    Returns
    -------
    climate_similarities : numpy.array (float) n x 1 array of percentage of
        climate similarities between all origins (i) and initial origins at
        start of simulation

    """
    climate_similarities = np.zeros(len(countries))

    for j in range(len(countries)):
        destination = countries.iloc[j, :]
        destination_climates = destination.loc[
            [
                "Af",
                "Am",
                "Aw",
                "BWh",
                "BWk",
                "BSh",
                "BSk",
                "Csa",
                "Csb",
                "Csc",
                "Cwa",
                "Cwb",
                "Cwc",
                "Cfa",
                "Cfb",
                "Cfc",
                "Dsa",
                "Dsb",
                "Dsc",
                "Dsd",
                "Dwa",
                "Dwb",
                "Dwc",
                "Dwd",
                "Dfa",
                "Dfb",
                "Dfc",
                "Dfd",
                "ET",
                "EF",
            ]
        ]

        delta_kappa_j = climate_similarity_origins(
            origins_climate_list, destination_climates
        )

        climate_similarities[j] = delta_kappa_j

    return climate_similarities

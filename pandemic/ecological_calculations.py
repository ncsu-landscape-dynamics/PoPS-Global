"""
PoPS Pandemic - Simulation

Module containing all calcualtions for ecological similarity

Copyright (C) 2019-2020 by the authors.

Authors: Chris Jones (cmjone25 ncsu edu)

The code contained herein is licensed under the GNU General Public
License. You may obtain a copy of the GNU General Public License
Version 3 or later at the following locations:

http://www.opensource.org/licenses/gpl-license.html
http://www.gnu.org/copyleft/gpl.html
"""

import numpy as np


def climate_similarity(origin_climates, destination_climates):
    """
    Returns the climate similarity between origin (i) and destion (j) by
    simply checking whether or not the climate type is present in both the
    origin (i) and destination (j) and summing the total area in the
    destination (j) that is also in the origin (i).

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
        What percentage of the total area of the origin country

    """

    similarity = 0.00
    for clim in range(len(origin_climates)):
        if origin_climates[clim] > 0 and destination_climates[clim] > 0:
            similarity += destination_climates[clim]

    return similarity

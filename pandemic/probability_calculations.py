""" 
Module containing all probability calculations (entry, establishment, and 
introduction) used for the pandemic simulation.
"""

import numpy as np
import scipy as sp
import math


def probability_of_entry(
    rho_i, rho_j, zeta_it, lamda_c, T_ijct, sigma_T, mu, d_ij, chi_it
):
    """
    probability_of_entry(rho_i, rho_j, zeta_it, lamda_c, T_ijct, sigma_T, mu, d_ij, chi_it)

    Returns the probability of entry given trade volume, distance, and 
    capacity between two locations. We are thinking of locations as ports or 
    countries in which international trade happens. Meant to be used in the 

    Parameters
    ----------
    rho_i : float
        The phytosanitary capacity of origin (i)
    rho_j : float
        The phytosanitary capacity of destination (j)
    zeta_it : bool
        Species presence in origin (i) at time (t)
    lamda_c : float
        The commodity importance [0,1] of commodity (c) in transporting the pest or pathogen
    T_ijct : float
        The trade volume between origin (i) and destination (j) for commodity (c) at time (t) in metric tons
    mu : float
        The mortality rate of the pest or pathogen during transport
    d_ij : int
        the distance between origin (i) and destination (j)
    chi_it : bool
        The seasonality of the pest or pathogen in its ability to be in a shipment

    Returns
    -------
    probability_of_entry : float
        The probability of a pest to entry the origin location 

    See Also
    probability_of_establishment : Calculates the probability of establishment 
    probability_of_introduction : Calculates the probability of introduction 
        from the probability_of_establishment and probability_of_entry
    """

    probability_of_entry = (
        (1 - rho_i)
        * (1 - rho_j)
        * zeta_it
        * (1 - math.exp((-1) * lamda_c * (T_ijct / sigma_T)))
        * math.exp(mu * d_ij)
        * chi_it
    )
    return probability_of_entry

"""
PoPS Global

Module containing all probability calculations (entry, establishment, and
introduction) used for the PoPS Global model.

Copyright (C) 2019-2021 by the authors.

Authors: Chris Jones (cmjone25 ncsu edu)
         Chelsey Walden-Schreiner (cawalden ncsu edu)
         Kellyn Montgomery
         Ariel Saffer

The code contained herein is licensed under the GNU General Public
License. You may obtain a copy of the GNU General Public License
Version 3 or later at the following locations:

http://www.opensource.org/licenses/gpl-license.html
http://www.gnu.org/copyleft/gpl.html
"""

import math


def probability_of_entry(
    rho_i,
    rho_j,
    zeta_it,
    lamda_c,
    T_ijct,
    min_Tc,
    max_Tc,
    mu,
    d_ij,
    chi_it,
    lamda_c_weight=None,
):
    """
    Returns the probability of entry given trade volume, distance, and
    capacity between two locations. We are thinking of locations as ports or
    countries in which international trade happens.

    Parameters
    ----------
    rho_i : float
        The phytosanitary capacity of origin (i)
    rho_j : float
        The phytosanitary capacity of destination (j)
    zeta_it : bool
        Species presence in origin (i) at time (t)
    lamda_c : float
        The commodity importance [0,1] of commodity (c) in transporting the
        pest or pathogen
    T_ijct : float
        The trade value/volume between origin (i) and destination (j) for commodity
        (c) at time (t) in dollar value or metric tons
    min_Tc : float
        Minimum trade value/volume for all origin and destination pairs for commodity
        (c) at time (t) in dollar value or metric tons
    max_Tc : float
        Minimum trade value/volume for all origin and destination pairs for commodity
        (c) at time (t) in dollar value or metric tons
    mu : float
        The mortality rate of the pest or pathogen during transport
    d_ij : int
        the distance between origin (i) and destination (j)
    chi_it : bool
        The seasonality of the pest or pathogen in its ability to be in a
        shipment

    Returns
    -------
    probability_of_entry : float
        The probability of a pest to enter the origin location

    See Also
    probability_of_establishment : Calculates the probability of establishment
    probability_of_introduction : Calculates the probability of introduction
        from the probability_of_establishment and probability_of_entry
    """

    return (
        (1 - rho_i)
        * (1 - rho_j)
        * zeta_it
        # * (1 - math.exp((-1) * lamda_c * (T_ijct - min_Tc) / (max_Tc - min_Tc)))
        * (
            1
            - math.exp(
                (-1)
                * ((1 + lamda_c_weight) * lamda_c)
                * ((T_ijct - min_Tc) / (max_Tc - min_Tc))
            )
        )
        * math.exp((-1) * mu * d_ij)
        * chi_it
    )


def probability_of_establishment(
    alpha,
    beta,
    delta_kappa_ijt,
    sigma_kappa,
    h_jt,
    sigma_h,
    phi,
    w_phi,
):
    """
    Returns the probability of establishment between origin (i) and destination
    (j) given climate similarity between (i and j), host area in (j),
    ecological distrubance in (j), and degree of polyphagy of the pest species.

    Parameters
    ----------
    alpha : float
        A parameter that allows the equation to be adapated to various discrete
        time steps
    beta : float
        A parameter that allows the equation to be adapted to various discrete
        time steps
    delta_kappa_ijt : float
        The climate dissimilarity between the origin (i) and destination (j)
        at time (t)
    sigma_kappa : float
        The climate dissimilarity normalizing constant
    h_jt : float
        The percent of area in the destination (j) that does not have
        suitable host for the pest
    sigma_h : float
        The host normalizing constant
    phi : int
        The degree of polyphagy of the pest of interest described as the number
        of host families
    w_phi : float
        The degree of polyphagy weight

    Returns
    -------
    probability_of_establishment : float
        The probability of a pest to establish in the origin location

    See Also
    probability_of_entry : Calculates the probability of entry
    probability_of_introduction : Calculates the probability of introduction
        from the probability_of_establishment and probability_of_entry
    """

    return (
        phi
        * w_phi
        * alpha
        * math.exp(
            (-1)
            * beta
            * (((delta_kappa_ijt / sigma_kappa) ** 2) + ((h_jt / sigma_h) ** 2))
        )
    )


def probability_of_introduction(
    probability_of_entry_ijct, probability_of_establishment_ijt
):
    """
    Returns the probability of introduction given a vector of
    probability_of_entry between origin (i) and destination (j) at time t
    with c commodities and a probability_of_establishment between origin (i)
    and destination (j)

    Parameters
    ----------
    probability_of_entry_ijct : float
        The probability of a pest entering destination (j) from origin (i) on
        commodity (c) at time (t)
    probability_of_establishment_ijt : float
        The probability of a pest establishing in destination (j) coming from
        origin (i) at time (t)
    Returns
    -------
    probability_of_introduction : float
        The probability of a pest being introduced in the origin (i) location
        from destination j

    See Also
    probability_of_entry : Calculates the probability of entry
    probability_of_establishment : Calculates the probability of establishment
    """

    return probability_of_entry_ijct * probability_of_establishment_ijt

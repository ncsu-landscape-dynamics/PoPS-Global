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

import time
import numpy as np
import pandas as pd

from pandemic.probability_calculations import (
    probability_of_entry,
    probability_of_establishment,
    probability_of_introduction,
)

from pandemic.helpers import location_pairs_with_host


def pandemic_single_time_step(
    trade,
    distances,
    locations,
    locations_list,
    climate_similarities,
    alpha,
    beta,
    mu,
    lamda_c,
    phi,
    sigma_epsilon,
    sigma_h,
    sigma_kappa,
    sigma_phi,
    sigma_T,
    time_step,
    season_dict,
    time_infect,
):
    """
    Returns the probability of establishment, probability of entry, and
    probability of introduction as an n x n matrices betweem every origin (i)
    and destination (j) and update species presence and the combined
    probability of presence for each origin (i) given climate similarity
    between (i and j), host area in (j), ecological distrubance in (j), degree
    of polyphagy of the pest species, trade volumes, distance, and
    phytosanitary capacity.

    Parameters
    ----------
    locations : data_frame
        data frame of countries, species presence, phytosanitry capacity,
        koppen climate classifications % of total area for each class.
    locations_list : list
        list of possible location tuples (origin, destination) pairs with
        corresponding attributes where the origin is capable of transmitting
        species propagule and the destination host species presence is greater
        than 0%
    trade : numpy.array
        list (c) of n x n x t matrices where c is the # of commoditites,
        n is the number of locations, and t is # of time steps
    distances : numpy.array
        n x n matrix of distances from one location to another where n is
        number of locations.
    climate_similarities : data_frame
        n x n matrix of climate similarity calculations between locations
        where n is the number of locations
    alpha : float
        A parameter that allows the equation to be adapated to various discrete
        time steps
    beta : float
        A parameter that allows the equation to be adapted to various discrete
        time steps
    mu : float
        The mortality rate of the pest or pathogen during transport
    lamda_c : float
        The commodity importance [0,1] of commodity (c) in transporting the
        pest or pathogen
    phi : int
        The degree of polyphagy of the pest of interest described as the number
        of host families
    sigma_kappa : float
        The climate dissimilarity normalizing constant
    sigma_h : float
        The host normalizing constant
    sigma_epsilon : float
        The ecological disturbance normalizing constant
    phi : int
        The degree of polyphagy of the pest of interest described as the number
        of host families
    sigma_phi : int
        The degree of polyphagy normalizing constant
    sigma_T : int
        The trade volume normalizing constant
    time_step : str
        string representing the name of the discrete time step (i.e., YYYYMM
        for monthly or YYYY for annual)

    Returns
    -------
    probability_of_establishment : float
        The probability of a pest to establish in the origin location

    See Also
    probability_of_entry : Calculates the probability of entry
    probability_of_introduction : Calculates the probability of introduction
        from the probability_of_establishment and probability_of_entry

    """

    establishment_probabilities = np.zeros_like(trade, dtype=float)
    entry_probabilities = np.zeros_like(trade, dtype=float)
    introduction_probabilities = np.zeros_like(trade, dtype=float)

    introduction_country = np.zeros_like(trade, dtype=float)
    locations["Probability of introduction"] = np.zeros(len(locations))
    origin_destination = pd.DataFrame(columns=["Origin", "Destination"])

    for k in range(len(locations_list)):
        # get position index of location k with known host presence
        # in data frame with all locations for selecting attributes
        # and populating output matrices
        loc_pair = locations_list[k]
        j = locations.index[locations["ISO3"] == loc_pair[1]][0]
        destination = locations.iloc[j, :]

        # check that Phytosanitary capacity data is available if not set
        # the value to 0 to remove this aspect of the equation
        if "Phytosanitary Capacity" in destination:
            rho_j = destination["Phytosanitary Capacity"]
        else:
            rho_j = 0

        # get position index of location l with known host presence
        # in data frame with all locations for selecting attributes
        # and populating output matrices
        i = locations.index[locations["ISO3"] == loc_pair[0]][0]
        origin = locations.iloc[i, :]
        # check that Phytosanitary capacity data is available if not
        # set value to 0 to remove this aspect of the equation
        if "Phytosanitary Capacity" in origin:
            rho_i = origin["Phytosanitary Capacity"]
        else:
            rho_i = 0

        T_ijct = trade[j, i]
        d_ij = distances[j, i]

        # check if time steps are annual (YYYY) or monthly (YYYYMM)
        # if monthly, parse dates to determine if species is in the
        # correct life cycle to be transported (set value to 1),
        # based on the geographic location of the origin
        # country (i.e., Northern or Southern Hemisphere)
        if len(time_step) > 4:
            if (
                origin["centroid_lat"] >= 0
                and time_step[-2:] not in season_dict["NH_season"]
            ):
                chi_it = 0
            elif (
                origin["centroid_lat"] < 0
                and time_step[-2:] not in season_dict["SH_season"]
            ):
                chi_it = 0
            else:
                chi_it = 1
        else:
            chi_it = 1

        h_jt = destination["Host Percent Area"]

        # check if species is present in origin country
        # and sufficient time has passed to faciliate transmission
        if (origin["Infective"] is not None) and (
            int(time_step) >= int(origin["Infective"])
        ):
            zeta_it = 1
            delta_kappa_ijt = climate_similarities[j, i]

            if "Ecological Disturbance" in destination:
                epsilon_jt = destination["Ecological Disturbance"]
            else:
                epsilon_jt = 0

            probability_of_entry_ijct = probability_of_entry(
                rho_i, rho_j, zeta_it, lamda_c, T_ijct, sigma_T, mu, d_ij, chi_it
            )
            probability_of_establishment_ijt = probability_of_establishment(
                alpha,
                beta,
                delta_kappa_ijt,
                sigma_kappa,
                h_jt,
                sigma_h,
                epsilon_jt,
                sigma_epsilon,
                phi,
                sigma_phi,
            )
        else:
            zeta_it = 0
            probability_of_entry_ijct = 0.0
            probability_of_establishment_ijt = 0.0

        probability_of_introduction_ijtc = probability_of_introduction(
            probability_of_entry_ijct, probability_of_establishment_ijt
        )
        entry_probabilities[j, i] = probability_of_entry_ijct
        establishment_probabilities[j, i] = probability_of_establishment_ijt
        introduction_probabilities[j, i] = probability_of_introduction_ijtc

        # decide if an introduction happens
        introduced = np.random.binomial(1, probability_of_introduction_ijtc)
        if bool(introduced):
            print("\t\t", origin["NAME"], "-->", destination["NAME"])
            print("\t\t\tProb intro: ", probability_of_introduction_ijtc)
            introduction_country[j, i] = bool(introduced)
            locations.iloc[j, locations.columns.get_loc("Presence")] = bool(introduced)
            # if no previous introductions, set infective column to current time
            # step plus period to infectivity; assumes period to infectivity is
            # given in number of years
            if locations.iloc[j, locations.columns.get_loc("Infective")] is None:
                locations.iloc[j, locations.columns.get_loc("Infective")] = str(
                    int(time_step[:4]) + time_infect
                ) + str(time_step[4:])
                print(
                    f'\t\t\t{destination["NAME"]} infective: ',
                    locations.iloc[j, locations.columns.get_loc("Infective")],
                )

            if origin_destination.empty:
                origin_destination = pd.DataFrame(
                    [[origin["NAME"], destination["NAME"]]],
                    columns=["Origin", "Destination"],
                )
            else:
                origin_destination = origin_destination.append(
                    pd.DataFrame(
                        [[origin["NAME"], destination["NAME"]]],
                        columns=["Origin", "Destination"],
                    ),
                    ignore_index=True,
                )
        else:
            introduction_country[j, i] = bool(introduced)

    # calculate combined probability of introduction for a destination
    # in a given time step
    for r in range(0, introduction_probabilities.shape[0]):
        dst = introduction_probabilities[r, :]
        combined_probability_no_introduction = np.prod(list(map(lambda x: 1 - x, dst)))
        locations.iloc[r, locations.columns.get_loc("Probability of introduction")] = (
            1 - combined_probability_no_introduction
        )

    return (
        entry_probabilities,
        establishment_probabilities,
        introduction_probabilities,
        introduction_country,
        locations,
        origin_destination,
    )


def pandemic_multiple_time_steps(
    trades,
    distances,
    climate_similarities,
    locations,
    alpha,
    beta,
    mu,
    lamda_c,
    phi,
    sigma_epsilon,
    sigma_h,
    sigma_kappa,
    sigma_phi,
    sigma_T,
    start_year,
    date_list,
):
    """
    Returns the probability of establishment, probability of entry, and
    probability of introduction as an n x n matrices betweem every origin (i)
    and destination (j) and update species presence and the combined
    probability of presence for each origin (i) given climate similarity
    between (i and j), host area in (j), ecological distrubance in (j), degree
    of polyphagy of the pest species, trade volumes, distance, and
    phytosanitary capacity.

    Parameters
    ----------
    locations : data_frame
        data frame of countries, species presence, phytosanitry capacity,
        koppen climate classifications % of total area for each class.
    trades : numpy.array
        list (c) of n x n x t matrices where c is the # of commoditites,
        n is the number of locations, and t is # of time steps
    distances : numpy.array
        n x n matrix of distances from one location to another where n is
        number of locations.
    climate_similarities : data_frame
        n x n matrix of climate similarity calculations between locations
        where n is the number of locations
    alpha : float
        A parameter that allows the equation to be adapated to various discrete
        time steps
    beta : float
        A parameter that allows the equation to be adapted to various discrete
        time steps
    mu : float
        The mortality rate of the pest or pathogen during transport
    lamda_c : float
        The commodity importance [0,1] of commodity (c) in transporting the
        pest or pathogen
    phi : int
        The degree of polyphagy of the pest of interest described as the number
        of host families
    sigma_kappa : float
        The climate dissimilarity normalizing constant
    sigma_h : float
        The host normalizing constant
    sigma_epsilon : float
        The ecological disturbance normalizing constant
    phi : int
        The degree of polyphagy of the pest of interest described as the number
        of host families
    sigma_phi : int
        The degree of polyphagy normalizing constant
    sigma_T : int
        The trade volume normalizing constant
    start_year : int
        The year in which to start the simulation
    date_list : list
        List of unique time step values (YYYY or YYYYMM)

    Returns
    -------
    probability_of_establishment : float
        The probability of a pest to establish in the origin location

    See Also
    probability_of_entry : Calculates the probability of entry
    probability_of_introduction : Calculates the probability of introduction
        from the probability_of_establishment and probability_of_entry
    """
    model_start = time.perf_counter()
    # time_steps = trades.shape[0]

    entry_probabilities = np.zeros_like(trades, dtype=float)
    establishment_probabilities = np.zeros_like(trades, dtype=float)
    introduction_probabilities = np.zeros_like(trades, dtype=float)

    introduction_countries = np.zeros_like(trades, dtype=float)
    locations["Probability of introduction"] = np.zeros(shape=len(locations))
    origin_destination = pd.DataFrame(columns=["Origin", "Destination", "Year"])

    for t in range(trades.shape[0]):
        ts_time_start = time.perf_counter()
        ts = date_list[t]
        print("TIME STEP: ", ts)
        trade = trades[t]

        if f"Host Percent Area T{t}" in locations.columns:
            locations["Host Percent Area"] = locations[f"Host Percent Area T{t}"]
        else:
            locations["Host Percent Area"] = locations["Host Percent Area"]

        locations[f"Presence {ts}"] = locations["Presence"]
        locations[f"Probability of introduction {ts}"] = locations[
            "Probability of introduction"
        ]

        if f"Phytosanitary Capacity {ts[:4]}" in locations.columns:
            locations["Phytosanitary Capacity"] = locations[
                f"Phytosanitary Capacity {ts[:4]}"
            ]
        else:
            locations["Phytosanitary Capacity"] = locations["Phytosanitary Capacity"]

        # filter locations to those where host percent area is greater
        # than 0 and therefore has potential for pest spread
        locations_list = location_pairs_with_host(locations)

        ts_out = pandemic_single_time_step(
            trade=trade,
            distances=distances,
            locations=locations,
            locations_list=locations_list,
            climate_similarities=climate_similarities,
            alpha=alpha,
            beta=beta,
            mu=mu,
            lamda_c=lamda_c,
            phi=phi,
            sigma_epsilon=sigma_epsilon,
            sigma_h=sigma_h,
            sigma_kappa=sigma_kappa,
            sigma_phi=sigma_phi,
            sigma_T=sigma_T,
            time_step=ts,
        )

        establishment_probabilities[t] = ts_out[1]
        entry_probabilities[t] = ts_out[0]
        introduction_probabilities[t] = ts_out[2]
        introduction_countries[t] = ts_out[3]
        locations = ts_out[4]
        origin_destination_ts = ts_out[5]
        origin_destination_ts["TS"] = ts
        if origin_destination.empty:
            origin_destination = origin_destination_ts
        else:
            origin_destination = origin_destination.append(
                origin_destination_ts, ignore_index=True
            )
        locations["Presence " + str(ts)] = locations["Presence"]
        locations["Probability of introduction " + str(ts)] = locations[
            "Probability of introduction"
        ]
        ts_time_end = time.perf_counter()
        print(f"\t\tloop: {round(ts_time_end - ts_time_start, 2)} seconds")

    model_end = time.perf_counter()
    print(f"model run: {round((model_end - model_start)/60, 2)} minutes")

    return (
        locations,
        entry_probabilities,
        establishment_probabilities,
        introduction_probabilities,
        origin_destination,
        introduction_countries,
    )

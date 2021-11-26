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

"""Module containing functions to run the PoPS Global simulation."""

import numpy as np
import pandas as pd

from pandemic.probability_calculations import (
    probability_of_entry,
    probability_of_establishment,
    probability_of_introduction,
)

from pandemic.helpers import location_pairs_with_host, adjust_trade_scenario


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
    sigma_h,
    sigma_kappa,
    w_phi,
    min_Tc,
    max_Tc,
    time_step,
    season_dict,
    transmission_lag_type,
    time_infect,
    gamma_shape,
    gamma_scale,
    scenario_list=None,
):
    """
    Returns the probability of establishment, probability of entry, and
    probability of introduction as n x n matrices betweem every origin (i)
    and destination (j) and updates species presence and the combined
    probability of presence for each origin (i) given climate similarity,
    host area in (j), ecological distrubance in (j), degree
    of polyphagy of the pest species, trade volumes, distance, and
    phytosanitary capacity.


    Parameters
    ----------
    locations : data_frame
        data frame of nodes with species presence, phytosanitry capacity,
        % of total area for each koppen climate class
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
        n x n or n x 1 matrix of climate similarity calculations between locations
        where n is the number of locations. May be similarity between origin-destination
        pairs or between initial origins and destinations
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
    w_phi : int
        The degree of polyphagy weight
    min_Tc : float
        The minimum value/volume of trade for the year of timestep (t) for all
        origin and desitnation pairs for commodity (c) in dollar value or metric tons.
    max_Tc : float
        The maximum value/volume of trade for the year of timestep (t) for all
        origin and desitnation pairs for commodity (c) in dollar value or metric tons.
    time_step : str
        String representing the name of the discrete time step (i.e., YYYYMM
        for monthly or YYYY for annual)
    season_dict : dict
        Dictionary of months (i.e., MM) when a pest can be transported in
        a commodity, denoted by hemisphere key (i.e.,
        {NH_season: [05, 06'], SH_season: [11, 12]})
    transmission_lag_type : str
        Type of transmission lag used in the simulation (i.e., None,
        static, or stochastic)
    time_infect : int
        Time until a node is infectious, set for static transmission lag
    gamma_shape : float
        Shape parameter for gamma distribution used in stochastic transmission
    gamma_scale : float
        Scale parameter for gamma distribution used in stochastic transmission
    scenario_list : list (optional)
        Nested list of scenarios, with elements ordered as: year (YYYY),
        origin ISO3 code, destination ISO3 code, adjustment type (e.g.,
        "increase", "decrease"), and adjustment percent.

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

        # If trade scenarios exist, check if origin-destination
        # pair has a scenario for this time step.
        # If so, adjust T_ijct according to scenario.
        if scenario_list:
            if len(time_step) == 6:
                time_step_year = int(time_step[:4])
            elif len(time_step) == 4:
                time_step_year = int(time_step)
            scenario = [
                item
                for item in scenario_list
                if item[0] == time_step_year
                and item[1] == origin["ISO3"]
                and item[2] == destination["ISO3"]
            ]
            if len(scenario) == 1:
                print(f"\tAdjusting trade for {origin['ISO3']}-{destination['ISO3']}")
                print(f"\t\tfrom: {T_ijct}")
                T_ijct = adjust_trade_scenario(T_ijct=T_ijct, scenario=scenario)
                print(f"\t\tto: {T_ijct}")

        d_ij = distances[j, i]

        # check if time steps are annual (YYYY) or monthly (YYYYMM)
        # if monthly, parse dates to determine if species is in the
        # correct life cycle to be transported (set value to 1),
        # based on the geographic location of the origin
        # node (i.e., Northern or Southern Hemisphere)
        if len(time_step) > 4:
            if origin["LAT"] >= 0 and time_step[-2:] not in season_dict["NH_season"]:
                chi_it = 0
            elif origin["LAT"] < 0 and time_step[-2:] not in season_dict["SH_season"]:
                chi_it = 0
            else:
                chi_it = 1
        else:
            chi_it = 1

        h_jt = 1 - destination["Host Percent Area"]

        # check if species is present in origin node
        # and sufficient time has passed to faciliate transmission
        if (origin["Infective"] is not None) and (
            int(time_step) >= int(origin["Infective"])
        ):
            zeta_it = 1
            if len(climate_similarities.shape) == 1:
                delta_kappa_ijt = 1 - climate_similarities[j]
            else:
                delta_kappa_ijt = 1 - climate_similarities[j, i]

            if T_ijct == 0:
                probability_of_entry_ijct = 0
            else:
                probability_of_entry_ijct = probability_of_entry(
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
                )
            probability_of_establishment_ijt = probability_of_establishment(
                alpha,
                beta,
                delta_kappa_ijt,
                sigma_kappa,
                h_jt,
                sigma_h,
                phi,
                w_phi,
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
        if len(time_step) == 4:
            introduced = np.random.binomial(12, probability_of_introduction_ijtc)
        else:
            introduced = np.random.binomial(1, probability_of_introduction_ijtc)
        if bool(introduced):
            print("\t\t", origin["NAME"], "-->", destination["NAME"])
            introduction_country[j, i] = bool(introduced)
            locations.iloc[j, locations.columns.get_loc("Presence")] = bool(introduced)

            # if no previous introductions, set infective column to current time
            # step plus period to infectivity; currently assumes period to infectivity
            # is given in number of years
            if transmission_lag_type is None:
                time_infect = 0
                if locations.iloc[j, locations.columns.get_loc("Infective")] is None:
                    locations.iloc[j, locations.columns.get_loc("Infective")] = str(
                        int(time_step[:4]) + time_infect
                    ) + str(time_step[4:])
                    print(
                        f'\t\t\t{destination["NAME"]} infective: ',
                        locations.iloc[j, locations.columns.get_loc("Infective")],
                    )
            # Static lag is a tranmission lag of a set number of time units
            if transmission_lag_type == "static":
                if locations.iloc[j, locations.columns.get_loc("Infective")] is None:
                    locations.iloc[j, locations.columns.get_loc("Infective")] = str(
                        int(time_step[:4]) + time_infect
                    ) + str(time_step[4:])
                    print(
                        f'\t\t\t{destination["NAME"]} infective: ',
                        locations.iloc[j, locations.columns.get_loc("Infective")],
                    )
            # Stochastic lag draws from a gamma distribution to determine
            # the number of time units until infectivity for each introduction
            if transmission_lag_type == "stochastic":
                time_infect = int(
                    round(np.random.gamma(gamma_shape, gamma_scale, 1)[0])
                )
                if locations.iloc[j, locations.columns.get_loc("Infective")] is None:
                    print("\t\t\tfirst intro...")
                    print("\t\t\ttime to infectious: ", time_infect)
                    locations.iloc[j, locations.columns.get_loc("Infective")] = str(
                        int(time_step[:4]) + time_infect
                    ) + str(time_step[4:])
                    print(
                        f'\t\t\t\t{destination["NAME"]} infective: ',
                        locations.iloc[j, locations.columns.get_loc("Infective")],
                    )
                else:
                    print("\t\t\treintroduction....")
                    current = int(
                        locations.iloc[j, locations.columns.get_loc("Infective")]
                    )
                    new = str(int(time_step[:4]) + time_infect) + str(time_step[4:])
                    print("\t\t\tlatest time to infectious: ", time_infect)
                    print(f"\t\t\t\torig: {current} \t new: {new}")
                    if int(new) < int(current):
                        print("\t\t\t\trevising to: ", new)
                        locations.iloc[j, locations.columns.get_loc("Infective")] = str(
                            new
                        )
                    else:
                        print("\t\t\t\tkeeping: ", current)

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
    sigma_h,
    sigma_kappa,
    w_phi,
    start_year,
    date_list,
    season_dict,
    transmission_lag_type,
    time_infect,
    gamma_shape,
    gamma_scale,
    scenario_list=None,
):
    """
    Returns the probability of establishment, probability of entry, and
    probability of introduction as n x n matrices betweem every origin (i)
    and destination (j) and updates species presence and the combined
    probability of presence for each origin (i) given climate similarity,
    host area in (j), ecological distrubance in (j), degree
    of polyphagy of the pest species, trade volumes, distance, and
    phytosanitary capacity.

    Parameters
    ----------
    locations : data_frame
        data frame of nodes with species presence, phytosanitry capacity,
        % of total area for each koppen climate class
    trades : numpy.array
        list (c) of n x n x t matrices where c is the # of commoditites,
        n is the number of locations, and t is # of time steps
    distances : numpy.array
        n x n matrix of distances from one location to another where n is
        number of locations.
    climate_similarities : data_frame
        n x n or n x 1 matrix of climate similarity calculations between locations
        where n is the number of locations. May be similarity between origin-destination
        pairs or between initial origins and destinations
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
    w_phi : int
        The degree of polyphagy normalizing constant
    start_year : int
        The year in which to start the simulation
    date_list : list
        List of unique time step values (YYYY or YYYYMM)
    season_dict : dict
        Dictionary of months (i.e., MM) when a pest can be transported in
        a commodity, separated by hemisphere (i.e.,
        {NH_season: [05, 06', SH_season: [11, 12]})
    transmission_lag_type : str
        Type of transmission lag used in the simulation (i.e., None,
        static, or stochastic)
    time_infect : int
        Time until a node is infectious, set for static transmission lag
    gamma_shape : float
        Shape parameter for gamma distribution used in stochastic transmission
    gamma_scale: float
        Scale parameter for gamma distribution used in stochastic transmission
    scenario_list : list (optional)
        Nested list of scenarios, with elements ordered as: year (YYYY),
        origin ISO3 code, destination ISO3 code, adjustment type (e.g.,
        "increase", "decrease"), and adjustment percent.

    Returns
    -------
    probability_of_establishment : float
        The probability of a pest to establish in the origin location

    See Also
    probability_of_entry : Calculates the probability of entry
    probability_of_introduction : Calculates the probability of introduction
        from the probability_of_establishment and probability_of_entry

    """

    entry_probabilities = np.zeros_like(trades, dtype=float)
    establishment_probabilities = np.zeros_like(trades, dtype=float)
    introduction_probabilities = np.zeros_like(trades, dtype=float)

    introduction_countries = np.zeros_like(trades, dtype=float)
    locations["Probability of introduction"] = np.zeros(shape=len(locations))
    origin_destination = pd.DataFrame(columns=["Origin", "Destination", "Year"])

    # Get minimum and maximum trade values for scaling
    min_Tc = np.min(trades)
    max_Tc = np.nanmax(trades)

    for t in range(trades.shape[0]):
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
            sigma_h=sigma_h,
            sigma_kappa=sigma_kappa,
            w_phi=w_phi,
            min_Tc=min_Tc,
            max_Tc=max_Tc,
            time_step=ts,
            season_dict=season_dict,
            transmission_lag_type=transmission_lag_type,
            time_infect=time_infect,
            gamma_shape=gamma_shape,
            gamma_scale=gamma_scale,
            scenario_list=scenario_list,
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

    return (
        locations,
        entry_probabilities,
        establishment_probabilities,
        introduction_probabilities,
        origin_destination,
        introduction_countries,
    )

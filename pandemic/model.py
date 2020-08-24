import pandas as pd
import numpy as np


from pandemic.probability_calculations import (
    probability_of_entry,
    probability_of_establishment,
    probability_of_introduction,
)


def pandemic(
    trade,
    distances,
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
    trade : numpy.array
        list (c) of n x n x t matrices where c is the # of commoditites,
        n is the number of locations, and t is # of time steps
    distances : numpy.array
        n x n matrix of distances from one location to another where n is
        number of locations.
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

    Returns
    -------
    probability_of_establishment : float
        The probability of a pest to establish in the origin location

    See Also
    probability_of_entry : Calculates the probability of entry
    probability_of_introduction : Calculates the probability of introduction
        from the probability_of_establishment and probability_of_entry
    """

    # time_steps = trade.shape[2]
    establishment_probabilities = np.empty_like(trade, dtype=float)
    entry_probabilities = np.empty_like(trade, dtype=float)
    introduction_probabilities = np.empty_like(trade, dtype=float)
    for j in range(len(locations)):
        destination = locations.iloc[j, :]
        # check that Phytosanitary capacity data is available if not set
        # the value to 0 to remove this aspect of the equation
        if "Phytosanitary capacity" in destination:
            rho_j = destination["Phytosanitary capacity"]
        else:
            rho_j = 0

        for i in range(len(locations)):
            origin = locations.iloc[i, :]
            # check that Phytosanitary capacity data is available if not
            # set value to 0 to remove this aspect of the equation
            if "Phytosanitary capacity" in origin:
                rho_i = origin["Phytosanitary capacity"]
            else:
                rho_i = 0
            delta_kappa_ijt = 0.4  # need to pull this in from data (TO DO)

            T_ijct = trade[i, j]
            d_ij = distances[i, j]
            chi_it = 1  # need to pull this in from data (TO DO)
            h_jt = origin["Host Percent Area"]
            if origin["Presence"] and h_jt > 0:
                zeta_it = int(origin["Presence"])
                if "Ecological Disturbance" in origin:
                    epsilon_jt = origin["Ecological Disturbance"]
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
            entry_probabilities[i, j] = probability_of_entry_ijct
            establishment_probabilities[i, j] = probability_of_establishment_ijt
            introduction_probabilities[i, j] = probability_of_introduction_ijtc

    return entry_probabilities, establishment_probabilities


# trade = np.array(
#     [
#         [
#             [[0, 500, 15], [50, 0, 10], [20, 30, 0]],
#             [[0, 500, 15], [50, 0, 10], [20, 30, 0]],
#             [[0, 500, 15], [50, 0, 10], [20, 30, 0]],
#             [[0, 500, 15], [50, 0, 10], [20, 30, 0]],
#         ],
#         [
#             [[0, 500, 15], [50, 0, 10], [20, 30, 0]],
#             [[0, 500, 15], [50, 0, 10], [20, 30, 0]],
#             [[0, 500, 15], [50, 0, 10], [20, 30, 0]],
#             [[0, 500, 15], [50, 0, 10], [20, 30, 0]],
#         ],
#     ]
# )
# trade[1, :, :, :] = trade[1, :, :, :] * 1.1

trade = np.array([[0, 500, 15], [50, 0, 10], [20, 30, 0]])
distances = np.array([[1, 5000, 105000], [5000, 1, 7500], [10500, 7500, 1]])
locations = pd.DataFrame(
    {
        "name": ["United States", "China", "Brazil"],
        "phytosanitary_compliance": [0.00, 0.00, 0.00],
        "Presence": [True, False, True],
        "Host Percent Area": [0.25, 0.50, 0.35],
    }
)

e = pandemic(
    trade=trade,
    distances=distances,
    locations=locations,
    alpha=0.2,
    beta=1,
    mu=0.0002,
    lamda_c=1,
    phi=5,
    sigma_epsilon=0.5,
    sigma_h=0.5,
    sigma_kappa=0.5,
    sigma_phi=2,
    sigma_T=20,
)

# print("Ecological" in locations)
print(np.all(e[0] >= 0) | (e[0] <= 1))
print((e[0] >= 0).all() and (e[0] <= 1).all())
print((e[1] >= 0).all() and (e[1] <= 1).all())
print((e[2] >= 0).all() and (e[2] <= 1).all())

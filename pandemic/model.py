import os 
import sys 
import glob
import pandas as pd
import numpy as np
import geopandas
from datetime import datetime

from pandemic.helpers import (
    distance_between,
    row_mode
)
from pandemic.probability_calculations import (
    probability_of_entry,
    probability_of_establishment,
    probability_of_introduction,
)
from pandemic.ecological_calculations import (
    climate_similarity
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
    time_step
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

    establishment_probabilities = np.empty_like(trade, dtype=float)
    entry_probabilities = np.empty_like(trade, dtype=float)
    introduction_probabilities = np.empty_like(trade, dtype=float)
    
    introduction_country = np.empty_like(trade, dtype=float)
    locations["Probability of introduction"] = np.empty(len(locations))
    origin_destination = pd.DataFrame(columns=['Origin', 'Destination'])
    
    for j in range(len(locations)):
        destination = locations.iloc[j, :]
        combined_probability_no_introduction = 1
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
            if "Phytosanitary Capacity" in origin:
                rho_i = origin["Phytosanitary Capacity"]
            else:
                rho_i = 0

            T_ijct = trade[j, i]
            d_ij = distances[j, i]

            # TO DO: Need to generalize -- this is for SLF 
            # Northern Hemisphere & Fall/Winter Months
            if (origin['centroid_lat'] >= 0 and time_step[-2:] in
                    ['09', '10', '11', '12', '01', '02', '03', '04']):
                chi_it = 1
            # Southern Hemisphere & Fall/Winter Months
            elif (origin['centroid_lat'] < 0 and time_step[-2:] in
                      ['04', '05', '06', '07', '08', '09', '10']):
                chi_it = 1
            else:
                chi_it = 0

            h_jt = destination["Host Percent Area"]

            if origin["Presence"] and h_jt > 0:
                zeta_it = int(origin["Presence"])

                origin_climates = origin.loc[['Af', 'Am',	'Aw',	'BWh', 'BWk', 
                                              'BSh', 'BSk', 'Csa',	'Csb', 
                                              'Csc', 'Cwa', 'Cwb', 'Cwc', 
                                              'Cfa',	'Cfb', 'Cfc',	'Dsa', 
                                              'Dsb',	'Dsc', 'Dsd',	'Dwa', 
                                              'Dwb',	'Dwc', 'Dwd',	'Dfa', 
                                              'Dfb',	'Dfc', 'Dfd',	'ET', 'EF']]

                destination_climates = destination.loc[['Af', 'Am',	'Aw',	
                                                        'BWh', 'BWk', 'BSh',
                                                        'BSk', 'Csa',	'Csb',
                                                        'Csc', 'Cwa', 'Cwb',
                                                        'Cwc', 'Cfa',	'Cfb',
                                                        'Cfc',	'Dsa', 'Dsb',
                                                        'Dsc', 'Dsd',	'Dwa', 
                                                        'Dwb',	'Dwc', 'Dwd',
                                                        'Dfa', 'Dfb',	'Dfc',
                                                        'Dfd',	'ET', 'EF']]
                
                delta_kappa_ijt = climate_similarity(
                    origin_climates, destination_climates)

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
                    sigma_phi
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
            combined_probability_no_introduction = (
                combined_probability_no_introduction * 
                (1 - probability_of_introduction_ijtc)
            )
            if bool(introduced):
                introduction_country[j, i] = bool(introduced)
                locations.iloc[j, locations.columns.get_loc("Presence")] = (
                    bool(introduced)
                )
                # print('\t', origin['NAME'], '-->', destination['NAME'])
                
                if origin_destination.empty:
                    origin_destination = pd.DataFrame([[origin['NAME'], 
                                                        destination['NAME']]], 
                                                      columns=['Origin', 
                                                               'Destination']
                                                      )
                else:
                    origin_destination = (origin_destination.append(
                        pd.DataFrame([[origin['NAME'],
                                       destination['NAME']]],
                                     columns=['Origin', 'Destination']),
                                     ignore_index=True)
                    )
            else:
                introduction_country[j, i] = bool(introduced)
        locations.iloc[j, locations.columns.get_loc("Probability of introduction")] = 1 - combined_probability_no_introduction

    return (
        entry_probabilities, 
        establishment_probabilities,
        introduction_probabilities,
        introduction_country,
        locations,
        origin_destination
    )

def pandemic_multiple_time_steps(
    trades,
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
    start_year,
    random_seed = None
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
    random_seed : int (optional)
        The number to use for initializing random values. If not provided, a new
        value will be used for every simulation and results may differ for the
        same input data and function parameters. If provided, the results of a
        simulation can be reproduced.

    Returns
    -------
    probability_of_establishment : float
        The probability of a pest to establish in the origin location

    See Also
    probability_of_entry : Calculates the probability of entry
    probability_of_introduction : Calculates the probability of introduction
        from the probability_of_establishment and probability_of_entry
    """

    time_steps = trades.shape[0]
    
    entry_probabilities = np.empty_like(trades, dtype=float)
    establishment_probabilities = np.empty_like(trades, dtype=float)
    introduction_probabilities = np.empty_like(trades, dtype=float)
    
    introduction_countries = np.empty_like(trades, dtype=float)
    locations["Probability of introduction"] = np.zeros(shape=len(locations))
    origin_destination = pd.DataFrame(columns=['Origin', 'Destination', 'Year'])
    
    ## TO DO: Adapt to dynamic annual or monthly date list
    date_list = pd.date_range(f'{str(start_year)}-01', 
                              f'{str(start_year + int(time_steps/12)-1)}-12', 
                              freq='MS').strftime('%Y%m').to_list() 
    
    for t in range(trades.shape[0]):
        ts = date_list[t]
        print('TIME STEP: ', ts)
        trade = trades[t]
        
        ##TO DO: generalize for changing host percent area
        locations["Host Percent Area"] = locations["Host Percent Area"]
        # if locations["Host Percent Area T" + str(t)] in locations.columns:
        #   locations["Host Percent Area"] = locations["Host Percent Area T" + str(t)]
        # else:
        #   locations["Host Percent Area"] = locations["Host Percent Area"]
        locations["Presence " + str(ts)] = locations['Presence']
        locations["Probability of introduction "  + str(ts)] = locations["Probability of introduction"]
        ## TO DO: increase flexibility in dynamic or static phytosanitary capacity
        # locations["Phytosanitary Capacity"] = locations ['Phytosanitary Capacity ' + ts[:4]]
        locations["Phytosanitary Capacity"] = locations["pc_mode"]

        ts_out = pandemic(
        trade=trade,
        distances=distances,
        locations=locations,
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
        time_step=ts
        )

        establishment_probabilities[t] = ts_out[1]
        entry_probabilities[t] = ts_out[0]
        introduction_probabilities[t] = ts_out[2]
        introduction_countries[t] = ts_out[3]
        locations = ts_out[4]
        origin_destination_ts = ts_out[5]
        origin_destination_ts['TS'] = ts
        if origin_destination.empty:
            origin_destination = origin_destination_ts
        else:
            origin_destination = origin_destination.append(origin_destination_ts, ignore_index=True)

    locations["Presence " + str(ts)] = locations["Presence"]
    locations["Probability of introduction "  + str(ts)] = locations["Probability of introduction"]

    return (
        locations, 
        entry_probabilities, 
        establishment_probabilities, 
        introduction_probabilities, 
        origin_destination, 
        introduction_countries, 
        date_list
    )

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

# trade = np.array([[0, 500, 15], [50, 0, 10], [20, 30, 0]])
# distances = np.array([[1, 5000, 105000], [5000, 1, 7500], [10500, 7500, 1]])
# locations = pd.DataFrame(
#     {
#         "name": ["United States", "China", "Brazil"],
#         "Phytosanitary Capacity": [0.00, 0.00, 0.00],
#         "Presence": [True, False, True],
#         "Host Percent Area": [0.25, 0.50, 0.35],
#     }
# )

data_dir = sys.argv[1]
countries = geopandas.read_file(sys.argv[2], driver='GPKG')
distances = distance_between(countries)

gdp_data = pd.read_csv(sys.argv[3], index_col = 0)
gdp_data_with_mode = row_mode(dataframe = gdp_data, 
                              mode_col_name = 'pc_mode', 
                              input_columns = gdp_data.columns[3:-1], 
                              column_prefix = 'Phytosanitary Capacity '
)
countries = countries.merge(gdp_data_with_mode, how='left', on='UN', suffixes = [None, '_y'])
gdp_dict = {
    'low': sys.argv[4],
    'mid': sys.argv[5],
    'high': sys.argv[6],
    np.nan: 0
}
countries.replace(gdp_dict, inplace=True)

## TO DO: increase flexibility to select specific years from directory
commodity_path = sys.argv[7]
file_list_historical = glob.glob(commodity_path)
file_list_historical.sort()
file_list_forecast = glob.glob(sys.argv[8])
file_list_forecast.sort()
file_list = file_list_historical + file_list_forecast

trades = np.zeros(shape = (len(file_list), 
                           distances.shape[0], 
                           distances.shape[0]))
for i in range(len(file_list)):
    trades[i] = pd.read_csv(file_list[i], 
                            sep = ",", 
                            header= 0, 
                            index_col=0, 
                            encoding='latin1').values

traded = pd.read_csv(file_list[1], 
                     sep = ",",
                     header= 0, 
                     index_col=0, 
                     encoding='latin1')

#Native range to start presence = True at T0
native_countries_list = sys.argv[9]

# Run Model for Selected Time Steps
trades = trades
distances = distances
locations = countries
prob = np.zeros(len(countries.index))
pres_ts0 = [False] *len(prob)
for country in native_countries_list:
    country_index = countries.index[countries['NAME'] == country][0]
    pres_ts0[country_index] = True
locations["Presence"] = pres_ts0

alpha = float(sys.argv[10])
beta = float(sys.argv[11])
mu = float(sys.argv[12])
lamda_c = 1
phi = float(sys.argv[13])
sigma_epsilon = 0.5
sigma_h = 1 - countries['Host Percent Area'].mean()
sigma_kappa = 1 - 0.3 #mean koppen climate matches
sigma_phi =  int(sys.argv[14])
sigma_T = np.mean(trades)
start_year = int(sys.argv[15])

random_seed = sys.argv[16] 

e = pandemic_multiple_time_steps(
    trades=trades,
    distances=distances,
    locations=locations,
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
    start_year=start_year,
    random_seed=random_seed
)

# # print("Ecological" in locations)
# print(np.all(e[0] >= 0) | (e[0] <= 1))
# print((e[0] >= 0).all() and (e[0] <= 1).all())
# print((e[1] >= 0).all() and (e[1] <= 1).all())
# print((e[2] >= 0).all() and (e[2] <= 1).all())

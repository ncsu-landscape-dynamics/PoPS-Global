import os 
import sys
import glob
import json
import time
import numpy as np
import pandas as pd
import geopandas
from datetime import datetime
from shapely.geometry.polygon import Polygon
from shapely.geometry.multipolygon import MultiPolygon

from pandemic.helpers import (
    distance_between,
    locations_with_hosts
)
from pandemic.probability_calculations import (
    probability_of_entry,
    probability_of_establishment,
    probability_of_introduction,
)
from pandemic.ecological_calculations import (
    climate_similarity
)
from pandemic.output_files import (
    create_model_dirs,
    save_model_output,
    agg_prob,
    get_feature_cols,
    create_feature_dict,
    add_dict_to_geojson,
    aggregate_monthly_output_to_annual
)


def pandemic(
    trade,
    distances,
    locations,
    locations_list,
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
    locations_list : list
        list of locations with corresponding attributes where host 
        species presence is greater than 0% 
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
    origin_destination = pd.DataFrame(columns=['Origin', 'Destination'])
    
    for k in range(len(locations_list)):
        # get position index of location k with known host presence
        # in data frame with all locations for selecting attributes
        # and populating output matrices  
        j = locations.index[locations['UN'] == locations_list[k]['UN']][0]
        destination = locations.iloc[j, :]
        combined_probability_no_introduction = 1
        # check that Phytosanitary capacity data is available if not set
        # the value to 0 to remove this aspect of the equation
        if "Phytosanitary Capacity" in destination:
            rho_j = destination["Phytosanitary Capacity"]
        else:
            rho_j = 0

        for l in range(len(locations_list)):
            # get position index of location l with known host presence
            # in data frame with all locations for selecting attributes
            # and populating output matrices
            i = locations.index[locations['UN'] == locations_list[l]['UN']][0]
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
            # if monthly, parse dates to determine if species is in the correct life cycle
            # to be transported (set value to 1), based on the geographic location of the origin
            # country (i.e., Northern or Southern Hemisphere)
            if len(time_step) > 4:
                if (origin['centroid_lat'] >= 0 and time_step[-2:] not in season_dict['NH_season']):
                    chi_it = 0
                elif (origin['centroid_lat'] < 0 and time_step[-2:] not in season_dict['SH_season']):
                    chi_it = 0
                else:
                    chi_it = 1
            else:
                chi_it = 1

            h_jt = destination["Host Percent Area"]

            if origin["Presence"] and h_jt > 0:
                zeta_it = int(origin["Presence"])
                
                delta_kappa_ijt = climate_similarities[j, i]

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
                print('\t', origin['NAME'], '-->', destination['NAME'])
                
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
    date_list
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
    time_steps = trades.shape[0]

    entry_probabilities = np.zeros_like(trades, dtype=float)
    establishment_probabilities = np.zeros_like(trades, dtype=float)
    introduction_probabilities = np.zeros_like(trades, dtype=float)
    
    introduction_countries = np.zeros_like(trades, dtype=float)
    locations["Probability of introduction"] = np.zeros(shape=len(locations))
    origin_destination = pd.DataFrame(columns=['Origin', 'Destination', 'Year'])
    
    for t in range(trades.shape[0]):
        ts_time_start = time.perf_counter()
        ts = date_list[t]
        print('TIME STEP: ', ts)
        trade = trades[t]
        
        if f"Host Percent Area T{t}" in locations.columns:
            locations["Host Percent Area"] = locations[f"Host Percent Area T{t}"]
        else:
            locations["Host Percent Area"] = locations["Host Percent Area"]
        
        locations[f"Presence {ts}"] = locations['Presence']
        locations[f"Probability of introduction {ts}"] = locations["Probability of introduction"]

        if f"Phytosanitary Capacity {ts[:4]}" in locations.columns:
            locations["Phytosanitary Capacity"] = locations[f"Phytosanitary Capacity {ts[:4]}"]
        else:
            locations["Phytosanitary Capacity"] = locations["pc_mode"]

        # filter locations to those where host percent area is greater
        # than 0 and therefore with potential for pest spread
        locations_list = locations_with_hosts(locations)

        ts_out = pandemic(
            trade=trade,
            distances=distances,
            locations=locations,
            locations_list=locations_list,
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
        ts_time_end = time.perf_counter()
        print(f'\t\tloop: {round(ts_time_end - ts_time_start, 2)} seconds')
    locations["Presence " + str(ts)] = locations["Presence"]
    locations["Probability of introduction "  + str(ts)] = locations["Probability of introduction"]
    model_end = time.perf_counter()
    print(f'model run: {round((model_end - model_start)/60, 2)} minutes')

    return (
        locations, 
        entry_probabilities, 
        establishment_probabilities, 
        introduction_probabilities, 
        origin_destination, 
        introduction_countries
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


path_to_config_json = sys.argv[1]

with open(path_to_config_json) as json_file:
    data = json.load(json_file)

data_dir = data['data_dir']
countries_path = data['countries_path']
phyto_path = data['phyto_path']
phyto_low = data['phyto_low']
phyto_mid = data['phyto_mid']
phyto_high = data['phyto_high']
commodity_path = data['commodity_path']
commodity_forecast_path = data['commodity_forecast_path']
native_countries_list = data['native_countries_list']
season_dict = data['season_dict']
alpha = data['alpha']
beta = data['beta']
mu = data['mu']
lamda_c = data['lamda_c']
phi = data['phi']
sigma_epsilon = data['sigma_epsilon']
sigma_phi = data['sigma_phi']
start_year = data['start_year']
random_seed = data['random_seed']
out_dir = data['out_dir']
columns_to_drop = data['columns_to_drop']

data_dir = data_dir
countries = geopandas.read_file(
    countries_path,
    driver='GPKG')
distances = distance_between(countries)
phyto_data = pd.read_csv(phyto_path, index_col = 0)
phyto_year_cols = phyto_data.columns[3:].to_list()

phyto_data['pc_mode'] = phyto_data[phyto_year_cols].mode(axis=1)[0]

phyto_data.columns = (np.where(
    phyto_data.columns.isin(phyto_year_cols),
    'Phytosanitary Capacity ' + phyto_data.columns,
    phyto_data.columns
    )
)

countries = countries.merge(phyto_data, how='left', on='UN', suffixes = [None, '_y'])
phyto_dict = {
    'low': phyto_low,
    'mid': phyto_mid,
    'high': phyto_high,
    np.nan: 0
}
countries.replace(phyto_dict, inplace=True)

# Read in trade data 
file_list_historical = glob.glob(commodity_path + '/*.csv')
file_list_historical.sort()
if commodity_forecast_path != None:
    file_list_forecast = glob.glob(commodity_forecast_path + '/*.csv')
    file_list_forecast.sort()
    file_list = file_list_historical + file_list_forecast
else:
    file_list = file_list_historical

# Filter list based on selected start year
for i, f in enumerate(file_list):
    date_tag = str.split(os.path.splitext(os.path.split(f)[1])[0], '_')[-1][:4]
    if int(date_tag) < int(start_year):
        file_list[i] = None
file_list_filtered = [f for f in file_list if f is not None]

# Create list of unique time steps from trade data 
date_list = []
for f in file_list_filtered:
    fn = os.path.split(f)[1]
    ts = str.split(os.path.splitext(fn)[0], '_')[-1]
    date_list.append(ts)
date_list.sort()

trades = np.zeros(shape = (len(file_list_filtered), 
                           distances.shape[0], 
                           distances.shape[0]))
for i in range(len(file_list_filtered)):
    trades[i] = pd.read_csv(file_list_filtered[i], 
                            sep = ",", 
                            header= 0, 
                            index_col=0, 
                            encoding='latin1').values

traded = pd.read_csv(file_list_filtered[1], 
                     sep = ",",
                     header= 0, 
                     index_col=0, 
                     encoding='latin1')

# Create an n x n array of climate similarity calculations
climate_similarities = np.zeros_like(traded, dtype=float)

for j in range(len(countries)):
  destination = countries.iloc[j, :]
  for i in range(len(countries)):
    origin = countries.iloc[i, :]
    
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
                
    delta_kappa_ij = climate_similarity(
        origin_climates, destination_climates)
    
    climate_similarities[j, i] = delta_kappa_ij

# Run Model for Selected Time Steps
trades = trades
print('Number of time steps: ', trades.shape[0])
distances = distances
locations = countries
prob = np.zeros(len(countries.index))
pres_ts0 = [False] *len(prob)
for country in native_countries_list:
    country_index = countries.index[countries['NAME'] == country][0]
    pres_ts0[country_index] = True
locations["Presence"] = pres_ts0

sigma_h = 1 - countries['Host Percent Area'].mean()
sigma_kappa = 1 - 0.3 # mean koppen climate matches, TO DO: automate
sigma_T = np.mean(trades)

np.random.seed(random_seed)

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
    date_list=date_list
)


# # print("Ecological" in locations)
# print(np.all(e[0] >= 0) | (e[0] <= 1))
# print((e[0] >= 0).all() and (e[0] <= 1).all())
# print((e[1] >= 0).all() and (e[1] <= 1).all())
# print((e[2] >= 0).all() and (e[2] <= 1).all())
run_num = sys.argv[2]
run_iter = sys.argv[3]

arr_dict = {'prob_entry': 'probability_of_entry',
           'prob_intro': 'probability_of_introduction',
           'prob_est': 'probability_of_establishment',
           'country_introduction': 'country_introduction'}
outpath = out_dir + f'/run{run_num}/iter{run_iter}/'

create_model_dirs(
    outpath = outpath,
    output_dict=arr_dict
    )

full_out_df = save_model_output(
    model_output_object = e,
    columns_to_drop = columns_to_drop,
    example_trade_matrix = traded,
    outpath = outpath,
    date_list = date_list
    )

aggregate_monthly_output_to_annual(
    formatted_geojson = full_out_df,
    outpath = outpath
    )

main_model_output = e[0]
final_presence_col = sorted(
        [c for c in main_model_output.columns if c.startswith('Presence')]
        )[-1]
meta = {}
meta['PARAMETERS'] = []
meta['PARAMETERS'].append({
        'alpha': str(alpha),
        'beta': str(beta),
        'mu': str(mu),
        'lamda_c': str(lamda_c),
        'phi': str(phi),
        'sigma_epsilon': str(sigma_epsilon),
        'sigma_h': str(sigma_h),
        'sigma_kappa': str(sigma_kappa),
        'sigma_phi': str(sigma_phi),
        'sigma_T': str(sigma_T),
        'start_year': str(start_year),
        'random_seed': str(random_seed)
    })
meta['NATIVE_COUNTRIES_T0'] = native_countries_list 
meta['COMMODITIES'] = commodity_path
meta['FORECASTED'] = commodity_forecast_path
meta['PHYTOSANITARY_CAPACITY_WEIGHTS'] = phyto_dict
meta['TOTAL COUNTRIES INTRODUCTED'] = str(main_model_output[final_presence_col].value_counts()[1] - len(native_countries_list))

with open(f'{outpath}/run{run_num}_meta.txt', 'w') as file:
    json.dump(meta, file, indent=4)



#%%# 
import os 
import sys
import glob
import json
import numpy as np
import pandas as pd
import geopandas
from datetime import datetime
from shapely.geometry.polygon import Polygon
from shapely.geometry.multipolygon import MultiPolygon
import multiprocessing as mp
#%%#
sys.path.append('C:/Users/cwald/Documents/GitHub/Pandemic_Model/')
from pandemic.helpers import (
    distance_between
)
from pandemic.probability_calculations import (
    probability_of_entry,
    probability_of_establishment,
    probability_of_introduction,
)
from pandemic.ecological_calculations import (
    climate_similarity
)
from pandemic.config import (
    data_dir,
    countries_path,
    gdp_path,
    gdp_low,
    gdp_mid,
    gdp_high,
    commodity_path,
    commodity_forecast_path,
    native_countries_list,
    alpha,
    beta,
    mu, 
    lamda_c,
    phi, 
    sigma_epsilon,
    sigma_phi,
    start_year,
    random_seed,
    run_num,
    out_dir,
    columns_to_drop
)
#%%#
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

def create_model_dirs(
    outpath, 
    output_dict
    ):
    """
    Creates directory and folders for model output files. 

    Parameters
    ----------
    run_num : Integer
        Number identifying the model run
    outpath : String
        Absolute path of directory where model output are saved
    output_dict : Dictionary
        Key-value pairs identifying the object name and folder name
        of model output components. 

    Returns
    -------
    None

    """

    os.makedirs(outpath, exist_ok = True)
    
    for key in output_dict.keys():
        os.makedirs(outpath + key, exist_ok = True)
        print(outpath + key)

def save_model_output(
    model_output_object, 
    columns_to_drop,
    example_trade_matrix, 
    outpath
    ):
    """
    Saves model output, including probabilities for entry, establishment,
    and introduction. Full forecast dataframe, origin-destination pairs,
    and list of time steps formatted as YYYYMM. 

    Parameters
    ----------
    model_output_object : numpy array
        List of 7 n x n arrays created by running pandemic model, ordered as 
        1) full forecast dataframe; 2) probability of entry; 
        3) probability of establishment; 4) probability of introduction;
        5) origin - destination pairs; 6) list of countries where pest is
        predicted to be introduced; and 7) list of time steps used in the 
        model formatted as YearMonth (i.e., YYYYMM). 

    columns_to_drop : list
        List of columns used or created by the model that are to be dropped
        from the final output (e.g., Koppen climate classifications). 

    example_trade_matrix : numpy array
        Array of trade data from one time step as example to format
        output dataframe columns and indices. 

    outpath : string
        String specifying absolute path of output directory

    Returns
    -------
    out_df : geodataframe
        Geodataframe of model outputs
    date_list_out : list
        List of time steps used in the model formatted 
        as YearMonth (i.e., YYYYMM)

    """
    
    model_output_df = model_output_object[0] 
    prob_entry = model_output_object[1]
    prob_est = model_output_object[2] 
    prob_intro = model_output_object[3]
    origin_dst = model_output_object[4] 
    country_intro = model_output_object[5]
    date_list_out = model_output_object[6]
    
    out_df = model_output_df.drop(columns_to_drop, axis=1)
    out_df["geometry"] = [MultiPolygon([feature]) if type(feature) == Polygon 
                          else feature for feature in out_df["geometry"]]
    out_df.to_file(outpath + f'pandemic_output.geojson', driver='GeoJSON')

    origin_dst.to_csv(outpath + f'origin_destination.csv')
    
    for i in range(0, len(date_list_out)):
        ts = date_list_out[i]
        
        pro_entry_pd = pd.DataFrame(prob_entry[i])
        pro_entry_pd.columns = example_trade_matrix.columns
        pro_entry_pd.index = example_trade_matrix.index
        pro_entry_pd.to_csv(outpath 
                            + f"prob_entry/probability_of_entry_{str(ts)}.csv", 
                            float_format='%.2f', 
                            na_rep="NAN!")
        
        pro_intro_pd = pd.DataFrame(prob_intro[i])
        pro_intro_pd.columns = example_trade_matrix.columns
        pro_intro_pd.index = example_trade_matrix.index
        pro_intro_pd.to_csv(outpath 
                            + f"prob_intro/probability_of_introduction_{str(ts)}.csv", 
                            float_format='%.2f', 
                            na_rep="NAN!")
        
        pro_est_pd = pd.DataFrame(prob_est[i])
        pro_est_pd.columns = example_trade_matrix.columns
        pro_est_pd.index = example_trade_matrix.index
        pro_est_pd.to_csv(outpath 
                          + f"prob_est/probability_of_establishment_{str(ts)}.csv", 
                          float_format='%.2f', 
                          na_rep="NAN!")
        
        country_int_pd = pd.DataFrame(country_intro[i])
        country_int_pd.columns = example_trade_matrix.columns
        country_int_pd.index = example_trade_matrix.index
        country_int_pd.to_csv(outpath 
                              + f"country_introduction/country_introduction_{str(ts)}.csv", 
                              float_format='%.2f', 
                              na_rep="NAN!")
    
    return out_df, date_list_out

def cumulative_prob(row, column_list):
   non_neg = []
   for i in range(0, len(column_list)):
     if row[column_list[i]] > 0.:
       non_neg.append(row[column_list[i]])
   sub_list = list(map(lambda x: 1 - x, non_neg))
   prod_out = np.prod(sub_list)
   final_prob = 1 - prod_out
   return final_prob

def get_feature_cols(geojson_obj, feature_chars):
    feature_cols = [c for c in geojson_obj.columns if 
                    c.startswith(feature_chars)]
    feature_cols_monthly = [c for c in feature_cols if 
                            len(c.split(' ')[-1]) > 5]
    feature_cols_annual = [c for c in feature_cols if 
                           c not in feature_cols_monthly]
    
    return feature_cols, feature_cols_monthly, feature_cols_annual   

def create_feature_dict(geojson_obj, column_list, chars_to_strip):
    d = geojson_obj[column_list].to_dict('index')
    for key in d.keys():
        d[key] = {k.strip(chars_to_strip): v for k, v in d[key].items()}
    
    return d

def add_dict_to_geojson(geojson_obj, new_col_name, dictionary_obj):
    geojson_obj[new_col_name] = geojson_obj.index.map(dictionary_obj)
    
    return geojson_obj

def aggregate_monthly_output_to_annual(formatted_geojson, outpath):
  presence_cols = [c for c in formatted_geojson.columns 
                   if c.startswith('Presence')]
  prob_intro_cols = [c for c in formatted_geojson.columns 
                     if c.startswith('Probability of introduction')]
  annual_ts_list = sorted(set([y.split(' ')[-1][:4] 
                               for y in prob_intro_cols]))
  for year in annual_ts_list:
    prob_cols = [c for c in prob_intro_cols if str(year) in c]
    formatted_geojson[f'Agg Prob Intro {year}'] = (
        formatted_geojson.apply(lambda row: 
                                cumulative_prob(row = row,
                                                column_list = prob_cols),
                                axis=1)
    )
    formatted_geojson[f'Presence {year}'] = formatted_geojson[f'Presence {year}12']
    
  formatted_geojson.to_file(outpath + f'pandemic_output_aggregated.geojson', 
                            driver='GeoJSON')
  out_csv = pd.DataFrame(formatted_geojson)
  out_csv.drop(['geometry'], axis=1, inplace=True)
  out_csv.to_csv(outpath + f'pandemic_output_aggregated.csv', 
                float_format='%.2f', 
                na_rep="NAN!")
  presence_cols_monthly =  [c for c in presence_cols 
                            if len(c.split(' ')[-1]) > 5]
  presence_cols_annual = [c for c in presence_cols 
                          if c not in presence_cols_monthly]
  agg_prob_cols_annual = [c for c in formatted_geojson.columns 
                          if c.startswith('Agg')]
  
  presence_d = create_feature_dict(geojson_obj = formatted_geojson,
                                   column_list = presence_cols_annual,
                                   chars_to_strip = 'Presence ')
  agg_prob_d = create_feature_dict(geojson_obj = formatted_geojson,
                                   column_list = agg_prob_cols_annual,
                                   chars_to_strip = 'Agg Prob Intro ')
  new_gdf = add_dict_to_geojson(geojson_obj = formatted_geojson,
                              new_col_name = 'Presence',
                              dictionary_obj = presence_d)
  new_gdf = add_dict_to_geojson(geojson_obj = new_gdf,
                              new_col_name = 'Agg Prob Intro',
                              dictionary_obj = agg_prob_d)
  cols_to_drop = [c for c in new_gdf.columns if
                  c in presence_cols_monthly or
                  c.startswith('Probability')]

  sm_gdf = new_gdf.drop(cols_to_drop, axis=1)
  sm_gdf.to_file(outpath + f'pandemic_output_aggregated_select.geojson', 
                  driver='GeoJSON')
  sm_csv = pd.DataFrame(sm_gdf)
  sm_csv.drop(['geometry'], axis=1, inplace=True)
  sm_csv.to_csv(outpath + f'pandemic_output_aggregated_select.csv', 
                float_format='%.2f', 
                na_rep="NAN!")

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
#%%#
run_iter = 0
data_dir = data_dir
countries = geopandas.read_file(
    countries_path,
    driver='GPKG')
distances = distance_between(countries)
gdp_data = pd.read_csv(gdp_path, index_col = 0)
gdp_year_cols = gdp_data.columns[3:].to_list()

gdp_data['pc_mode'] = gdp_data[gdp_year_cols].mode(axis=1)[0]

gdp_data.columns = (np.where(
    gdp_data.columns.isin(gdp_year_cols),
    'Phytosanitary Capacity ' + gdp_data.columns,
    gdp_data.columns
    )
)

countries = countries.merge(gdp_data, how='left', on='UN', suffixes = [None, '_y'])
gdp_dict = {
    'low': gdp_low,
    'mid': gdp_mid,
    'high': gdp_high,
    np.nan: 0
}
countries.replace(gdp_dict, inplace=True)

## TO DO: increase flexibility to select specific years from directory
file_list_historical = glob.glob(commodity_path + '/*.csv')
file_list_historical.sort()
file_list_forecast = glob.glob(commodity_forecast_path + '/*.csv')
file_list_forecast.sort()
file_list = file_list_historical + file_list_forecast
print('Number of time steps: ', len(file_list))
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

sigma_h = 1 - countries['Host Percent Area'].mean()
sigma_kappa = 1 - 0.3 # mean koppen climate matches
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
    random_seed=random_seed
)

# # print("Ecological" in locations)
# print(np.all(e[0] >= 0) | (e[0] <= 1))
# print((e[0] >= 0).all() and (e[0] <= 1).all())
# print((e[1] >= 0).all() and (e[1] <= 1).all())
# print((e[2] >= 0).all() and (e[2] <= 1).all())


# %%
arr_dict = {'prob_entry': 'probability_of_entry',
           'prob_intro': 'probability_of_introduction',
           'prob_est': 'probability_of_establishment',
           'country_introduction': 'country_introduction'}
outpath = out_dir + f'/run{run_num}/{run_iter}/'
create_model_dirs(
    outpath = outpath,
    output_dict=arr_dict
    )

#%%#
full_out_df, date_list_out = save_model_output(
    model_output_object = e,
    columns_to_drop = columns_to_drop,
    example_trade_matrix = traded,
    outpath = outpath
    )

aggregate_monthly_output_to_annual(
    formatted_geojson = full_out_df,
    outpath = outpath
    )

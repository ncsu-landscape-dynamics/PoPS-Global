#import necessary packages
import os
import sys
import pandas as pd
import numpy as np
import glob
from scipy.spatial import distance
import math
import geopandas
from shapely.geometry import Polygon, MultiPolygon 
import warnings
#import matplotlib.pyplot as plt

## Packages for use in CoLab
# Import PyDrive and associated libraries.
# This only needs to be done once per notebook.
# from pydrive.auth import GoogleAuth
# from pydrive.drive import GoogleDrive
# from google.colab import auth
# from oauth2client.client import GoogleCredentials

warnings.filterwarnings("ignore")

# # **Directory Path(s)**
#data_dir = 'G:/Shared drives/APHIS  Projects/Pandemic/Data/'
print('LENGTH SYS.ARGV: ', len(sys.argv))
ii = 0
for i in sys.argv:
    print(f'{ii}: {i}')
    ii +=1
data_dir = str(sys.argv[1])

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


def distance_between(shapefile):
    """
    Returns a n x n numpy array with the the distance from each element in a
    shapefile to all other elements in that shapefile.

    Parameters
    ----------
    shapefile : geodataframe
        A geopandas dataframe of countries with crs(epsg = 4326)

    Returns
    -------
    distance : numpy array
        An n x n numpy array of distances from each location to every other
        location in kilometer

    """

    centroids = shapefile.centroid.geometry
    centroids = centroids.to_crs(epsg=3395)
    shapefile["centroid_lon"] = centroids.x
    shapefile["centroid_lat"] = centroids.y
    centroids_array = shapefile.loc[:, ["centroid_lon", "centroid_lat"]].values
    distance_array = distance.cdist(centroids_array, 
                                    centroids_array, 
                                    "euclidean")
    distance_array = distance_array/1000
    
    return distance_array


def probability_of_entry(
        rho_i, rho_j, zeta_it, lamda_c, T_ijct, sigma_T, mu, d_ij, chi_it
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
        The trade volume between origin (i) and destination (j) for commodity
        (c) at time (t) in metric tons
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
        * (1 - math.exp((-1) * lamda_c * (T_ijct / sigma_T)))
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
    epsilon_jt,
    sigma_epsilon,
    phi,
    sigma_phi,
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
    delta_kappa_ijt :float
        The climate dissimilarity between the origin (i) and destination (j)
        at time (t)
    sigma_kappa : float
        The climate dissimilarity normalizing constant
    h_jt : float
        The percent of area in the destination (j) that has suitable host for
        the pest
    sigma_h : float
        The host normalizing constant
    epsilon_jt : float
        The ecological disturbance index of destination (j) at time (t)
    sigma_epsilon : float
        The ecological disturbance normalizing constant
    phi : int
        The degree of polyphagy of the pest of interest described as the number
        of host families
    sigma_phi : int
        The degree of polyphagy normalizing constant

    Returns
    -------
    probability_of_establishment : float
        The probability of a pest to establish in the origin location

    See Also
    probability_of_entry : Calculates the probability of entry
    probability_of_introduction : Calculates the probability of introduction
        from the probability_of_establishment and probability_of_entry
    """

    return alpha * math.exp(
        (-1)
        * beta
        * (
            ((1 - delta_kappa_ijt) / sigma_kappa) ** 2
            + ((1 - h_jt) / sigma_h) ** 2
            + ((1 - epsilon_jt) / sigma_epsilon) ** 2
            + (phi / sigma_phi) ** (-2)
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
    probability of introduction as an n x n matrices between every origin (i)
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
    time_step: str
      The year-month combination of the time step. 
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

    entry_probabilities = np.empty_like(trade, dtype=float)
    establishment_probabilities = np.empty_like(trade, dtype=float)
    introduction_probabilities = np.empty_like(trade, dtype=float)
    
    introduction_country = np.empty_like(trade, dtype=float)
    locations["Probability of introduction"] = np.empty(len(locations))
    origin_destination = pd.DataFrame(columns=['Origin', 'Destination'])

    
    for j in range(len(locations)):
        destination = locations.iloc[j, :]
        combined_probability_no_introduction = 1
    
        # check that Phytosanitary capacity data is available if not set
        # the value to 0 to remove this aspect of the equation
        if "Phytosanitary Capacity" in destination:
            rho_j = destination["Phytosanitary Capacity"]
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
            
            ## Need to generalize -- this is for SLF; add column for seasonality flag
            #Northern Hemisphere & Fall/Winter Months
            if (origin['centroid_lat'] >= 0 and time_step[-2:] in 
                ['09', '10', '11', '12', '01', '02', '03', '04']):
                chi_it = 1
            #Southern Hemisphere & Fall/Winter Months
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
                
                delta_kappa_ijt = climate_similarity(origin_climates, destination_climates)

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
            
            entry_probabilities[j, i] = probability_of_entry_ijct
            establishment_probabilities[j, i] = probability_of_establishment_ijt
            introduction_probabilities[j, i] = probability_of_introduction_ijtc
            
            # decide if an introduction happens
            introduced = np.random.binomial(1, probability_of_introduction_ijtc)
            combined_probability_no_introduction = combined_probability_no_introduction * (1 - probability_of_introduction_ijtc)
            
            if bool(introduced):
                introduction_country[j, i] = bool(introduced)
                locations.iloc[j, locations.columns.get_loc("Presence")] = bool(introduced)
                print('\t', origin['NAME'], '-->', destination['NAME'])
                
                if origin_destination.empty:
                    origin_destination = pd.DataFrame([[origin['NAME'], 
                                                        destination['NAME']]], 
                                                      columns=['Origin', 'Destination'])
                else:
                    origin_destination = origin_destination.append(pd.DataFrame([[origin['NAME'], 
                                                                                  destination['NAME']]], 
                                                                                columns=['Origin', 'Destination']), 
                                                                   ignore_index=True)
            else:
                introduction_country[j, i] = bool(introduced)

        locations.iloc[j, locations.columns.get_loc("Probability of introduction")] = 1 - combined_probability_no_introduction

    return entry_probabilities, establishment_probabilities, introduction_probabilities, introduction_country, locations, origin_destination


def pandemic2(
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
    start_year
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
    
    #TO DO: only works for full year - will need to update to accomodate partial year data
    date_list = pd.date_range(f'{str(start_year)}-01', 
                              f'{str(start_year + int(time_steps/12)-1)}-12', 
                              freq='MS').strftime("%Y%m").tolist()

    for t in range(trades.shape[0]):
        ts = date_list[t]
        
        print('TIME STEP: ', ts)
        trade = trades[t]
        
        ##TO DO: generalize for changing host percent area, static phytosanitary capacity, etc
        locations["Host Percent Area"] = locations["Host Percent Area"]
        # if locations["Host Percent Area T" + str(t)] in locations.columns:
        #   locations["Host Percent Area"] = locations["Host Percent Area T" + str(t)]
        # else:
        #   locations["Host Percent Area"] = locations["Host Percent Area"]
        locations["Presence " + str(ts)] = locations['Presence']
        locations["Probability of introduction "  + str(ts)] = locations["Probability of introduction"]
        locations["Phytosanitary Capacity"] = locations['pc_mode']
        #locations["Phytosanitary Capacity"] = locations ['Phytosanitary Capacity ' + ts[:4]]

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
        time_step=ts)

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

    return locations, entry_probabilities, establishment_probabilities, introduction_probabilities, origin_destination, introduction_countries, date_list


# # **Data**

# countries = geopandas.read_file(data_dir + '/slf_model/inputs/countries4.gpkg',
#                                 driver = 'GPKG')
countries = geopandas.read_file(data_dir + str(sys.argv[2]),
                                driver = 'GPKG')

# get distance n x n matrix
distances = distance_between(countries)
print(f'countries: {countries.shape}\tdistances: {distances.shape}')


#gdp = pd.read_csv(data_dir + '/GDP/2000_2019_GDP_perCapita/gdp_perCapita_binned.csv', index_col =0)
gdp = pd.read_csv(data_dir + sys.argv[3], index_col =0)
gdp['pc_mode'] = gdp[['2000', '2001', '2002', '2003', '2004',
       '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013',
       '2014', '2015', '2016', '2017', '2018', '2019']].mode(axis=1)[0]
year_cols = gdp.columns[3:-1]
gdp.columns = np.where(gdp.columns.isin(year_cols), 
                       'Phytosanitary Capacity ' + gdp.columns, 
                       gdp.columns)

countries = countries.merge(gdp, how='left', on='UN', suffixes = [None, '_y'])


gdp_dict = {'low': float(sys.argv[4]),
            'mid': float(sys.argv[5]),
            'high': float(sys.argv[6]),
            np.nan: 0}

countries.replace(gdp_dict,
                  inplace=True)



## path to directory  
#directory_path = data_dir + "/slf_model/inputs/monthly/select_commodities/*.csv" #codes 6801-6804
directory_path = data_dir + str(sys.argv[7])
file_list_historical = glob.glob(directory_path)
file_list_historical.sort()

#file_list_forecast = glob.glob(data_dir + '/slf_model/inputs/monthly/forecast/static/*.csv')
if str(sys.argv[8]) == 'forecast':
    file_list_forecast = glob.glob(data_dir + '/slf_model/inputs/monthly/forecast/static/*.csv')
    file_list_forecast.sort()
    file_list = file_list_historical + file_list_forecast
elif str(sys.argv[8]) == 'historical':
    file_list = file_list_historical

trades = np.zeros(shape = (len(file_list), distances.shape[0], distances.shape[0]))
for i in range(len(file_list)):
    trades[i] = pd.read_csv(file_list[i], sep = ",", header= 0, index_col=0, encoding='latin1').values

traded = pd.read_csv(file_list[1], sep = ",",header= 0, index_col=0, encoding='latin1')
trades.shape


#Native range to start presence = True at T0
china_index = countries.index[countries['NAME'] == 'China'][0]
viet_nam_index = countries.index[countries['NAME'] == 'Viet Nam'][0]
india_index = countries.index[countries['NAME'] == 'India'][0]
native_countries_list = ['China', 'Viet Nam', 'India']

#Known Introductions 
skorea_index = countries.index[countries['NAME'] == 'Korea, Republic of'][0]
japan_index = countries.index[countries['NAME'] == 'Japan'][0]
us_index = countries.index[countries['NAME'] == 'United States'][0]
known_introductions_list = ['United States', 'Korea, Republic of', 'Japan']


# # **Model Parameters & Runs**

### notes on numbers used and rationale
## Parameters that should be calibrated and validated as much as possible
# alpha - just choose these as starting values
# beta - just choose these as starting values
# mu - just choose these as starting values

## Parameters that we can set based on underlying data to normalize
# sigma_h = 1 - the mean of the host percent area (not sure that this is the best assumption here but normalizes and gives results that make sense here)
# sigma_phi = 1 (assummes that a specialist that feeds on only one type of host will have a harder time invading than a generalist) (needs to be an integer)
# sigma_kappa = just selected a value but plan on 1 - mean of the koppen climate matches
# sigma_epsilon doesn't matter right now (we aren't using ecological disturbance this part of the equation drops out (i.e. changing this value doesn't afffect the simulation))
# sigma_T - I still need to adjust this 


# alpha = 0.2 #@param {type:"number"}
# beta = 0.2 #@param {type:"number"}
# mu = 0.0002 #@param {type:"number"}
# lamda_c = 1 #@param {type:"number"}
# phi = 2 #@param {type:"integer"}
# sigma_epsilon = 0.5 #@param {type:"number"}
# sigma_h = 1 - 0.16 #@param {type:"number"}
# sigma_kappa = 1 - 0.3 #@param {type:"number"}
# sigma_phi =  1 #@param {type:"integer"}
# sigma_T = 9500 #@param {type:"integer"}
# start_year = 2000 #@param {type:"integer"}

alpha = float(sys.argv[9])
beta = float(sys.argv[10]) 
mu = float(sys.argv[11]) 

sigma_h = 1 - countries['Host Percent Area'].mean() 
sigma_kappa = 1 - .002 #mean of Koppen climate matches
sigma_phi =  1 
sigma_T = int(sys.argv[12]) #trades.mean = 11000

lamda_c = 1 #@param {type:"number"}
phi = 2 #@param {type:"integer"}
sigma_epsilon = 0.5 #@param {type:"number"}
start_year = 2000 #@param {type:"integer"}

run_num = int(sys.argv[13])


# Runs the full model
if len(sys.argv) == 15:
    np.random.seed(seed=int(sys.argv[14]))
else:
    np.random.seed(seed=None)
    
trades = trades
distances = distances
locations = countries
prob = np.zeros(len(countries.index))
pres_ts0 = [False] *len(prob)
pres_ts0[china_index] = True 
pres_ts0[viet_nam_index] = True
pres_ts0[india_index] = True
locations["Presence"] = pres_ts0
#locations["Phytosanitary Capacity"] = prob

print('PARAMETER VALUES:')
print(f'alpha: {alpha}\tbeta: {beta}\tmu: {mu}')
print(f'sigma_h: {sigma_h}\tsigma_kappa: {sigma_kappa}\tsigma_T: {sigma_T}')

#comms = directory_path.split('/')[9]
#time_agg = directory_path.split('/')[8]
comms = 'select_commodities'
time_agg = 'monthly'
print(f'Commodities: {comms} @ {time_agg}')
print(f'GPD vals:\n{gdp_dict}')

e = pandemic2(
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
    start_year=start_year
)


# # **View and Save Output**

#Save model output objects
check = e[0] #locations
prob_entry = e[1]
prob_est = e[2] 
prob_intro = e[3]
origin_dst = e[4] 
country_intro = e[5]
date_list_out = e[6]


arr_dict = {'prob_entry': 'probability_of_entry',
           'prob_intro': 'probability_of_introduction',
           'prob_est': 'probability_of_establishment',
           'country_introduction': 'country_introduction'}


outpath = f'{data_dir}/slf_model/outputs/{time_agg}/{comms}/phytosanitary/run{run_num}/'


def create_model_dirs(run_num, outpath, output_dict):
    os.makedirs(outpath, exist_ok = True)
    
    for key in output_dict.keys():
        os.makedirs(outpath + key, exist_ok = True)
        print(outpath + key)


create_model_dirs(run_num = run_num,
                  outpath = outpath,
                  output_dict = arr_dict)


def generate_model_metadata(outpath, run_num, native_countries_list, comms, time_agg, gdp_dict, main_model_output, select_origin_dst):
    with open(f'{outpath}run{run_num}_meta.txt', 'w') as file:
        file.write(f'PARAMETER VALS: \n\talpha: {alpha}\n\tbeta: {beta}\n\tmu: {mu}')
        file.write(f'\tsigma_h: {sigma_h}\n\tsigma_kappa: {sigma_kappa}\n\tsigma_T: {sigma_T}\n\n')
        file.write(f'NATIVE COUNTRIES AT T0:\n\t{native_countries_list}')
        file.write(f'COMMODITIES: {comms} @ {time_agg}\n\n')
        file.write('PHYTOSANITARY CAPACITY:\n\tDynamic by Year\n\tAggregated by equal intervals (i.e., "Length")\n')
        file.write(f'\tGPD vals:{gdp_dict}\n\n')
        file.write('COUNTRY INTRODUCTIONS:')
        file.write(f'\nTotal Number of Countries: {main_model_output["Presence 201812"].value_counts()[1]}')
        file.write(f'\n{select_origin_dst.to_string()}')
        file.close()
        print(f'saving: {outpath}run{run_num}_meta.txt')


generate_model_metadata(outpath = outpath,
                        run_num = run_num,
                        native_countries_list = native_countries_list,
                        comms = comms, 
                        time_agg = time_agg,
                        gdp_dict = gdp_dict, 
                        main_model_output = e[0],
                        select_origin_dst = origin_dst[origin_dst['Destination'].isin(known_introductions_list)])



def save_monthly_model_output(model_output_object, columns_to_drop, outpath, run_num):
    check = model_output_object[0] #locations
    prob_entry = model_output_object[1]
    prob_est = model_output_object[2] 
    prob_intro = model_output_object[3]
    origin_dst = model_output_object[4] 
    country_intro = model_output_object[5]
    date_list_out = model_output_object[6]
    
    out_gdf = check.drop(columns_to_drop, axis=1)
    out_gdf["geometry"] = [MultiPolygon([feature]) if type(feature) == Polygon else feature for feature in out_gdf["geometry"]]
    out_gdf.to_file(outpath + f'pandemic_output.geojson', driver='GeoJSON')

    origin_dst.to_csv(outpath + f'origin_destination.csv')
    
    for i in range(0, len(date_list_out)):
        ts = date_list_out[i]
        
        pro_entry_pd = pd.DataFrame(prob_entry[i])
        pro_entry_pd.columns = traded.columns
        pro_entry_pd.index = traded.index
        pro_entry_pd.to_csv(outpath + f"prob_entry/probability_of_entry_{str(ts)}.csv", float_format='%.2f', na_rep="NAN!")
        
        pro_intro_pd = pd.DataFrame(prob_intro[i])
        pro_intro_pd.columns = traded.columns
        pro_intro_pd.index = traded.index
        pro_intro_pd.to_csv(outpath + f"prob_intro/probability_of_introduction_{str(ts)}.csv", float_format='%.2f', na_rep="NAN!")
        
        pro_est_pd = pd.DataFrame(prob_est[i])
        pro_est_pd.columns = traded.columns
        pro_est_pd.index = traded.index
        pro_est_pd.to_csv(outpath + f"prob_est/probability_of_establishment_{str(ts)}.csv", float_format='%.2f', na_rep="NAN!")
        
        country_int_pd = pd.DataFrame(country_intro[i])
        country_int_pd.columns = traded.columns
        country_int_pd.index = traded.index
        country_int_pd.to_csv(outpath + f"country_introduction/country_introduction_{str(ts)}.csv", float_format='%.2f', na_rep="NAN!")
    
    return out_gdf, date_list_out


columns_to_drop = ['AREA_x', 
                   'Af',
                   'Am',
                   'Aw',
                   'BWh',
                   'BWk',
                   'BSh',
                   'BSk',
                   'Csa',
                   'Csb',
                   'Csc',
                   'Cwa',
                   'Cwb',
                   'Cwc',
                   'Cfa',
                   'Cfb',
                   'Cfc',
                   'Dsa',
                   'Dsb',
                   'Dsc',
                   'Dsd',
                   'Dwa',
                   'Dwb',
                   'Dwc',
                   'Dwd',
                   'Dfa',
                   'Dfb',
                   'Dfc',
                   'Dfd',
                   'ET',
                   'EF',
                   'NAME_y','Phytosanitary Capacity 2000',
                   'Phytosanitary Capacity 2001',
                   'Phytosanitary Capacity 2002',
                   'Phytosanitary Capacity 2003',
                   'Phytosanitary Capacity 2004',
                   'Phytosanitary Capacity 2005',
                   'Phytosanitary Capacity 2006',
                   'Phytosanitary Capacity 2007',
                   'Phytosanitary Capacity 2008',
                   'Phytosanitary Capacity 2009',
                   'Phytosanitary Capacity 2010',
                   'Phytosanitary Capacity 2011',
                   'Phytosanitary Capacity 2012',
                   'Phytosanitary Capacity 2013',
                   'Phytosanitary Capacity 2014',
                   'Phytosanitary Capacity 2015',
                   'Phytosanitary Capacity 2016',
                   'Phytosanitary Capacity 2017',
                   'Phytosanitary Capacity 2018',
                   'Phytosanitary Capacity 2019',
                   'Presence',
                   'Probability of introduction',
                   'pc_mode']

out_gdf, date_list_out = save_monthly_model_output(model_output_object = e,
                                   columns_to_drop = columns_to_drop,
                                   outpath = outpath, 
                                   run_num = run_num)


def aggregate_monthly_output_to_annual(start_year, date_list_out, formatted_geojson, outpath):
    annual_ts_list = range(start_year, int(start_year + len(date_list_out)/12), 1)
    #presence_cols = [c for c in formatted_geojson.columns if c.startswith('Presence')]
    prob_intro_cols = [c for c in formatted_geojson.columns if c.startswith('Probability of introduction')]
    nh_list = ['09', '10', '11', '12', '01', '02', '03', '04']
    sh_list = ['04', '05', '06', '07', '08', '09', '10']

    for year in annual_ts_list:
        prob_cols = [c for c in prob_intro_cols if str(year) in c]
        nh_prob_cols = [x for x in prob_cols if x[-2:] in nh_list]
        sh_prob_cols = [x for x in prob_cols if x[-2:] in sh_list]
        ##TO DO: add in check for seasonality flag, otherwise use average for entire year
        formatted_geojson[f'Avg Probability of introduction {str(year)}'] = (np.where(formatted_geojson['centroid_lat']>=0, 
                                                                  formatted_geojson[nh_prob_cols].mean(axis=1), 
                                                                  formatted_geojson[sh_prob_cols].mean(axis=1)))
        formatted_geojson[f'Max Probability of introduction {str(year)}'] = np.max(formatted_geojson[prob_cols], axis=1)
        formatted_geojson[f'Presence {year}'] = formatted_geojson[f'Presence {year}12']
    
    formatted_geojson.to_file(outpath + f'pandemic_output_aggregated.geojson', driver='GeoJSON')
    formatted_df = pd.DataFrame(formatted_geojson)
    formatted_df.drop(['geometry'], axis=1, inplace=True)
    formatted_df.to_csv(outpath + f'pandemic_output_aggregated.csv', float_format='%.2f', na_rep="NAN!")
    return formatted_geojson, formatted_df


formatted_geojson, formatted_df = aggregate_monthly_output_to_annual(start_year = start_year,
                                   date_list_out = date_list_out,
                                   formatted_geojson = out_gdf,
                                   outpath = outpath)


#TO DO: Add in seasonality adjustment for average values 
def aggregate_monthly_array_outputs(num_time_steps, output_type, model_arr, output_name, start_year, out_path, run_num):
    t=0
    for i in range(0, num_time_steps, 12):
        avg_arr = (model_arr[i]+ 
            model_arr[i+1]+
            model_arr[i+2]+ 
            model_arr[i+3]+
            model_arr[i+4]+
            model_arr[i+5]+
            model_arr[i+6]+ 
            model_arr[i+7]+
            model_arr[i+8]+ 
            model_arr[i+9]+
            model_arr[i+10]+ 
            model_arr[i+11]) / 12
        avg_df = pd.DataFrame(avg_arr)
        avg_df.columns = traded.columns
        avg_df.index = traded.index
        avg_df.to_csv(outpath +  f"{output_type}/avg_{output_name}_{str(start_year + t)}.csv", float_format='%.2f', na_rep="NAN!")

        max_arr = np.maximum.reduce([model_arr[i],
                                    model_arr[i+1],
                                    model_arr[i+2], 
                                    model_arr[i+3],
                                    model_arr[i+4],
                                    model_arr[i+5],
                                    model_arr[i+6], 
                                    model_arr[i+7],
                                    model_arr[i+8], 
                                    model_arr[i+9],
                                    model_arr[i+10], 
                                    model_arr[i+11]])
        max_df = pd.DataFrame(max_arr)
        max_df.columns = traded.columns
        max_df.index = traded.index
        max_df.to_csv(outpath +  f"{output_type}/max_{output_name}_{str(start_year + t)}.csv", float_format='%.2f', na_rep="NAN!")

        t += 1


aggregate_monthly_array_outputs(num_time_steps = prob_entry.shape[0],
                                output_type = 'prob_entry', 
                                model_arr = prob_entry,
                                output_name = 'probability_of_entry',
                                start_year = start_year,
                                out_path = outpath,
                                run_num = run_num)


aggregate_monthly_array_outputs(num_time_steps = prob_est.shape[0],
                                output_type = 'prob_est', 
                                model_arr = prob_est,
                                output_name = 'probability_of_establishment',
                                start_year = start_year,
                                out_path = outpath,
                                run_num = run_num)


aggregate_monthly_array_outputs(num_time_steps = prob_intro.shape[0],
                                output_type = 'prob_intro', 
                                model_arr = prob_intro,
                                output_name = 'probability_of_introduction',
                                start_year = start_year,
                                out_path = outpath,
                                run_num = run_num)




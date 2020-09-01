data_dir = 'G:/Shared drives/APHIS  Projects/Pandemic/Data'
countries_path = 'G:/Shared drives/APHIS  Projects/Pandemic/Data/slf_model/inputs/countries4.gpkg'
gdp_path = 'G:/Shared drives/APHIS  Projects/Pandemic/Data/GDP/2000_2019_GDP_perCapita/gdp_perCapita_binned.csv'
gdp_low = 0.2
gdp_mid = 0.7
gdp_high = 0.9
commodity_path = 'G:/Shared drives/APHIS  Projects/Pandemic/Data/slf_model/inputs/monthly/select_commodities/'
commodity_forecast_path = 'G:/Shared drives/APHIS  Projects/Pandemic/Data/slf_model/inputs/monthly/forecast/static/'
native_countries_list = [
    'China', 
    'Viet Nam', 
    'India']
alpha = 0.2
beta = 0.2
mu = 0
lamda_c = 1
phi = 2
sigma_epsilon = 0.5
sigma_phi = 1
start_year = 2000
random_seed = None
out_dir = 'G:/Shared drives/APHIS  Projects/Pandemic/Data/slf_model/outputs/'
columns_to_drop = [
                   'AREA_x', 
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
                   'NAME_y',
                   'Phytosanitary Capacity 2000',
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
                   'pc_mode'
                   ]

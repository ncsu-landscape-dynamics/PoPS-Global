# ************************************************************************
# Downloading data from Comtrade API, keeping a log of successful downloads and errors.
# Example: ice cream imports and exports
# https://github.com/evpu
# ************************************************************************

import pandas as pd
import numpy as np
import os
import json
from urllib.request import urlopen
import time
import csv

# print(os.getcwd())
# os.chdir('.')  # set your current directory

# create a directory where to save downloaded data
if not os.path.exists('data'):
    os.makedirs('data')

# ************************************************************************
# Obtain Comtrade country codes
# ************************************************************************
url = urlopen('http://comtrade.un.org/data/cache/partnerAreas.json')

country_code = json.loads(url.read().decode())
url.close()
country_code = pd.DataFrame(country_code['results'])
country_code['id']=country_code['id'].astype(str)

# Clean up country codes a bit
# drop 'all' and 'world'
country_code = country_code.drop([0, 1])
# drop areas that are "nes" or "Fmr"
#country_code = country_code[~country_code['text'].str.contains(', nes|Fmr')]
# drop some specific countries
# country_code = country_code[country_code['text'] != 'Czechoslovakia']
# country_code = country_code[country_code['text'] != 'East and West Pakistan']
# country_code = country_code[country_code['text'] != 'Fr. South Antarctic Terr.']
# country_code = country_code[country_code['text'] != 'Free Zones']
# country_code = country_code[country_code['text'] != 'Belgium-Luxembourg']
# country_code = country_code[country_code['text'] != 'Antarctica']
# country_code = country_code[country_code['text'] != 'Br. Antarctic Terr.']
# country_code = country_code[country_code['text'] != 'Br. Indian Ocean Terr.']
# country_code = country_code[country_code['text'] != 'India, excl. Sikkim']
# country_code = country_code[country_code['text'] != 'Peninsula Malaysia']
# country_code = country_code[country_code['text'] != 'Ryukyu Isd']
# country_code = country_code[country_code['text'] != 'Sabah']
# country_code = country_code[country_code['text'] != 'Sarawak']
# country_code = country_code[country_code['text'] != 'Sikkim']
# country_code = country_code[country_code['text'] != 'So. African Customs Union']
# country_code = country_code[country_code['text'] != 'Special Categories']
# country_code = country_code[country_code['text'] != 'USA (before 1981)']
# country_code = country_code[country_code['text'] != 'Serbia and Montenegro']


# ************************************************************************
# Error log file
# ************************************************************************
if os.path.isfile('log.csv'):  # if file exists, open to append
    csv_file = open('log.csv', 'a', newline='')
    error_log = csv.writer(csv_file, delimiter=',', quotechar='"')
else:  # else if file does not exist, create it
    csv_file = open('log.csv', 'w', newline='')
    error_log = csv.writer(csv_file, delimiter=',', quotechar='"')
    error_log.writerow(['reporter_id', 'reporter', 'hs', 'year', 'status', 'message', 'time'])


# ************************************************************************
# Imports
# ************************************************************************

auth_code = "jXIKwJ2httdcPDHwwJCj7GzbDh8fva23HYV17lyN+BeKrxX3fSviSAT9vgH5zQ+XnKj75SBnqPn25kXrwD1viUgtdDMNhpjrw4ZPcpdznaYq1nH8F/wxSoUBSMUzwVVb3YsoqruN04qDiJU/NleTCA=="

# add lines to import HS list and loop through codes of interest at 4 digits
#hs = "6801"
hs_68 = np.arange(6802, 6815+1, 1)
for hs in hs_68:
    start_year = 1994
    end_year = 2018
    years = np.arange(start_year, end_year+1, 1)
    for year in years:
        HS_matrix = country_code[['id']]
        for i in country_code['id']:  # loop over all countries
            #time.sleep(45)  # prob not needed

            try:
                url = urlopen('http://comtrade.un.org/api/get?max=250000&type=C&px=HS&cc=' + str(hs) + '&r=' + str(i) + '&rg=1&p=all&freq=A&ps=' + str(year) + '&fmt=json&token=' + str(auth_code))
                raw = json.loads(url.read().decode())
                url.close()
            except:  # if did not load, try again
                try:
                    url = urlopen('http://comtrade.un.org/api/get?max=250000&type=C&px=HS&cc=' + str(hs) + '&r=' + str(i) + '&rg=1&p=all&freq=A&ps=' + str(year) + '&fmt=json&token=' + str(auth_code))
                    raw = json.loads(url.read().decode())
                    url.close()
                except:  # if did not load again, move on to the next country in the loop
                    error_log.writerow([country_code[country_code['id'] == str(i)]['text'].tolist()[0], i, hs, year, 'Fail', raw['validation']['message'], time.ctime()])
                    print('Fail: country ' + str(i) + ', ' + str(year) + ", " + str(hs) + '. Message: ' + str(raw['validation']['message']))
                    continue

            # if no data was downloaded, add column of zeros to commodity/year df and move to next country
            if len(raw['dataset']) == 0:
                HS_matrix.assign(x = 0)
                HS_matrix.rename(columns={"x": i}, inplace=True)
                error_log.writerow([country_code[country_code['id'] == str(i)]['text'].tolist()[0], i, hs, year, 'no data', raw['validation']['message'], time.ctime()])
                print('No data: country ' + str(i) + '. Message: ' + str(raw['validation']['message']))
                continue

            # Merge quantity to commodity/year df
            data = pd.DataFrame(raw['dataset'])
            data['ptCode']=data['ptCode'].astype(str)        
            data = data[['ptCode', 'NetWeight']]
            HS_matrix = pd.merge(HS_matrix, data, how = 'left', left_on='id', right_on='ptCode')
            HS_matrix.drop("ptCode", axis=1, inplace=True)
            HS_matrix.rename(columns={"NetWeight": i}, inplace=True)
            print(str(i) + ": finished")


        HS_matrix.fillna(0, inplace=True)
        HS_matrix.to_csv('data/' + str(hs) + '_' + str(year) + '.csv', index=False)

csv_file.close()
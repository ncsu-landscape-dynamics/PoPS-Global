
'''
PoPS Pandemic - simulation

Copyright (C) 2020-2020 by the authors.

Authors: Chris Jones (cmjone25 ncsu edu)

The code contained herein is licensed under the GNU General Public
License. You may obtain a copy of the GNU General Public License
Version 3 or later at the following locations:

http://www.opensource.org/licenses/gpl-license.html
http://www.gnu.org/copyleft/gpl.html
'''

# Read in Data

# Calculate probability of introduction for each pest/pathogen between countries (including probability of re-introduction?)
    # trade 
    # species presence
    ## phytosanitary compliance origin
    ## phytosanitary compliance arrival
    ## commodity type
    ### host similarity between countries
    ### seasonality of host country and species lifecycle

# Calculate probability of establishment for each pest/pathogen between countries (maybe a categorical prediction)
    # Seasonality
    # Environmental similarity
    ## host similarity


## function list
# trade
# presence (pest)
# phytosanitary compliance
# climate match
# host similirity
# Seasonality
# 

# Trade function
    #1 total volume
    #2 commodity type (ag, forest, manufactured, waste)
    #3 commodity



# Function to calculate probability of arrival from one location to another given a 
# matrix of trade volume, matrix of distances, and locations with attribute information such as species presence and phytosanitary compliance
def arrival(arrivals, trades, locations, distances, index):
    for location in locations:
        arrivals[index, location] = trades[index, location] * distances[index, location] * locations.phytosanitary_compliance[location] # * locations.establishment


## this is a function to determine the similarity between country of origin and country of introduction
def host_similarity(source, country_introduction):
    print(source)


def introduction(s):
    print(s)


def establishment(s):
    print(s)

def invasion(s):
    print(s)
import pandas as pd
import numpy as np
import scipy as sp
# This runs the loop across the functions in the simulation

#locations = pd.DataFrame({'name': ["United States", "China", "Brazil"], 'phytosanitary_compliance': [0.25, 0.50, 0.75], 'Presence': [True, False, True]})
#trades = np.array([[0, 50, 15], [50, 0, 10], [20, 30, 0]])

#distances = np.array([[1, 5000, 105000], [5000, 1, 7500], [10500, 7500, 1]])

#distances = 1/distances

#trades = np.genfromtxt("Data/TradeData.csv", delimiter=',', names= True, dtype=int, usecols=range(0,242))
trades = pd.read_csv("Data/TradeData.csv")
trades = trades.iloc[0:241,1:242].to_numpy()
locations = pd.read_csv("Data/Presence.csv")
arrivals = np.empty(shape=(len(trades),len(trades)), dtype='object')
distances = np.ones_like(arrivals)
#locations = locations.iloc[0:241,]

def arrival(arrivals, trades, distances, locations, index):
    for ind in range(len(locations)):
        location = locations.iloc[index,:]
        arrivals[index, ind] = trades[index, ind] * distances[index, ind] #* location['phytosanitary_compliance'] # * locations.establishment
    return(arrivals)


def invasion(arrivals, trades, distances, locations):
    for ind in range(len(locations)):
        location = locations.iloc[ind,:]
        if (location['Presence']):
            arrivals = arrival(arrivals, trades, distances, locations, ind)
    return(arrivals)

invasion(arrivals, trades, distances, locations)
print(arrivals)

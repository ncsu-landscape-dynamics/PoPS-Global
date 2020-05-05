import pandas as pd
import numpy as np
import scipy as sp
import math

#locations = pd.DataFrame({'name': ["United States", "China", "Brazil"], 'phytosanitary_compliance': [0.25, 0.50, 0.75], 'Presence': [True, False, True]})
trade = np.array([ [[0, 50, 15], [50, 0, 10], [20, 30, 0]], [[0, 50, 15], [50, 0, 10], [20, 30, 0]], [[0, 50, 15], [50, 0, 10], [20, 30, 0]]])

#distances = np.array([[1, 5000, 105000], [5000, 1, 7500], [10500, 7500, 1]])

#distances = 1/distances

#trades = np.genfromtxt("Data/TradeData.csv", delimiter=',', names= True, dtype=int, usecols=range(0,242))
# trade = pd.read_csv("Data/TradeData.csv")
# trade = trade.iloc[0:241,1:242].to_numpy()
# locations = pd.read_csv("Data/Presence.csv")
# distances = np.ones_like(arrivals)
#locations = locations.iloc[0:241,]

# alpha is parameter that allows the equation to be adapated to various discrete time steps
# beta is a parameter that allows the equation to be adapted to various discrete time steps
# delta_kappa_ijt is the climate dissimilarity between the origin (i) and destination (j) at time (t)
# sigma_kappa is a normalizing constant for climate dissimilarity
# h_jt is the percent of area in the destination (j) that is suitable host for the pest
# sigma_h is the host normalizing constant
# epsilon_jt is the ecological disturbance index of destination (j) at time (t)
# sigma_epsilon is the ecological disturbance normalizing constant
# phi is the degree of polyphagy of the pest of interest described as the number of host families
# sigma_phi is the degree of polyphagy normalizing constant
def probability_of_establishment(alpha, beta, delta_kappa_ijt, sigma_kappa, h_jt, sigma_h, epsilon_jt, sigma_epsilon, phi, sigma_phi):
    alpha * math.exp((-1) * beta * ( ((1 - delta_kappa_ijt)/sigma_kappa)**2 + ((1 - h_jt)/sigma_h)**2 + ((1 - epsilon_jt)/sigma_epsilon)**2 + (phi/sigma_phi)**(-2) )) 

# rho_i is the phytosanitary compliance of origin (i)
# rho_j is the phytosanitary compliance of destination (j)
# zeta_it is the species presence (binary 0 or 1) in origin (i) at time (t)
# lamda_c is the commodity importance [0,1] of commodity (c) in transporting the pest or pathogen
# T_ijct is the trade volume between origin (i) and destination (j) for commodity (c) at time (t) in metric tons
# mu is the mortality rate of the pest or pathogen during transport
# d_ij is the distance between origin (i) and destination (j)
# chi_it is the seasonality of the pest or pathogen in its ability to be in a shipment
def probability_of_entry(rho_i, rho_j, zeta_it, lamda_c, T_ijct, sigma_T, mu, d_ij, chi_it):    
    (1 - rho_i) * (1 - rho_j) * zeta_it * (1 - math.exp((-1) * lamda_c * (T_ijct / sigma_T))) * math.exp(mu *d_ij) * chi_it

def probability_of_introduction(probability_of_entry_ijct, probability_of_establishment_ijt):
    p_nointro = 1
    for c in range(len(probability_of_entry_ijct)):
        p_nointro = p_nointro * (1 - probability_of_entry_ijct[c] * probability_of_establishment_ijt)

    p_intro = 1 - p_nointro
    

# locations is a data frame of countries, species presence, phytosanitry capacity koppen climate classifications % of total area for each class, 
# trade is a list (c) of n x n x t matrices where c is the # of commoditites, n is the number of locations, and t is # of time steps
# distances is a n x n matrix of distances from one location to another where n is number of locations, 
def pandemic(trade, distances, locations, alpha, beta, mu, sigma_epsilon, sigma_h, sigma_kappa, sigma_phi, sigma_T):
    time_steps = trade.shape[2]
    establishment_probs = np.empty_like(trade, dtype=float)
    entry_probs = np.empty_like(trade, dtype='object')
    introduction_probs = np.empty_like(trade, dtype='object')
    for j in range(len(locations)):
        destination = locations.iloc[j,:] # this is the destination location (could be a country or port)
        # check that Phytosanitary capacity data is available if not set value to 0
        if ('Phytosanitary capacity' in destination):
            rho_j = destination['Phytosanitary capacity']
        else:
            rho_j = 0

        for i in range(len(locations)):
            origin = locations.iloc[i,:] # this is the origin location (could be a country or port)
            # check that Phytosanitary capacity data is available if not set value to 0
            if ('Phytosanitary capacity' in origin):
                rho_i = origin['Phytosanitary capacity']
            else:
                rho_i = 0
                delta_kappa_ijt = locations
            if (origin['Presence']):
                zeta_it = origin['Presence']
                probability_of_establishment(alpha, beta, delta_kappa_ijt, sigma_kappa, h_jt, sigma_h, epsilon_jt, sigma_epsilon, phi, sigma_phi)
                
                probability_of_entry(rho_i, rho_j, zeta_it, lamda_c, T_ijct, sigma_T, mu, d_ij, chi_it)
                probability_of_introduction(probability_of_entry_ijct, probability_of_establishment_ijt)
            else:

        # arrivals[i, j] = trades[i, j] * distances[i, j] * location['phytosanitary_compliance'] # * locations.establishment
    return(establishment_probs)


# def invasion(arrivals, trades, distances, locations):
#     for ind in range(len(locations)):
#         location = locations.iloc[ind,:]
#         if (location['Presence']):
#             arrivals = arrival(arrivals, trades, distances, locations, ind)
#     return(arrivals)



# invasion(arrivals, trades, distances, locations)
arrivals = np.zeros((4,4,4))
print(arrivals)
print(arrivals.shape[2])
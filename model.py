# This runs the loop across the functions in the simulation

locations = ["United States", "China"]
trade = [[0, 50], [50, 0]]

def invasion(locations):
    for location in locations:
        print(location)

invasion(locations)
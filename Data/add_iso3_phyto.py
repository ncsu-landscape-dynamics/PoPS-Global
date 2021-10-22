import os
import pycountry
import pandas
from fuzzywuzzy import process

drive_letter = "Q"
os.chdir(
    rf"{drive_letter}:/Shared drives/APHIS  Projects/"
    rf"Pandemic/Data/phytosanitary_capacity/"
)

capacities = pandas.read_csv(
    "Report_summary_english_spanish_french_russian_arabic_forPatricia.csv",
    encoding="ISO-8859-1",
)

# Drop rows with no data
capacities = capacities[capacities["Report_ava"].notna()]

# Get list of ISO country names
country_objs = list(pycountry.countries)
names = [o.name for o in country_objs]

# Use fuzzy string matching to match capacity data names with ISO names.
capacities["iso_name"] = ""
capacities["match_score"] = 0
for index in capacities.index:
    (
        capacities.loc[index, "iso_name"],
        capacities.loc[index, "match_score"],
    ) = process.extractOne(capacities.loc[index, "Report_ava"], names)

# List country name matches with score less than 98 for manual check.
check = capacities.loc[
    capacities["match_score"] < 100, ["Report_ava", "iso_name", "match_score"]
]
check

# Create dictionary of manual corrections for mismatches.
corrections = {
    "Bosnia and Herz.": "Bosnia and Herzegovina",
    "Korea (South Korea)": "Korea, Republic of",
    "St. Vin. and Gren.": "Saint Vincent and the Grenadines",
    "Swaziland": "Eswatini",
    "Dem. Rep. Congo": "Congo, The Democratic Republic of the",
    "Dominican Rep.": "Dominican Republic",
    "Eq. Guinea": "Equatorial Guinea",
}

# Replace ISO name with correct value.
for key in corrections.keys():
    capacities.loc[capacities["Report_ava"] == key, "iso_name"] = corrections[key]


# Add ISO3 code based on ISO names
capacities["ISO3"] = ""

for index in capacities.index:
    capacities.loc[index, "ISO3"] = (
        pycountry.countries.get(name=capacities.loc[index, "iso_name"])
    ).alpha_3

# Add UN codes if needed
un_to_iso_path = (
    rf"{drive_letter}:/Shared drives/APHIS  Projects/"
    rf"Pandemic/Data/Country_list_shapefile/temp.csv"
)
un_to_iso = pandas.read_csv(un_to_iso_path)
capacities = capacities.merge(un_to_iso, "left", left_on="ISO3", right_on="alpha_3")
capacities = capacities.append(
    {
        "Country_global": "United States",
        "reactive": 3,
        "proactive": 3,
        "ISO3": "USA",
        "UN": 840,
    },
    ignore_index=True,
)
# capacities.UN_id = capacities.UN_id.astype(int)
capacities.sort_values(by="UN")
capacities.to_csv("phytosanitary_capacity_iso3.csv")

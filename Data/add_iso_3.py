import os
import pycountry
import pandas
from fuzzywuzzy import process

os.chdir("H:/Shared drives/APHIS  Projects/Pandemic/Data/phytosanitary_capacity/")

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
    capacities["match_score"] < 98, ["Report_ava", "iso_name", "match_score"]
]

# Create dictionary of manual corrections for mismatches.
corrections = {
    "Bosnia and Herz.": "Bosnia and Herzegovina",
    "Korea (South Korea)": "Korea, Republic of",
    "St. Vin. and Gren.": "Saint Vincent and the Grenadines",
    "Swaziland": "Eswatini",
}

# Replace ISO name with correct value.
for key in corrections.keys():
    capacities.loc[capacities["Report_ava"] == key, "iso_name"] = corrections[key]


# Add ISO3 code based on ISO names
capacities["alpha_3"] = ""

for index in capacities.index:
    capacities.loc[index, "alpha_3"] = (
        pycountry.countries.get(name=capacities.loc[index, "iso_name"])
    ).alpha_3


capacities.to_csv("phytosanitary_capacity_iso3.csv")

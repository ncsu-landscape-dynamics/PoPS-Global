import requests
from time import sleep

import pandas as pd
import itertools

# GBIF Match API call: exact and fuzzy matching of species name to GBIF codes


def get_GBIF_key(species):
    call = f"https://api.gbif.org/v1/species/match?verbose=true&name={species}"
    response = requests.get(call).json()
    try:
        usageKey = response["usageKey"]
        scientificName = response["scientificName"]
        print(f"Match found for {species}: {scientificName} ({usageKey})!")
    except KeyError:
        usageKey = None
        print(f"No match was found for: {species}")
    return usageKey


# GBIF API call: occurrence status = present,
# count for each species/year, for all countries


def gbif_counts_api(usageKey, year):
    call = (
        f"https://api.gbif.org/v1/occurrence/search?year={year}"
        f"&occurrence_status=present&taxonKey={usageKey}"
        f"&facet=country&facetlimit=300&limit=0"
    )
    return call


# Unpack the response (JSON) into just the country - count values
def call_gbif_api(call):
    try:
        response = requests.get(call).json()
    except requests.exceptions.RequestException:
        print("Just a second...")
        sleep(5)
        try:
            response = requests.get(call, verify=False).json()
        except requests.exceptions.RequestException:
            print("Trying a minute...")
            sleep(20)
            response = requests.get(call).json()
    response_vals = response["facets"][0]["counts"]
    country = []
    counts = []
    for i in range(0, len(response_vals)):
        country.append(response_vals[i]["name"])
        counts.append(response_vals[i]["count"])
    return [country, counts]


def get_GBIF_records(species, year_list):

    species_year = [[species], year_list]
    species_years = list(itertools.product(*species_year))

    api_calls_df = pd.DataFrame(species_years, columns=["species", "years"])
    api_calls_df["api_call"] = api_calls_df.apply(
        lambda x: gbif_counts_api(x.species, x.years), axis=1
    )

    # Sending API calls - response of [[countries],[counts]]

    api_calls_df["result"] = api_calls_df.api_call.apply(call_gbif_api)

    # Expanding the results into lists of countries and counts
    api_calls_df[["country", "counts"]] = api_calls_df.result.apply(pd.Series)

    # Coverting the lists of countries and counts to individual rows
    first_records = (
        api_calls_df.drop(columns="result")
        .set_index(["species", "years", "api_call"])
        .apply(pd.Series.explode)
        .reset_index()[["years", "country"]]  # Extracting just country year
        .groupby("country")
        .min()
        .years.reset_index()  # as first records
        # Match ISO2 to ISO3
        .rename(columns={"country": "ISO2", "years": "ObsFirstIntro"})
    )

    iso_map = pd.read_csv("Data/un_to_iso.csv")
    first_records = first_records.merge(
        iso_map[["ISO2", "ISO3"]].drop_duplicates(), on="ISO2"
    )[["ISO3", "ObsFirstIntro"]]

    print(
        f"First recorded observations ({len(first_records.index)}"
        " countries) processed successfully!"
    )

    return first_records

# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import geopandas
import numpy as np
import os

os.chdir("H:/Shared drives/APHIS  Projects/Pandemic/Data/")

countries_path = "H:/Shared drives/APHIS  Projects/Pandemic/Data/Country_list_shapefile/TM_WORLD_BORDERS-0.3"
countries = geopandas.read_file(countries_path)
countries


# %%
# UN to ISO crosswalk
crosswalk = pd.read_csv(
    "H:/Shared drives/APHIS  Projects/Pandemic/Data/Comtrade Country Code and ISO list.csv",
    encoding="ISO-8859-1",
    index_col="Country Code",
)
crosswalk


# %%
# Check if any ISO3 duplicates overlap
dups = crosswalk.pivot_table(index=["ISO3-digit Alpha"], aggfunc="size")
dups[dups > 1]
# All duplicate ISO3 codes are ok (are not used within same time step)


# %%
shp_iso3 = countries["ISO3"]
shp_iso3 = shp_iso3.to_frame()


# %%
un_iso3 = crosswalk["ISO3-digit Alpha"]
un_iso3 = un_iso3.to_frame()
un_iso3 = un_iso3.rename(columns={"ISO3-digit Alpha": "ISO3"})


# %%
both_iso3 = un_iso3.merge(shp_iso3, on="ISO3")


# %%
# Identify ISO3 codes that are not in the countries shapefile, but are in the UN to ISO3 crosswalk. Omit N/As.
no_shp_iso3 = un_iso3[~un_iso3.ISO3.isin(both_iso3.ISO3)]
no_shp_iso3[no_shp_iso3.ISO3.notnull()]


# %%
# Where possible, change UN crosswalk to use ISO3 codes that match the shapefile (codes that represent the correct geography).
# Create dictionary of manual corrections for mismatches.
corrections = {
    "BES": "ANT",  #
    "CUW": "ANT",
    "DDR": "DEU",
    "VDR": "VNM",
    "YMD": "YEM",
    "PCZ": "PAN",
    "SCG": "SRB",
    "SSD": "SDN",
}
corrections


# %%
# ISO3 codes that do not have matching geography in shapefile

# CSW - now two separate ISO3 codes (CZE, SVK)
# PCI - former Pacific islands (1962 - 1991)
# SUN - former USSR
# YUG - former Yugoslavia
# SXM - Sint Maarten (southern part of St Martin, not in geospatial data?)


# %%
# Replace ISO code with correct values.
for key in corrections.keys():
    crosswalk.loc[
        crosswalk["ISO3-digit Alpha"] == key, "ISO3-digit Alpha"
    ] = corrections[key]


# %%
crosswalk[["ISO3-digit Alpha", "Country Name, Abbreviation"]]


# %%
crosswalk[["ISO3-digit Alpha", "Country Name, Abbreviation"]].to_csv("un_to_iso.csv")

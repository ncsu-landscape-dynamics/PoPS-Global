# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import geopandas
import os
import dotenv

# Load variables and paths from .env
dotenv.load_dotenv(".env")

dir_path = os.getenv("DATA_PATH")

countries_path = dir_path + "Country_list_shapefile/TM_WORLD_BORDERS-0.3"
countries = geopandas.read_file(countries_path)
countries


# %%
# UN to ISO crosswalk
crosswalk = pd.read_csv(
    dir_path + "Comtrade Country Code and ISO list.csv",
    encoding="ISO-8859-1",
)


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
# Identify ISO3 codes that are not in the countries shapefile,
# # but are in the UN to ISO3 crosswalk. Omit N/As.
no_shp_iso3 = un_iso3[~un_iso3.ISO3.isin(both_iso3.ISO3)]
no_shp_iso3[no_shp_iso3.ISO3.notnull()]


# %%
# Where possible, change UN crosswalk to use ISO3 codes that
# match the shapefile (codes that represent the correct geography).
# If changes result in duplicate ISO3 codes in a timestep, trade data
# will need to be summed for the duplicate codes in the timestep
# (when 2 former countries combine into one modern country)
# Remove ISO3 codes that do not have matching geography in shapefile
# (either border changes, or split into multiple countries)
# Codes that are removed will not have trade data for the model
# during the years affected (specified in parentheses below)

# Create dictionary of manual corrections.
corrections = {
    # (1962 - 1990) Former German Democratic Republic to Germany,
    # combine with Former Fed. Rep. of Germany (DEU) for these years
    "DDR": "DEU",
    # (1962 - 1974) Former Democratic Republic of Viet-Nam to Viet Nam,
    # combine with Former Rep. of Vietnam (VNM) for these years
    "VDR": "VNM",
    # (1962 - 1990) Former Democratic Yemen to Yemen,
    # combine with Former Arab Rep of Yemen (YEM) for these years
    "YMD": "YEM",
    # (1992 - 2005) Serbia and Montenegro to Serbia,
    # for these years, Montenegro's trade will be mapped to Serbia
    "SCG": "SRB",
    # (1962 - 1977) Zone of the Panama Canal
    "PCZ": "",
    # (1962 - 1992) now two separate ISO3 codes (CZE, SVK),
    # during these years there will be no trade data for Czech Rep
    # and Slovakia. Could consider mapping to Czech Rep if needed.
    "CSK": "",
    # (1962 - 1991) former Pacific islands
    "PCI": "",
    # (1962 - 1991) former USSR, now 12 USO3 codes, during these year
    # there will be no trade data for Russia, Georgia, Ukraine, Moldova,
    # Belarus, Armenia, Azerbaijan, Kazakhstan, Uzbekistan,
    # Turkmenistan, Kyrgyzstan, Tajikistan
    "SUN": "",
    # (1962 - 1991) former Yugoslavia, now 7 ISO3 codes, during these
    # year there will be no trade data for Bosnia and Herzgovina,
    # Croatia, Kosovo, Montenegro, North Macedonia, Serbia, Slovenia
    "YUG": "",
    # (1962 - 2010) Netherland Antilles, split into 4 ISO3 codes,
    # during these year, will be no trade data for Sint Eustatius,
    # Bonaire, Curacao. Except Aruba which will only be missing
    # from 1962 - 1988
    "ANT": "",
    # European Union
    "EU2": "",
    # world
    "WLD": "",
}

# Other notes:
# Prior to 2012, SDN refers to Sudan prior to splitting into two countries.
# Trade data will include all of former area but will be mapped to northern part
# (modern SDN). # If historical modeling is done (pre-1992), should check to see
# if additional changes need to be made
# Many N/As in original crosswalk were dropped. Some are historical countries
# (Tanganyika, Zanzibar, Peninsula Malaysia) or uninhabited areas (Bouvet Island).
# VIR, MTQ, GLP, GUF - these territories have current ISO3 codes, but do not have
# current UN codes. If modeling prior to 1996 (or prior to 1981 for VIR), these codes
# will be included but will not have modern trade data.


# %%
# Replace ISO code with correct values.
for key in corrections.keys():
    crosswalk.loc[
        crosswalk["ISO3-digit Alpha"] == key, "ISO3-digit Alpha"
    ] = corrections[key]


# %%
crosswalk = crosswalk[
    [
        "Country Code",
        "ISO3-digit Alpha",
        "ISO2-digit Alpha",
        "Country Name, Abbreviation",
        "Start Valid Year",
        "End Valid Year",
    ]
].rename(
    columns={
        "Country Code": "UN",
        "ISO3-digit Alpha": "ISO3",
        "ISO2-digit Alpha": "ISO2",
        "Country Name, Abbreviation": "Name",
        "Start Valid Year": "Start",
        "End Valid Year": "End",
    },
)


# %%
crosswalk.to_csv("un_to_iso.csv", index=False)

# %%

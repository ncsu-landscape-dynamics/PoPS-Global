import os
import sys
import json
import time
import numpy as np
import pandas as pd
import geopandas

from pandemic.helpers import (
    distance_between,
    location_pairs_with_host,
    filter_trades_list,
    create_trades_list,
)
from pandemic.probability_calculations import (
    probability_of_entry,
    probability_of_establishment,
    probability_of_introduction,
)
from pandemic.ecological_calculations import (
    climate_similarity,
    create_climate_similarities_matrix,
)
from pandemic.output_files import (
    create_model_dirs,
    save_model_output,
    agg_prob,
    get_feature_cols,
    create_feature_dict,
    add_dict_to_geojson,
    aggregate_monthly_output_to_annual,
)

from pandemic.model_equations import (
    pandemic_single_time_step,
    pandemic_multiple_time_steps,
)

# Read model arguments from configuration file
# path_to_config_json = sys.argv[1]
path_to_config_json = (
    "C:/Users/cawalden/Documents/GitHub/Pandemic_Model/pandemic/config.json"
)

with open(path_to_config_json) as json_file:
    config = json.load(json_file)

data_dir = config["data_dir"]
countries_path = config["countries_path"]
phyto_path = config["phyto_path"]
commodity_path = config["commodity_path"]
commodity_forecast_path = config["commodity_forecast_path"]
native_countries_list = config["native_countries_list"]
season_dict = config["season_dict"]
alpha = config["alpha"]
beta = config["beta"]
mu = config["mu"]
lamda_c_list = config["lamda_c_list"]
phi = config["phi"]
sigma_epsilon = config["sigma_epsilon"]
sigma_phi = config["sigma_phi"]
start_year = config["start_year"]
random_seed = config["random_seed"]
out_dir = config["out_dir"]
columns_to_drop = config["columns_to_drop"]
time_infect = config["time_to_infectivity"]

countries = geopandas.read_file(countries_path, driver="GPKG")
distances = distance_between(countries)
phyto_data = pd.read_csv(phyto_path, index_col=0)
# Use only proactive capacity now. May incorporate reactive capacity dynamically later.
phyto_data = phyto_data[["proactive", "ISO3", "UN"]]
phyto_data = phyto_data.rename(columns={"proactive": "Phytosanitary Capacity"})

# Assign value to phytosanitary capacity estimates
countries = countries.merge(phyto_data, how="left", on="ISO3", suffixes=[None, "_y"])
phyto_dict = {
    np.nan: 0.0,
    0: 0.0,
    0.5: 0.15,
    1.0: 0.30,
    1.5: 0.45,
    2.0: 0.60,
    2.5: 0.75,
    3.0: 0.90,
}
countries.replace(phyto_dict, inplace=True)

# Read & format trade data
trades_list, file_list_filtered, code_list, commodities_available = create_trades_list(
    commodity_path=commodity_path,
    commodity_forecast_path=commodity_forecast_path,
    start_year=start_year,
    distances=distances,
)

# Create list of unique dates from trade data
date_list = []
for f in file_list_filtered:
    fn = os.path.split(f)[1]
    ts = str.split(os.path.splitext(fn)[0], "_")[-1]
    date_list.append(ts)
date_list.sort()

# Example trade array for formatting outputs
traded = pd.read_csv(
    file_list_filtered[0], sep=",", header=0, index_col=0, encoding="latin1"
)
# Checking trade array shapes
print("Length of trades list: ", len(trades_list))
for i in range(len(trades_list)):
    print("\tcommodity array shape: ", trades_list[i].shape)

# Create an n x n array of climate similarity calculations
print(f"Calculating climate similarities for {countries.shape[0]} locations")
climate_similarities = create_climate_similarities_matrix(
    array_template=traded, countries=countries
)

# Run Model for Selected Time Steps and Commodities
print("Number of commodities: ", len([c for c in lamda_c_list if c > 0]))
print("Number of time steps: ", trades_list[0].shape[0])
for i in range(len(trades_list)):
    if len(trades_list) > 1:
        code = code_list[i]
        print("\nRunning model for commodity: ", code)
    else:
        print(
            "\nRunning model for commodity: ",
            os.path.basename(commodities_available[0]),
        )
    trades = trades_list[i]
    distances = distances
    locations = countries
    prob = np.zeros(len(countries.index))
    pres_ts0 = [False] * len(prob)
    infect_ts0 = np.empty(locations.shape[0], dtype="object")
    for country in native_countries_list:
        country_index = countries.index[countries["NAME"] == country][0]
        pres_ts0[country_index] = True
        # if time steps are monthly and time to infectivity is in years
        if len(date_list[0]) > 4:
            infect_ts0[country_index] = str(start_year) + "01"
        # else if time steps are annual and time to infectivity is in years
        else:
            infect_ts0[country_index] = str(start_year)

    locations["Presence"] = pres_ts0
    locations["Infective"] = infect_ts0

    sigma_h = 1 - countries["Host Percent Area"].mean()
    iu1 = np.triu_indices(climate_similarities.shape[0], 1)
    sigma_kappa = 1 - climate_similarities[iu1].mean()
    sigma_T = np.mean(trades)
    np.random.seed(random_seed)
    lamda_c = lamda_c_list[i]

    if lamda_c > 0:
        e = pandemic_multiple_time_steps(
            trades=trades,
            distances=distances,
            locations=locations,
            climate_similarities=climate_similarities,
            alpha=alpha,
            beta=beta,
            mu=mu,
            lamda_c=lamda_c,
            phi=phi,
            sigma_epsilon=sigma_epsilon,
            sigma_h=sigma_h,
            sigma_kappa=sigma_kappa,
            sigma_phi=sigma_phi,
            sigma_T=sigma_T,
            start_year=start_year,
            date_list=date_list,
        )

        run_num = 9999 # sys.argv[2]
        run_iter = 9999 # sys.argv[3]

        arr_dict = {
            "prob_entry": "probability_of_entry",
            "prob_intro": "probability_of_introduction",
            "prob_est": "probability_of_establishment",
            "country_introduction": "country_introduction",
        }

        if len(trades_list) > 1:
            outpath = out_dir + f"/run_{run_num}/iter_{run_iter}/{code}/"
        else:
            outpath = out_dir + f"/run_{run_num}/iter_{run_iter}/"

        create_model_dirs(outpath=outpath, output_dict=arr_dict)
        print("saving model outputs: ", outpath)
        full_out_df = save_model_output(
            model_output_object=e,
            columns_to_drop=columns_to_drop,
            example_trade_matrix=traded,
            outpath=outpath,
            date_list=date_list,
        )

        # If time steps are monthly, aggregate predictions to
        # annual for dashboard display
        if len(date_list[i]) > 4:
            print("aggregating monthly predictions to annual time steps...")
            aggregate_monthly_output_to_annual(
                formatted_geojson=full_out_df, outpath=outpath
            )

        # Save model metadata to text file
        print("writing model metadata...")
        main_model_output = e[0]
        final_presence_col = sorted(
            [c for c in main_model_output.columns if c.startswith("Presence")]
        )[-1]
        meta = {}
        meta["PARAMETERS"] = []
        meta["PARAMETERS"].append(
            {
                "alpha": str(alpha),
                "beta": str(beta),
                "mu": str(mu),
                "lamda_c": str(lamda_c),
                "phi": str(phi),
                "sigma_epsilon": str(sigma_epsilon),
                "sigma_h": str(sigma_h),
                "sigma_kappa": str(sigma_kappa),
                "sigma_phi": str(sigma_phi),
                "sigma_T": str(sigma_T),
                "start_year": str(start_year),
                "infectivity_lag": str(time_infect),
                "random_seed": str(random_seed),
            }
        )
        meta["NATIVE_COUNTRIES_T0"] = native_countries_list
        meta["COMMODITY"] = commodities_available[i]
        meta["FORECASTED"] = commodity_forecast_path
        meta["PHYTOSANITARY_CAPACITY_WEIGHTS"] = phyto_dict
        meta["TOTAL COUNTRIES INTRODUCTED"] = str(
            main_model_output[final_presence_col].value_counts()[1]
            - len(native_countries_list)
        )

        with open(f"{outpath}/run_{run_num}_meta.txt", "w") as file:
            json.dump(meta, file, indent=4)

    else:
        print("\tskipping as pest is not transported with this commodity")

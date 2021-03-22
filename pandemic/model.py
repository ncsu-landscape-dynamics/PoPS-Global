import json
import os
import sys
import geopandas
import numpy as np
import pandas as pd
from dotenv import load_dotenv
sys.path.append("C:/Users/cawalden/Documents/GitHub/Pandemic_Model")
from pandemic.helpers import create_trades_list
from pandemic.model_equations import pandemic_multiple_time_steps
from pandemic.output_files import (
    aggregate_monthly_output_to_annual,
    create_model_dirs,
    save_model_output,
    write_model_metadata
)

# Read environmental variables
load_dotenv(os.path.join(".env"))
data_dir = os.getenv("DATA_PATH")
input_dir = os.getenv("INPUT_PATH")
out_dir = os.getenv("OUTPUT_PATH")
countries_path = os.getenv("COUNTRIES_PATH")

# Read model arguments from configuration file
path_to_config_json = sys.argv[1]
with open(path_to_config_json) as json_file:
    config = json.load(json_file)

commodity_path = config["commodity_path"]
commodity_forecast_path = config["commodity_forecast_path"]
native_countries_list = config["native_countries_list"]
season_dict = config["season_dict"]
alpha = config["alpha"]
beta = 0.5
mu = config["mu"]
lamda_c_list = config["lamda_c_list"]
phi = config["phi"]
w_phi = config["w_phi"]
start_year = config["start_year"]
stop_year = config["stop_year"]
random_seed = config["random_seed"]
cols_to_drop = config["columns_to_drop"]
time_infect_units = config["transmission_lag_unit"]
transmission_lag_type = config["transmission_lag_type"]
time_infect = config["time_to_infectivity"]
gamma_shape = config["transmission_lag_shape"]
gamma_scale = config["transmission_lag_scale"]
save_main_output = config["save_main_output"]
save_metadata = config["save_metadata"]
save_entry = config["save_entry"]
save_estab = config["save_estab"]
save_intro = config["save_intro"]
save_country_intros = config["save_country_intros"]
scenario_list = config['scenario_list']

countries = geopandas.read_file(countries_path, driver="GPKG")
distances = np.load(input_dir + "/distance_matrix_wTWN.npy")
climate_similarities = np.load(input_dir + "/climate_similarities_hiiMask16_wTWN.npy")

# Read & format trade data
trades_list, file_list_filtered, code_list, commodities_available = create_trades_list(
    commodity_path=commodity_path,
    commodity_forecast_path=commodity_forecast_path,
    start_year=start_year,
    stop_year=stop_year,
    distances=distances,
)

# Create list of unique dates from trade data
date_list = []
for f in file_list_filtered:
    fn = os.path.split(f)[1]
    ts = str.split(os.path.splitext(fn)[0], "_")[-1]
    date_list.append(ts)
date_list.sort()
end_sim_year = date_list[-1][:4]

# Example trade array for formatting outputs
traded = pd.read_csv(
    file_list_filtered[0], sep=",", header=0, index_col=0, encoding="latin1"
)

# Checking trade array shapes
print("Length of trades list: ", len(trades_list))
for i in range(len(trades_list)):
    print("\tcommodity array shape: ", trades_list[i].shape)


# Run Model for Selected Time Steps and Commodities
print("Number of commodities: ", len([c for c in lamda_c_list if c > 0]))
print("Number of time steps: ", trades_list[0].shape[0])
for i in range(len(trades_list)):
    if len(trades_list) > 1:
        code = code_list[i]
        print("\nRunning model for commodity: ", code)
    else:
        code = code_list[0]
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
    iu1 = np.triu_indices(climate_similarities.shape[0], 1)

    sigma_h = (1 - countries["Host Percent Area"]).std()
    sigma_kappa = np.std(1 - climate_similarities[iu1])

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
            sigma_h=sigma_h,
            sigma_kappa=sigma_kappa,
            w_phi=w_phi,
            start_year=start_year,
            date_list=date_list,
            season_dict=season_dict,
            transmission_lag_type=transmission_lag_type,
            time_infect_units=time_infect_units,
            time_infect=time_infect,
            gamma_shape=gamma_shape,
            gamma_scale=gamma_scale,
            scenario_list=scenario_list,
        )

        sim_name = sys.argv[2]
        add_descript = sys.argv[3]
        run_num = sys.argv[4]

        run_prefix = f"{add_descript}_{code}"

        arr_dict = {
            "prob_entry": "probability_of_entry",
            "prob_intro": "probability_of_introduction",
            "prob_est": "probability_of_establishment",
            "country_introduction": "country_introduction",
        }

        outpath = out_dir + f"/{sim_name}/{run_prefix}/run_{run_num}/"
        create_model_dirs(outpath=outpath, output_dict=arr_dict)
        print("saving model outputs: ", outpath)
        full_out_df = save_model_output(
            model_output_object=e,
            example_trade_matrix=traded,
            outpath=outpath,
            date_list=date_list,
            write_entry_probs=save_entry,
            write_estab_probs=save_estab,
            write_intro_probs=save_intro,
            write_country_intros=save_country_intros,
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
        write_model_metadata(
            main_model_output=e[0],
            alpha=alpha,
            beta=beta,
            mu=mu,
            lamda_c_list=lamda_c_list,
            phi=phi,
            sigma_h=sigma_h,
            w_phi=w_phi,
            sigma_kappa=sigma_kappa,
            start_year=start_year,
            end_sim_year=end_sim_year,
            transmission_lag_type=transmission_lag_type,
            time_infect_units=time_infect_units,
            gamma_shape=gamma_shape,
            gamma_scale=gamma_scale,
            random_seed=random_seed,
            time_infect=time_infect,
            native_countries_list=native_countries_list,
            countries_path=countries_path,
            commodities_available=commodities_available[i],
            commodity_forecast_path=commodity_forecast_path,
            phyto_weights=list(locations["Phytosanitary Capacity"].unique()),
            outpath=outpath,
            run_num=run_num,
            scenario_list=scenario_list,
        )
    else:
        print("\tskipping as pest is not transported with this commodity")

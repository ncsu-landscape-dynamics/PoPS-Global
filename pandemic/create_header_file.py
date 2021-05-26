import sys
import os
import re
import glob
import json
import pandas as pd

# Directory and file paths
root_dir = sys.argv[1]  # directory with model outputs
sim_name = sys.argv[2]  # name of simulation

# root_dir = "H:/Shared drives/SLF Paper Outputs/outputs"
# sim_name = "slf_scenarios_noTWN_wChinaVietnam"

# Generate header attributes from subdirectory names,
# and model output metadata
sim_path = os.path.join(root_dir, sim_name)
run_prefix_list = [
    os.path.basename(d) for d in (glob.glob(sim_path + "/*")) if os.path.isdir(d)
]
if "header.csv" in run_prefix_list:
    run_prefix_list.remove("header.csv")
if "summary_data" in run_prefix_list:
    run_prefix_list.remove("summary_data")

# commodity_codes_list = list(set([d.split("_")[-1] for d in run_prefix_list]))
commodity_codes_list = ["6801-6804"]
# add_descript_list = (
#     [d.split(f"_{i}")[0] for d in run_prefix_list for i in commodity_codes_list]
# )
add_descript_list = [os.path.basename(i) for i in glob.glob(sim_path + "/*/*")]


num_runs_list = []
for i in run_prefix_list:
    run_list = [r for r in glob.glob(sim_path + f"/{i}/{add_descript_list[0]}/run_*")]
    num_runs_list.append(len(run_list))

parameter_values_list = []
starting_countries_list = []
start_year_list = []
stop_year_list = []

for i in range(len(run_prefix_list)):
    run_prefix = run_prefix_list[i]
    run0 = re.sub(
        "run_",
        "",
        os.path.basename(
            glob.glob(f"{sim_path}/{run_prefix}/{add_descript_list[0]}/run_*")[0]
        ),
    )
    file_path = (
        rf"{sim_path}/{run_prefix}/{add_descript_list[0]}",
        rf"/run_{run0}/run_{run0}_meta.json",
    )
    metadata_file = open(file_path)
    meta_contents = json.load(metadata_file)
    parameter_values = meta_contents["PARAMETERS"]
    parameter_values_list.append(parameter_values)
    starting_countries = meta_contents["NATIVE_COUNTRIES_T0"]
    starting_countries_list.append(starting_countries)
    start_year = [param["start_year"] for param in meta_contents["PARAMETERS"]][0]
    start_year_list.append(start_year)
    stop_year = [param["end_sim_year"] for param in meta_contents["PARAMETERS"]][0]
    stop_year_list.append(stop_year)

# Write attributes to header file
header_file_path = f"{sim_path}/header.csv"
header_dict = {
    "attributes": [
        "root_dir",
        "simulation_name",
        "additional_descriptors",
        "run_prefix_list",
        "num_runs",
        "parameter_values",
        "starting_countries",
        "start_year",
        "stop_year",
        "commodity_codes",
        "folder_structure",
    ],
    "values": [
        root_dir,
        sim_name,
        add_descript_list,
        run_prefix_list,
        num_runs_list,
        parameter_values_list,
        starting_countries_list,
        start_year_list,
        stop_year_list,
        commodity_codes_list,
        r"/root_dir/simulation_name/additional_descriptor/"
        r"run_prefix/num_runs[x]/commodity_code_list[x]/*",
    ],
}
header_df = pd.DataFrame.from_dict(header_dict)
print("saving header file: ", header_file_path)
header_df.to_csv(header_file_path, header=True)

import os

project_loc = os.getcwd()
input = "/inputs"
output = "/outputs"
countries_file = "/countries.gpkg"

with open(".env", "w") as f:
    f.write(f"DATA_PATH='{project_loc}'\n")
    f.write(f"INPUT_PATH='{project_loc}{input}'\n")
    f.write(f"OUTPUT_PATH='{project_loc}{output}'\n")
    f.write(f"TEMP_OUTPATH='/scratch/temp_outputs'\n")
    f.write(f"COUNTRIES_PATH='{project_loc}{input}{countries_file}'\n")
    f.close()

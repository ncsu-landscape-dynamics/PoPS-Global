import itertools
import json

def write_commands(params, start_run, end_run, model_files="Keep"):
    # Name the script to be run
    if model_files == "Temp":
        script = "./hpc/wrapper_script.csh"
    else: 
        script = "python pandemic/multirun_helpers/model_run_args.py"
    output = (
        " ".join(
            [
                script,
                str(params[0]), # alpha
                str(params[1]), # beta
                str(params[2]), # lamda
                str(params[3]), # start year
                str(start_run),
                str(end_run),
            ]
        )
        + "\n"
    )
    return output 


if __name__ == "__main__":

    with open("config.json") as json_file:
        config = json.load(json_file)
    start_years = config["start_years"]
    alphas = config["alphas"]
    betas = config["betas"]
    lamdas = config["lamdas"]
    try:
        model_files = config["model_files"]
    except: 
        model_files = "Keep"

    param_list = [alphas, betas, lamdas, start_years]
    param_sets = list(itertools.product(*param_list))

    # Full run
    start_run = config["start_run"]
    end_run = config["end_run"]

    file1 = open("commands.txt", "w")
    # Write to a text file    
    for params in param_sets:
        file1.write(write_commands(model_files, params, start_run, end_run))

    file1.close()


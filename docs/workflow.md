# PoPS Global Workflow
A series of Jupyter notebooks are available for setting up the model configuration, formatting the input data, and running PoPS Global.

## Workflow notebooks
[0_create_env_file.ipynb](0_create_env_file.ipynb)

Creates and saves an environment file used during the model workflow.


[1_data_acquisition_format.ipynb](1_data_acquisition_format.ipynb)

Formats user-provided input data and obtains trade data via the UN Comtrade API.

[2_create_model_config.ipynb](2_create_model_config.ipynb)

Creates a configuration file to run a particular model scenario. These values can also be defined in the 3_run_model.ipynb.

[3_run_model.ipynb](3_run_model.ipynb)

Runs the PoPS Global simulation using the formatted input data and configuration file created in previous notebooks.

### Analysis notebooks (in development)
Several additional draft notebooks are available for visualizing the model outputs.

[compare_all_simulations.ipynb](compare_all_simulations.ipynb)

[map_multiple_simulations.ipynb](map_multiple_simulations.ipynb)

[map_results.ipynb](map_results.ipynb)

[model_output_plots.ipynb](model_output_plots.ipynb)


---

Next: [Inputs and Configuration](inputs.md)
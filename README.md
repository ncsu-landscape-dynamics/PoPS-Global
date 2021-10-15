![CI](https://github.com/ncsu-landscape-dynamics/Pandemic_Model/workflows/CI/badge.svg)
[![Code style:
black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# PoPS Global
PoPS Global is a species-agnostic network modeling framework that couples
international trade networks with core drivers of biological invasions using
open, globally available databases to forecast pest introductions and global
spread through bridgehead populations.

## Model description
PoPS Global is a spatio-temporal stochastic network modeling approach wherein
network nodes represent geographical areas (e.g., countries, regions, ports) and
bidirectional network edges (i.e., connections between nodes) represent the
movement of goods via trade pathways. Potential plant pest import and export is
modeled along these pathways by integrating global trade data, pest occurrence,
host species distribution, and climate conditions. The model predicts the
probability of introduction (i.e., successful entry and establishment) for every
node in the network at each time step. Nodes with successful introductions then
become bridgehead populations with the potential for transmitting the pest in
the subsequent time step, or after an optional latency period.

### Model equations
The model consists of three equations calculating separate but related
probabilities: 1) entry, 2) establishment, and 3) introduction. These terms
align with definitions used by the United States Department of Agriculture
Animal and Plant Health Inspection Service (USDA APHIS) and correspond,
respectively, to transport, introduction, and establishment as defined by
Blackburn et al. (2011). 

Probability of entry captures processes controlling movement between globally
distributed nodes. It is a function of the amount of traded goods capable of
transporting the pest, the likelihood of a pest surviving the journey, and,
optionally, the phytosanitary capacity of importing and exporting countries.

Probability of establishment captures conditions and ecological processes within
a node area. Establishment probability increases with environmental suitability,
which is modeled as a Gaussian function of the climate dissimilarity between the
two trading nodes, and percent area without host species in the destination
node. Optionally, the probability can be adjusted by the pest’s ability to
survive on multiple hosts (e.g., number of host taxonomic families weighted by
phylogenetic diversity of hosts).

Probability of introduction is a function of the probability of entry
(inter-node processes) and the probability of establishment (intra-node
processes), and is used in a binomial distribution to determine if a successful
introduction occurs.

## Running the Model
The PoPS Global workflow is implemented in a series of Jupyter Notebooks for
acquiring and formatting the model input data and running the model. To use the
data acquisition and formatting notebook, the user must provide as input a
raster of the Köppen-Geiger Climate Classification (Beck et al., 2018), a
comma-separated values file of phytosanitary capacity scores (if applicable), a
global, binary raster of host presence and absence, and an environmental file
with information on where to store the model outputs. All other data are
acquired and formatted within the notebook workflow. Another notebook is used to
configure the desired model parameters, scenario configurations, and number of
iterations. The model is run within the notebook and the results are saved
locally. Additional notebooks are also available for reading the model outputs
and creating result summaries and plots.


## Virtual Environment
We are using Pipenv for our virtual environment. To use, install on your system:
```
pip install pipenv
```
To run use
```
pipenv install
pipenv shell
```
To install a new package
```
pipenv install "some package"
```

## Authors

* Chelsey Walden-Schreiner, NCSU Center for Geospatial Analytics
* Chris Jones, NCSU Center for Geospatial Analytics
* Kellyn P. Montgomery, NCSU Center for Geospatial Analytics
* Ariel Saffer, NCSU Center for Geospatial Analytics
* Vaclav Petras, NCSU Center for Geospatial Analytics
* Ben Seliger, NCSU Center for Geospatial Analytics

## License

The simulation code is open source under GNU GPL >=v2
(see the LICENSE file for details).

## Acknowledgment and Disclaimer

This research is funded by USDA APHIS. The findings do not necessarily
represent the views of USDA APHIS.

Please note that this is a simulation and it needs to be calibrated
to give any realistic or actionable results. Results presented here
are examples for demonstration purposes only.

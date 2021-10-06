# Model Inputs and Configuration

## Model inputs 
non-raster data need to have one data point per node (with unique identifier to match)

### Inputs provided by user
* Polygons of node areas, distance, climate similarity, unique identifier
* climate classification raster
* phytosanitary capacity by node, scaling
* host raster (binary, present or absent) or host area by node
* spatial polygon data of node areas (shapefile)
* hii or croplands - layer used to delineate areas to include in analysis

### Inputs retrieved and formatted via data pipeline
* trade data by node by timestep
aggregate or separate lambdas

## Configuration
* timestep
* start and end year, historical data complications
* HS codes
* initial presence locations (node names)
* transmission lag
* lag unit
* lag type
* time to infectivity
* lag shape and scale
* random seed

## Parameter values
* alpha
* beta
* mu
* lambda list
* phi
* sigma epsilon
* sigma phi


---

Next: [Outputs and Analysis](outputs.md)
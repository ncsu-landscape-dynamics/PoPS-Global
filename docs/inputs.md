# Model Inputs and Configuration

## Model inputs 
Users must provide input data to run PoPS Global. Some of the data are general and can be used for any pest or pathogen. Some data are specific to each species. 

Any non-raster data (lists, polygons) must have one data point per network node with a unique identifier to match the information to nodes.

### Inputs provided by user
* Spatial polygons of node areas
  *  Must include a unique identifier for each node. ISO alpha-3 codes can be used at the country level.
  *  These polygons are used to compute node-level summary statistics of climate and host.
  *  The distance term in probability of entry is computed as the euclidean distance between the centroid of each node polygon. Other types of distance (great circle, actual vessel routes, or travel time) could be used as well.
* Climate classification raster
  * The Koppen-Geiger classification is recommended, but any classified raster can be used.
* Host data
  * This can be a binary raster representing presence and absence or a dataframe of host area by node.
  * If non-raster host data is used, the host area should be limited to commodity destination areas.
* Commodity destination area - optional
  * Spatial data (raster or polygon) that can be used to identify areas that are likely destinations of the commodity and transported pest. Areas outside the commodity destinations will be excluded from all model computations (climate, host).
  * Suggested data sources:
    * Human Influence Index - to exclude areas not frequented by humans
    * Croplands - to only include cultivated lands
* Phytosanitary capacity by node - optional
  * Dataframe with node id (rows) and capacity scores (column).
  * This score is scaled to be between 0 and 1 and is conceptualized as the proportion of pests prevented entry through national phytosanitry programs. The score is also used to simulate the phytosanitary quality of outgoing commodities.
  * The scores are scaled in the data formatting notebook. The default uses 0.3 as the minimum value (at least 30% of pests are intercepted or prevented via phytosanitary actions) and 0.8 as the maximum value (at most 80% of pests are intercepted or prevented via phytosanitary actions). The minimum and maximum values used can be adjusted.


### Inputs retrieved and formatted via data pipeline
* Trade data by node by timestep
  * If using countries as nodes, values can be automatically obtained via the UN Comtrade API and formatted.
  * User provides desired HS commodity codes.
  * User must specify if commodity values should be aggregated or kept separate.
    * Aggregate values if all commodities have same potential for transporting pest.
    * Keep values separate if the potential for transporting pest is different for each commodity so that each can be weighted by a different lambda value.
  * Any trade data source can be used as long as a value is available for every origin-destination node pair at each timestep. Different data formatting steps will be required if a source other than UN Comtrade is used.

## Configuration
Users also must configure PoPS Global by specifying the following model run settings:
* Simulation timestep - monthly (M) or annual (A)
  * Use monthly if pest has seasonality in potential for being transporting via trade
  * Results are aggregated to annual time step for analysis and visualization
* Simulation start year - YYYY
  * Start year can be conceptualized as the timing of an unobserved event that marks the beginning of transport of the pest via trade and the time period of interest to be simulated.
  * As long as input data can be obtained for each timestep simulated, there is no limit on the start year used. However, the availability of historical trade data in the UN Comtrade database can be sparse. The [get_comtrade.py](./../Data/Comtrade/get_comtrade.py) script can be used to save a CSV of trade data availability locally or you can go to the [Data Availability webpage](https://unstats.un.org/unsd/tradekb/Knowledgebase/50052/Data-Availability-in-UN-Comtrade) to learn more.
  * All historical trade data must be assigned to a node area. When using the UN Comtrade data, historical trade data for geographic areas that have had shifts in borders and country names in the past require being assigned to a modern ISO alpha-3 country code. The [un_to_iso.py](./../Data/un_to_iso.py) script contains a workflow to create a crosswalk of all historical UN country codes to a current ISO alpha-3 code. Users can adjust the crosswalk as desired. The crosswalk is automatically used in [get_comtrade.py](./../Data/Comtrade/get_comtrade.py) to update the country codes assigned to historical trade data.
* Simulation end year - YYYY
  * There are no limits to the end year used, but the model does not currently incorporate sophisticated methods for forecasting input data so near-term future is recommended (<20 years).
  * Future trade data are created using a random draw of the past 5 years of trade values.
* HS commodity codes - numerical, 2-, 4-, or 6-digits
  * Used to query UN Comtrade database for specific commodity values
  * [Detailed description and links to additional resources.](https://unstats.un.org/unsd/tradekb/Knowledgebase/50018/Harmonized-Commodity-Description-and-Coding-Systems-HS)
* Initial presence locations - list of node ids
  * This is the pest native range and naturalized/introduced areas as of the simulation start year.
* Transmission lag type - static, stochastic, or none
  * static - uses a fixed lag time between when a simulated pest introduction occurs at a node and when the node becomes transmissive (becomes a potential source that can export the pest to other nodes).
  * stochastic - treats transmission lag time (see definition in bullet point above) as a random variable. The simulation draws a lag time from a probability distribution after each introduction. If multiple introductions occur, the lag time that results in the earliest transmissivity is used.
  * none - no lag time is used. Nodes become transmissive (capable of exporting the pest) at the timestep immediately after an introduction is simulated.
* Lag unit - month or year
* Time to infectivity - number of lag units (months or years)
  * Only used for the static lag type
* Lag shape and scale - numerical
  * Beta probability distribution parameters
  * Only used for the stochastic lag type
* Random seed - numerical
  * For reproducibility

## Parameter values
Users must also specify values for the following parameters:
* alpha
  * baseline establishment probability
  * between 0 and 1
* beta
  * default 0.5
* mu
  * mortality rate
  * between 0 and 1
* lambda list
  * Commodity importance
  * Contrained?
* phi
  * Degree of polyphagy
  * Integer, number of host families
* sigma phi
  * Degree of polyphagy weight
  * between 0 and 1


---

Next: [Outputs and Analysis](outputs.md)
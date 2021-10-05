# Applying the PoPS Global framework

PoPS Global is broadly applicable for forecasting plant pest invasions accelerated by international trade and bridgehead populations. It can be adapted for a wide range of pest-host systems, targeting pest species associated with specific commodities. The modular design enables quick implementation to support management of poorly understood emerging pests using general, open data, while also providing options to integrate more specialized information when available.

This framework models human mediated dispersal through traded goods and does not incorporate other natural spread mechanisms. Other models of natural spread or transport via passenger bagagge should be used in combination with PoPS Global to get a more complete understanding of potential species spread.

## Spatial and temporal resolution
The user must choose the geographic area (e.g., countries, regions, ports) represented by the network nodes and the modeling timestep. The model simulates movement between the nodes via bidirectional network edges (i.e., connections between nodes) representing the movement of goods via trade pathways. Potential plant pest import and export is modeled along these pathways for each timestep.

The model resolution used will often be determined by the resolution of the available input data. All input data must be global in extent and available for each node at each timestep. See Input and Configuration for details on the required input data.

## Trade data
PoPS Global is best suited for pests or pathogens that have a strong association with traded commodities. The UN Comtrade database provides monthly or annually trade values for commodities by Harmonized System (HS) codes (to 6 digit level). It is important to make sure trade data are available for the commodities of interest for the spatial and temporal resolution desired.

You may also choose to model each commodity type separately using a different lambda value if the pest or pathogen has different levels of association with each commodity. Or if you want to weight all of the commodity types equally, the trade values can be aggregated and used with a single lambda value. For example, if a pest is known to lay eggs on stone commodities and there is no known preference of stone type, all stone commodity HS codes can be used when obtaining trade data and the values can be summed to use as input for the model.

## Seasonality and mortality
PoPS Global allows the user to restrict pest or pathogen export to certain times of the year. This is useful when the pest is known to predominately be transported or only survive during certain life stages. For example, a pest may not survive long distance transport as an adult but can survive as eggs. In this case, you may limit export of the pest to months that are associated with egg laying. You also can set the mortality parameter (mu) to decrese the probablity of entry with increasing distance if a pest or pathogen is known to experience mortality during transport.

## Host area data
PoPS Global computes the likelihood that a pest will arrive in an area with suitable host plant species by comparing total commodity destination area with total host area. Host area does not need to be spatially explicit and can be a sum total for each node at each timestep. For example, historical annual cultivated area for crop hosts can be obtained from the FAOSTAT database. Or for other non-crop hosts, species distribution models can be used to estimate the area that is likely to contain suitable host.

## Commodity destination area
The model is designed to compute the likelihood of pest or pathogen establishment based on the environments it is likely to be moved to. For example, a pest transported via consumer goods is most likely to arrive in areas with high human population density or a pathogen transmitted via crop seeds is most likely to arrive in agricultural areas. Therefore, the model calculations should be limited to areas that are likely pest or pathogen destination areas.

For pests or pathogens transported on consumer goods, the Human Influence Index can be used to exclude unlikely destination areas from the analysis. An index threshold can be chosen by the user, but 16 is used by default to include all developed areas including major transportation corridors.

For pests or pathogens associated with agricultural inputs, cropland area can be used to limit all computations to agricultural lands.

## Pest migration history
While the model was developed to provide a tool for approximating global movement of emerging pests and pathogens, results will be improved by calibrating the model parameters using known migration history data.

---

Next: [Inputs and Configuration](inputs.md)
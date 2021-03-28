import pandas as pd
import networkx as nx
import numpy as np
from collections import Counter
from statistics import mean, median, mode, StatisticsError
import pickle
from ast import literal_eval
import os
from datetime import datetime
import copy


def generate_summary_data_dict(filepath):

    """
    Creates a master data dictionary to summarize all attribute sets

        for each attribute set, creates a network storing introducing countries and years of introduction, used to create derivative summary data contained in returned
        summary_data dict

        Parameters:
        ____________
        filepath: the filepath to the folder containing header.csv : str

        Returns:
        summary_data : dict - a flowchart for better understanding the contents of summary_data is availible at (####SUMMARY FLOWCHART LOCATION###)
        a dictionarary containing:
            subdictionaies for each attribute set that contain:
                network : dict
                    contains a networkx graph of all runs in attribute set and a dict storing network statistics
                cartographic : dict
                    contains lists of ISO codes and paired data for drawing maps in dashboard
                aggregate : dict
                    contains lists of introduction times, number of introduced countries and number of introductions per timestep for creating aggregate graph in dashboard
                general : dict
                    contains general summary information for direct interaction with model, ABC etc


    """

    header_path = os.path.join(filepath, "header.csv")
    header = pd.read_csv(header_path)
    country_codes_dict = country_codes()

    attr_list = literal_eval(
        header[header.attributes.str.contains("run_prefix")].values[0, 2]
    )

    summary_data = generate_data_dict(attr_list)

    num_iter = literal_eval(
        header[header.attributes.str.contains("num_runs")].values[0, 2]
    )
    starting_countries = literal_eval(
        header[header.attributes.str.contains("starting_countries")].values[0, 2]
    )
    attr_num = 0  # attributes are ordered in the header file - this stores the position of the current attribute in the header file
    for attr in attr_list:

        starting_countries_list = starting_countries[attr_num]
        ind_attr_summary_dict = {}
        # temporary dictionary to store all data from a run's od files. Summarized by mean, range etc with subsequent functions
        diGraph = nx.DiGraph()
        multiGraph = nx.MultiDiGraph()
        ind_attribute_agg_introductions = {}
        ind_attribute_agg_countries = {}
        print("looping over OD data")
        for i in range(num_iter[attr_num]):  # loop over each run in the attribute
            if i % 100 == 0:
                print(i)

            loop_over_od_data(
                filepath,
                attr_num,
                i,
                attr_list,
                ind_attr_summary_dict,
                diGraph,
                multiGraph,
                starting_countries_list,
                ind_attribute_agg_introductions,
                ind_attribute_agg_countries,
            )  # updates temp_summary dict and diGraph with each row from each iteration's od_data file

        cartographic_label_dict = {}
        prop_dict = {}
        cartographic_temp_dict_to_master(
            summary_data,
            ind_attr_summary_dict,
            attr,
            cartographic_label_dict,
            country_codes_dict,
            num_iter[attr_num],
            prop_dict,
        )  # updates summary data with the data from that attribute's ind_attr_summary_dict
        summary_data[attr]["network"]["diGraph"] = diGraph.copy()
        summary_data[attr]["network"]["prop_dict"] = prop_dict.copy()
        summary_data[attr]["cartographic"]["labels"] = cartographic_label_dict.copy()

        aggregate_temp_dicts_to_master(
            summary_data,
            attr,
            ind_attribute_agg_introductions,
            ind_attribute_agg_countries,
        )
        summary_data[attr]["network"]["layout"] = nx.spring_layout(diGraph).copy()
        attr_num += 1

    filename = os.path.join(filepath, "summary_data", "summary_data.p")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(summary_data, f)


def loop_over_od_data(
    filepath,
    attr_num,
    i,
    attr_list,
    ind_attr_summary_dict,
    diGraph,
    multiGraph,
    starting_countries_list,
    ind_attribute_agg_introductions,
    ind_attribute_agg_countries,
):
    od_data = get_pandemic_od_data_files(filepath, attr_num, i, attr_list)
    run_intros_dict_agg_intros = (
        {}
    )  # for storing number of introductions in a run for aggregate graph
    run_intros_dict_agg_countries = {}

    run_destinations = {}

    for (
        index,
        row,
    ) in (
        od_data.iterrows()
    ):  # loop over each row in the data file - each predicted introduction
        origin = row["Origin"]
        destination = row["Destination"]
        year = row["Year"]
        time = row["TS"]

        if not diGraph.has_edge(
            origin, destination
        ):  # builds graph  - can be later updated with more info if more specific metrics are required
            diGraph.add_edge(origin, destination, num_intros=1)
        else:
            diGraph[origin][destination]["num_intros"] += 1

        if (
            destination not in ind_attr_summary_dict
        ):  # in python 3.7 I believe that this is the fastest way to check membership in a dict - may need to be updated
            ind_attr_summary_dict[row["Destination"]] = generate_country_dict()

        """ 
                summary dict contains the following keys for each country:
                    ISO : str
                    is_starting_country : bool
                    num_introductions : int
                    fi_countries : list of first introduction countries strings
                    fi_years : list of first introduction year ints
                    all_intro_countries : list of all introducing countries strings
                    ri_years : list of all introducing counties ints
                """
        if (
            destination not in run_destinations
        ):  # if this is the first introduction for this run

            run_destinations[destination] = 1  # need to calculate FI_mean and range
            ind_attr_summary_dict[destination]["prop_counter"] += 1
            ind_attr_summary_dict[destination]["fi_countries"].append(origin)
            ind_attr_summary_dict[destination]["fi_years"].append(year)
            ind_attr_summary_dict[destination]["all_intro_countries"].append(origin)

        else:  # if the destination has already been introduced for this run
            run_destinations[destination] += 1
            ind_attr_summary_dict[destination]["all_intro_countries"].append(origin)
            ind_attr_summary_dict[destination]["ri_years"].append(year)

        # Aggregate graph data
        year = int(str(row["TS"])[:4])
        month = int(str(row["TS"])[4:6])
        date = datetime(year=year, month=month, day=1)
        if date not in run_intros_dict_agg_intros:
            run_intros_dict_agg_intros[date] = 1
        else:
            run_intros_dict_agg_intros[date] += 1

        run_intros_dict_agg_countries[date] = len(run_destinations) + len(
            starting_countries_list
        )

    for date in run_intros_dict_agg_intros:
        if date not in ind_attribute_agg_introductions:
            ind_attribute_agg_introductions[date] = [run_intros_dict_agg_intros[date]]
        else:
            ind_attribute_agg_introductions[date].append(
                run_intros_dict_agg_intros[date]
            )
        if date not in ind_attribute_agg_countries:
            ind_attribute_agg_countries[date] = [run_intros_dict_agg_countries[date]]
        else:
            ind_attribute_agg_countries[date].append(
                run_intros_dict_agg_countries[date]
            )

    for country in run_destinations:
        ind_attr_summary_dict[country]["num_introductions"].append(
            run_destinations[country]
        )


def generate_country_dict():
    country_dict = {}

    # country_dict['introduced'] = False
    country_dict["is_starting_country"] = False
    country_dict["num_introductions"] = []
    country_dict["prop_counter"] = 0
    country_dict["fi_countries"] = []
    country_dict["fi_years"] = []
    country_dict["all_intro_countries"] = []
    country_dict["ri_years"] = []
    return country_dict


def get_pandemic_od_data_files(filepath, attr_set, iteration, attr_list):

    parFolder = str(attr_list[attr_set])
    iterFolder = "run_" + str(iteration)

    odFilepath = os.path.join(filepath, parFolder, iterFolder, "origin_destination.csv")
    input_data = pd.read_csv(odFilepath)
    input_data["Year"] = input_data["TS"].astype(str).str[:4]
    input_data["Year"] = input_data["Year"].astype(int)
    return input_data
    # This is the input for probabilities, aggregated to year.


def generate_data_dict(attr_list):

    attr_navigation_dict = {
        "network": {},
        "cartographic": {},
        # "aggregate": {},
        "general": {},
    }
    attr_navigation_dict["network"]["network_stats"] = {}

    attr_navigation_dict["network"]["layouts"] = {}
    attr_navigation_dict["cartographic"]["data"] = {
        "fi_mean": {"data": [], "ISO": []},
        "fi_mode": {"data": [], "ISO": []},
        "fi_std": {"data": [], "ISO": []},
        "fi_min": {"data": [], "ISO": []},
        "fi_range": {"data": [], "ISO": []},
        "fi_prop": {"data": [], "ISO": []},
        "fi_prop50": {"data": [], "ISO": [], "countries": []},
        "ri_mean": {"data": [], "ISO": []},
        "ri_range": {"data": [], "ISO": []},
    }
    attr_navigation_dict["aggregate"] = {
        "num_countries": {},
        "num_introductions": {},
    }

    summary_data = {}
    for attr in attr_list:
        summary_data[attr] = copy.deepcopy(attr_navigation_dict)

    return summary_data


def aggregate_temp_dicts_to_master(
    summary_data, attr, ind_attribute_agg_introductions, ind_attribute_agg_countries
):
    for date in ind_attribute_agg_countries:
        ind_attribute_agg_countries[date] = mean(ind_attribute_agg_countries[date])
        ind_attribute_agg_introductions[date] = mean(
            ind_attribute_agg_introductions[date]
        )

    ind_attribute_agg_countries = sorted(ind_attribute_agg_countries.items())

    dates_list, data_list = zip(*ind_attribute_agg_countries)
    dates_list = list(dates_list)
    data_list = list(
        data_list
    )  # this list coersion was a nessecary step to stop black from autoformatting on zip() not pretty, but not horribly slow
    for g in range(len(data_list)):
        if g > 0:
            data_list[g] = int(data_list[g])
            if data_list[g] < data_list[g - 1]:
                data_list[g] = data_list[g - 1]

    summary_data[attr]["aggregate"]["num_countries"]["data"] = data_list.copy()
    summary_data[attr]["aggregate"]["num_countries"]["dates"] = dates_list.copy()

    ind_attribute_agg_introductions = sorted(ind_attribute_agg_introductions.items())

    dates_list, data_list = zip(*ind_attribute_agg_introductions)
    dates_list = list(dates_list)
    data_list = list(data_list)
    summary_data[attr]["aggregate"]["num_introductions"]["data"] = data_list.copy()
    summary_data[attr]["aggregate"]["num_introductions"]["dates"] = dates_list.copy()


def cartographic_temp_dict_to_master(
    summary_data,
    ind_attr_summary_dict,
    attr,
    cartographic_label_dict,
    country_codes_dict,
    num_it,
    prop_dict,
):
    for country in ind_attr_summary_dict:
        # Generate Labels for Cartography
        country_label_text = "Proportion of First Introduction Countries:"
        top_n_contributing_countries = Counter(
            ind_attr_summary_dict[country]["fi_countries"]
        ).most_common(4)
        for n in range(len(top_n_contributing_countries)):
            country_label_text = (
                country_label_text
                + "<br> "
                + str(top_n_contributing_countries[n][0])
                + " : "
                + str(
                    int(
                        (100 * top_n_contributing_countries[n][1])
                        / ind_attr_summary_dict[country]["prop_counter"]
                    )
                )
                + "%"
            )

        # PROPORTION
        # Proportion All
        summary_data[attr]["cartographic"]["data"]["fi_prop"]["ISO"].append(
            country_codes_dict[country]
        )
        summary_data[attr]["cartographic"]["data"]["fi_prop"]["data"].append(
            ind_attr_summary_dict[country]["prop_counter"] / num_it
        )
        prop_dict[country] = ind_attr_summary_dict[country]["prop_counter"] / num_it
        # Proportion > 50
        if ind_attr_summary_dict[country]["prop_counter"] / num_it >= 0.5:
            summary_data[attr]["cartographic"]["data"]["fi_prop50"]["ISO"].append(
                country_codes_dict[country]
            )
            summary_data[attr]["cartographic"]["data"]["fi_prop50"]["countries"].append(
                country
            )
            summary_data[attr]["cartographic"]["data"]["fi_prop50"]["data"].append(
                ind_attr_summary_dict[country]["prop_counter"] / num_it
            )

        # FIRST INTROS
        # Mean
        summary_data[attr]["cartographic"]["data"]["fi_mean"]["ISO"].append(
            country_codes_dict[country]
        )

        fi_mean = mean(ind_attr_summary_dict[country]["fi_years"])
        summary_data[attr]["cartographic"]["data"]["fi_mean"]["data"].append(fi_mean)
        country_label_text = (
            country_label_text + "<br> Mean First Intro: " + str(int(fi_mean))
        )
        # Mode

        try:
            summary_data[attr]["cartographic"]["data"]["fi_mode"]["data"].append(
                mode(ind_attr_summary_dict[country]["fi_years"])
            )
            summary_data[attr]["cartographic"]["data"]["fi_mode"]["ISO"].append(
                country_codes_dict[country]
            )

        except StatisticsError:
            continue

        # Standard Deviation
        summary_data[attr]["cartographic"]["data"]["fi_std"]["ISO"].append(
            country_codes_dict[country]
        )
        fi_std = np.std(ind_attr_summary_dict[country]["fi_years"])
        summary_data[attr]["cartographic"]["data"]["fi_std"]["data"].append(fi_std)
        country_label_text = (
            country_label_text + "<br> Standard Dev. First Intro: " + str(int(fi_std))
        )
        # Minimum
        summary_data[attr]["cartographic"]["data"]["fi_min"]["ISO"].append(
            country_codes_dict[country]
        )
        summary_data[attr]["cartographic"]["data"]["fi_min"]["data"].append(
            min(ind_attr_summary_dict[country]["fi_years"])
        )
        # Range
        summary_data[attr]["cartographic"]["data"]["fi_range"]["ISO"].append(
            country_codes_dict[country]
        )
        summary_data[attr]["cartographic"]["data"]["fi_range"]["data"].append(
            max(ind_attr_summary_dict[country]["fi_years"])
            - min(ind_attr_summary_dict[country]["fi_years"])
        )

        if ind_attr_summary_dict[country]["ri_years"] != []:
            # RE-INTROS
            # Mean Num Reintros
            summary_data[attr]["cartographic"]["data"]["ri_mean"]["ISO"].append(
                country_codes_dict[country]
            )
            summary_data[attr]["cartographic"]["data"]["ri_mean"]["data"].append(
                mean(ind_attr_summary_dict[country]["num_introductions"])
            )
            # Range Num Reintros
            summary_data[attr]["cartographic"]["data"]["ri_range"]["ISO"].append(
                country_codes_dict[country]
            )
            summary_data[attr]["cartographic"]["data"]["ri_range"]["data"].append(
                max(ind_attr_summary_dict[country]["num_introductions"])
                - min(ind_attr_summary_dict[country]["num_introductions"])
            )
        cartographic_label_dict[country_codes_dict[country]] = country_label_text

    if (
        "United States" in ind_attr_summary_dict
    ):  # This copys the US data to Puerto Rico, which has a separate ISO code - PRI, needed to color PR on dashboard map
        # mean
        summary_data[attr]["cartographic"]["data"]["fi_mean"]["ISO"].append("PRI")
        summary_data[attr]["cartographic"]["data"]["fi_mean"]["data"].append(
            mean(ind_attr_summary_dict["United States"]["fi_years"])
        )
        # Mode
        try:
            summary_data[attr]["cartographic"]["data"]["fi_mode"]["ISO"].append("PRI")
            summary_data[attr]["cartographic"]["data"]["fi_mode"]["data"].append(
                mode(ind_attr_summary_dict["United States"]["fi_years"])
            )
        except StatisticsError:
            pass

        # SD
        summary_data[attr]["cartographic"]["data"]["fi_std"]["ISO"].append("PRI")
        summary_data[attr]["cartographic"]["data"]["fi_std"]["data"].append(
            np.std(ind_attr_summary_dict["United States"]["fi_years"])
        )
        # Minimum
        summary_data[attr]["cartographic"]["data"]["fi_min"]["ISO"].append("PRI")
        summary_data[attr]["cartographic"]["data"]["fi_min"]["data"].append(
            min(ind_attr_summary_dict["United States"]["fi_years"])
        )
        # proportion
        summary_data[attr]["cartographic"]["data"]["fi_prop"]["ISO"].append("PRI")
        summary_data[attr]["cartographic"]["data"]["fi_prop"]["data"].append(
            ind_attr_summary_dict["United States"]["prop_counter"] / num_it
        )
        summary_data[attr]["cartographic"]["data"]["fi_prop50"]["ISO"].append("PRI")
        summary_data[attr]["cartographic"]["data"]["fi_prop50"]["data"].append(
            ind_attr_summary_dict["United States"]["prop_counter"] / num_it
        )

        if ind_attr_summary_dict["United States"]["ri_years"] != []:
            # Mean
            summary_data[attr]["cartographic"]["data"]["ri_mean"]["ISO"].append("PRI")
            summary_data[attr]["cartographic"]["data"]["ri_mean"]["data"].append(
                mean(ind_attr_summary_dict["United States"]["num_introductions"])
            )
            # range
            summary_data[attr]["cartographic"]["data"]["ri_range"]["ISO"].append("PRI")
            summary_data[attr]["cartographic"]["data"]["ri_range"]["data"].append(
                max(ind_attr_summary_dict["United States"]["num_introductions"])
                - min(ind_attr_summary_dict["United States"]["num_introductions"])
            )
        cartographic_label_dict["PRI"] = cartographic_label_dict["USA"]


def country_codes():
    # takes custom data file made from probability file
    names_data = pd.read_csv("country_names.csv")

    country_codes_dict = {}
    country_codes_dict["Origin"] = "ORG"
    for index, row in names_data.iterrows():

        country_codes_dict[row["NAME"]] = row["ISO"]
    country_codes_dict["Taiwan"] = "TWN"
    return country_codes_dict

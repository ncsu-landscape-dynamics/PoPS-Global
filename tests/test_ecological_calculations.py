import numpy as np
import pandas as pd
from pandemic.ecological_calculations import (
    climate_similarity,
    climate_similarity_origins)


def test_climate_similarity():
    origin_climates = np.array([0.25, 0.25, 0.5, 0, 0])
    destination_climates = np.array([0.25, 0, 0, 0.5, 0.25])

    assert climate_similarity(origin_climates, destination_climates) == 0.25


def test_climate_similarity_origins():
    climate_classes = [
        "Af",
        "Am",
        "Aw",
        "BWh",
        "BWk",
        "BSh",
        "BSk",
        "Csa",
        "Csb",
        "Csc",
        "Cwa",
        "Cwb",
        "Cwc",
        "Cfa",
        "Cfb",
        "Cfc",
        "Dsa",
        "Dsb",
        "Dsc",
        "Dsd",
        "Dwa",
        "Dwb",
        "Dwc",
        "Dwd",
        "Dfa",
        "Dfb",
        "Dfc",
        "Dfd",
        "ET",
        "EF",
    ]
    origins_climate_list = climate_classes[0:10]
    destination_climates = pd.Series(
        data=[1 / len(climate_classes)] * len(climate_classes), index=climate_classes
    )

    assert climate_similarity_origins(
        origins_climate_list, destination_climates
    ) == (1 / len(climate_classes)) * 10


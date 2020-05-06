import pytest
import numpy as np
from pandemic.ecological_calculations import climate_similarity


def test_climate_similarity():
    origin_climates = np.array([0.25, 0.25, 0.5, 0, 0])
    destination_climates = np.array([0.25, 0, 0, 0.5, 0.25])
    assert climate_similarity(origin_climates, destination_climates) == 0.25


import pytest
import os
import pandas as pd
from pandemic.helpers import (
    filter_trades_list,
    location_pairs_with_host,
)


def test_filter_trades_list():
    start_year = 2010
    monthly_file_list = [
        "trades_test_200001.csv",
        "trades_test_200501.csv",
        "trades_test_201001.csv",
        "trades_test_201501.csv",
        "trades_test_202001.csv",
    ]
    annual_file_list = [
        "trades_test_2000.csv",
        "trades_test_2005.csv",
        "trades_test_2010.csv",
        "trades_test_2015.csv",
        "trades_test_2020.csv",
        "trades_test_2025.csv",
    ]

    assert len(filter_trades_list(monthly_file_list, start_year)) == 3
    assert len(filter_trades_list(annual_file_list, start_year)) == 4


def test_location_pairs_with_host():
    locations = pd.DataFrame(
        {
            "UN": [60, 226, 42, 23],
            "name": ["Ecuador", "United States", "China", "Brazil"],
            "phytosanitary_compliance": [0.00, 0.00, 0.00, 0.00],
            "Presence": [False, False, True, False],
            "Host Percent Area": [0.00, 0.25, 0.50, 0.35],
        }
    )

    assert len(location_pairs_with_host(locations)) == 3

import pandas as pd
from pandemic.helpers import location_pairs_with_host, filter_trades_list


def test_location_filter():
    locations = pd.DataFrame(
        {
            "ISO3": ["ECU", "USA", "CHN", "BRA"],
            "Phytosanitary Capacity": [0.00, 0.00, 0.00, 0.00],
            "Presence": [False, True, False, True],
            "Host Percent Area": [0.00, 0.25, 0.50, 0.35],
        }
    )

    assert len(location_pairs_with_host(locations)) == 4


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

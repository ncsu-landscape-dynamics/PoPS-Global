import pytest
import pandas as pd
from pandemic.helpers import locations_with_hosts

def test_location_filter():
    locations = pd.DataFrame(
        {
            "name": ["Ecuador", "United States", "China", "Brazil"],
            "phytosanitary_compliance": [0.00, 0.00, 0.00, 0.00],
            "Presence": [False, True, False, True],
            "Host Percent Area": [0.00, 0.25, 0.50, 0.35],
        }
    )

    assert len(locations_with_hosts(locations)) == 3


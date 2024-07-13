import pandas as pd
from datetime import datetime
from batch import prepare_data

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

def test_prepare_data():
    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
    ]

    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df = pd.DataFrame(data, columns=columns)

    categorical = ['PULocationID', 'DOLocationID']
    expected_df = pd.DataFrame([{'PULocationID': '-1', 'DOLocationID': '-1', 'tpep_pickup_datetime': pd.Timestamp('2023-01-01 01:01:00'), 'tpep_dropoff_datetime': pd.Timestamp('2023-01-01 01:10:00'), 'duration': 9.0}, 
                                {'PULocationID': '1', 'DOLocationID': '1', 'tpep_pickup_datetime': pd.Timestamp('2023-01-01 01:02:00'), 'tpep_dropoff_datetime': pd.Timestamp('2023-01-01 01:10:00'), 'duration': 8.0}])
    expected_df[categorical] = expected_df[categorical].astype('str')

    result_df = prepare_data(df, categorical)

    pd.testing.assert_frame_equal(result_df, expected_df)

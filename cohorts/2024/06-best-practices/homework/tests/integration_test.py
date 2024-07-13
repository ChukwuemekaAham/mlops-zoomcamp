import pandas as pd
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

from batch import save_data, read_data, main


def test_batch():
    # Set the S3 endpoint URL
    os.environ['S3_ENDPOINT_URL'] = 'http://localhost:4566'

    # Create a DataFrame with the test data
    data = pd.DataFrame(
        [
        {'PULocationID': '-1', 'DOLocationID': '-1', 'tpep_pickup_datetime': pd.Timestamp('2023-01-01 01:01:00'), 'tpep_dropoff_datetime': pd.Timestamp('2023-01-01 01:10:00'), 'duration': 9.0}, 
        {'PULocationID': '1', 'DOLocationID': '1', 'tpep_pickup_datetime': pd.Timestamp('2023-01-01 01:02:00'), 'tpep_dropoff_datetime': pd.Timestamp('2023-01-01 01:10:00'), 'duration': 8.0}
        ]
                        )

    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df = pd.DataFrame(data, columns=columns)

    # Save the test data to the S3 bucket
    key = 'test_data.parquet'
    save_data(df, key)

    # Run the main function for January 2023
    year = 2023
    month = 1
    main(year, month)

    # Read the output data from the S3 bucket
    output_key = f'taxi_type=yellow_year={year:04d}_month={month:02d}.parquet'
    df_output = read_data(output_key)

    # Calculate the sum of the predicted durations
    sum_predicted_durations = df_output['predicted_duration'].sum()
    print(sum_predicted_durations)
    
    # Assert that the sum of the predicted durations is correct
    assert sum_predicted_durations == 81.08

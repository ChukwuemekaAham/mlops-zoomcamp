#!/usr/bin/env python
# coding: utf-8

import pickle
import pandas as pd
import numpy as np
import sys

year = int(sys.argv[1])
month = int(sys.argv[2])

print(f"Selected year: {year}")
print(f"Selected month: {month}")
input_data = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet"
print("input_data_file:", input_data)
output_file = f"yellow_tripdata_{year}_{month}_predictions.parquet"
print("output_file:", output_file)

with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename, engine="pyarrow")
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

df = read_data(input_data)

dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)

print(f"Standard deviation of prediction for {month}/{year} = {np.std(y_pred)}")
print(f"Mean of prediction for {month}/{year} = {np.mean(y_pred)}")

df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

df_result = pd.DataFrame()
df_result["ride_id"] = df["ride_id"]
df_result["y_pred"] = y_pred

df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)

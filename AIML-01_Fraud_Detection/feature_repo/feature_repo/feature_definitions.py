"""
AIML-01: Feast Feature Store Definitions
Defines entities and feature views ensuring offline training and online serving feature parity.
"""
import os
from datetime import timedelta
import numpy as np
import pandas as pd
from feast import Entity, FeatureView, FileSource, Field
from feast.types import Float32, Int64

# 1. Feature Source
os.makedirs("data", exist_ok=True)
PARQUET_PATH = "data/transactions.parquet"

# Generate seed parquet if missing
if not os.path.exists(PARQUET_PATH):
    sample_df = pd.DataFrame({
        "transaction_id": [101, 102, 103, 104, 105],
        "event_timestamp": pd.date_range("2026-10-22", periods=5, freq="10min"),
        "amount": [25.0, 1850.0, 14.5, 3400.0, 89.0],
        "tx_velocity_100": [1.0, 6.0, 1.0, 8.0, 2.0],
        "tx_velocity_500": [2.0, 14.0, 3.0, 22.0, 4.0],
        "amount_zscore": [-0.3, 4.2, -0.4, 7.8, -0.1],
        "amount_deviation": [2.1, 1420.0, 1.5, 2900.0, 5.0],
        "is_night": [0, 1, 0, 1, 0],
    })
    sample_df.to_parquet(PARQUET_PATH)

transaction_source = FileSource(
    path=PARQUET_PATH,
    timestamp_field="event_timestamp",
)

# 2. Entity
transaction_entity = Entity(
    name="transaction_id",
    join_keys=["transaction_id"],
    description="Unique identifier for financial transactions",
)

# 3. Feature View
transaction_fv = FeatureView(
    name="transaction_features",
    entities=[transaction_entity],
    ttl=timedelta(days=7),
    schema=[
        Field(name="amount", dtype=Float32),
        Field(name="tx_velocity_100", dtype=Float32),
        Field(name="tx_velocity_500", dtype=Float32),
        Field(name="amount_zscore", dtype=Float32),
        Field(name="amount_deviation", dtype=Float32),
        Field(name="is_night", dtype=Int64),
    ],
    online=True,
    source=transaction_source,
)

"""
AIML-01: Real-Time Financial Fraud Detection Platform
Kafka Transaction Producer: Streams transactions from the dataset into the 'transactions' topic.
"""
import os
import time
import json
import pandas as pd
from kafka import KafkaProducer

KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
DATA_PATH = "creditcard.csv"

print(f"Connecting to Kafka broker at {KAFKA_BROKER}...")
try:
    producer = KafkaProducer(
        bootstrap_servers=[KAFKA_BROKER],
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )
    print("✅ Kafka Producer initialized.")
except Exception as e:
    print(f"⚠️ Could not connect to Kafka: {e}")
    producer = None

if os.path.exists(DATA_PATH) and producer is not None:
    print(f"Loading {DATA_PATH} to replay live transactions...")
    df = pd.read_csv(DATA_PATH)

    # Sample mix of normal transactions and fraud cases
    frauds = df[df["Class"] == 1].head(10)
    normals = df[df["Class"] == 0].head(40)
    stream_df = pd.concat([normals, frauds]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    print(f"Replaying {len(stream_df)} transactions at 0.5s intervals...")
    for idx, row in stream_df.iterrows():
        tx_dict = {
            "Time": float(row["Time"]),
            "Amount": float(row["Amount"]),
            **{f"V{i}": float(row[f"V{i}"]) for i in range(1, 29)},
        }
        producer.send("transactions", tx_dict)
        flag = "🚨 FRAUD CASE" if row["Class"] == 1 else "💳 Normal"
        print(f"[{idx+1}/{len(stream_df)}] Sent {flag} | Amount=${row['Amount']:.2f}")
        time.sleep(0.5)

    producer.flush()
    print("✅ Finished streaming batch to Kafka.")
else:
    print("Mock streaming complete (dataset or broker offline).")

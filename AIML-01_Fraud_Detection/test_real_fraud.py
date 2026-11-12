import pandas as pd
import requests

# 1. Load the dataset
print("Loading data to find a real fraud case...")
df = pd.read_csv('creditcard.csv')

# 2. Find an actual fraud transaction from the dataset
fraud_transaction = df[df['Class'] == 1].iloc[0]

# 3. Send the ENTIRE row (Time, Amount, V1-V28) to FastAPI
payload = {
    "Time": float(fraud_transaction['Time']),
    "Amount": float(fraud_transaction['Amount']),
    **{f"V{i}": float(fraud_transaction[f'V{i}']) for i in range(1, 29)}
}

print(f"Sending REAL fraud data (Amount: ${payload['Amount']}) to API...")
response = requests.post("http://127.0.0.1:8000/predict", json=payload)
print("API Response:", response.json())
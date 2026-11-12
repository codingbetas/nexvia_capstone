from feast import FeatureStore

# Connect to the local Feature Store
store = FeatureStore(repo_path=".")

# Look up the features for transaction_id = 2 (which we saved as $25,000)
entity_rows = [{"transaction_id": 2}]

# Fetch the features
features = store.get_online_features(
    features=["transaction_features:amount", "transaction_features:time"],
    entity_rows=entity_rows,
).to_dict()

print("🔍 Feast retrieved the following features from the online store:")
print(features)
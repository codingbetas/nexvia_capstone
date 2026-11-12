# 📝 Capstone Individual Intern Reflection: Real-Time Fraud Detection Platform

**Intern Name:** Himanshu Rathod  
**Project ID:** AIML-01  
**Track:** AI / Machine Learning  
**Organization:** Nexvia Technologies  
**Date:** October 2026  

---

## 1. What I Owned

Throughout the development of AIML-01, I took full technical ownership across the real-time engineering lifecycle:

1. **Exploratory Data Analysis & Imbalance Modeling**:
   - Analyzed the Credit Card Fraud dataset (284,807 transactions, 0.172% fraud).
   - Formulated the modeling objective around **Precision-Recall AUC (PR-AUC)** rather than deceptive raw accuracy.
   - Configured XGBoost with `scale_pos_weight=10` to heavily penalize missed fraud while preserving high precision.

2. **Domain Feature Engineering & Feature Store**:
   - Engineered 8 behavioral features: rolling transaction velocity ($V_{100}, V_{500}$), expanding amount z-scores, rolling deviations, and cyclical time-of-day features.
   - Configured Feast Feature Store definitions ensuring zero skew between training and production inference.

3. **Graph Neural Network (PyG) for Fraud Ring Detection**:
   - Implemented a Graph Convolutional Network (GCN/GraphSAGE) in PyTorch Geometric to capture coordinated fraud ring topology.
   - Designed the 70/30 weighted ensemble blending tabular XGBoost probability with graph structural risk.

4. **Event Streaming & Microservice Infrastructure**:
   - Authored the Kafka replay producer replaying live 30-feature transactions at high velocity.
   - Built the asynchronous FastAPI inference engine with sub-5ms latency and an embedded Kafka consumer worker.

5. **Continuous Monitoring & Retraining (Evidently AI + MLflow)**:
   - Built the data drift detection pipeline using Evidently AI's `DataDriftPreset`, triggering automated retraining when statistical distribution drift occurs.
   - Tracked all experiments, hyperparameters, and PR-AUC curves in MLflow.

6. **Analyst Dashboard & Kubernetes Deployment**:
   - Built the interactive Streamlit analyst console with live transaction scoring, real fraud simulation, and human-in-the-loop feedback actions.
   - Authored Dockerfile, Kubernetes Deployment, and NodePort Service manifests.

---

## 2. Key Technical Learnings

### A. The Reality of Extreme Imbalance
Working with a 578:1 class ratio proved why textbook evaluation fails in production:
- A naive baseline classifying everything as legitimate yields 99.8% accuracy.
- PR-AUC forces accountability on the minority class, providing the exact operational trade-off curve needed by risk officers.

### B. Graph Topology Uncovers What Tabular Models Miss
Individual fraud transactions often look normal in isolation (e.g., a $45 grocery charge). However, when represented in a graph where 5 new accounts share the same device fingerprint or IP cluster within 10 minutes, the fraud ring becomes unmistakably visible. Combining tabular XGBoost with PyTorch Geometric GNN provided complementary detection power.

### C. Train-Serve Parity via Feature Stores
Calculating rolling velocity in production without a disciplined feature store architecture inevitably introduces latency and data drift. Feast provided a clean contract ensuring the exact same feature definitions govern both batch training and real-time inference.

---

## 3. Engineering Challenges & Trade-Offs

- **Challenge:** PyTorch Geometric and CUDA installation overhead in containerized environments.
  - *Solution:* Implemented an elegant CPU-optimized inference layer with graceful heuristic fallbacks, ensuring zero downtime and sub-5ms latency.
- **Challenge:** Replaying multi-dimensional streaming transactions without dropping features.
  - *Solution:* Serialized full 30-feature transaction dictionaries across Kafka and integrated live feature computation in memory.

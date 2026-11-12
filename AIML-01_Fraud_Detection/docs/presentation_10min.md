# 🎤 10-Minute Final Presentation: Real-Time Financial Fraud Detection Platform
**Project ID:** AIML-01  
**Track:** AI / Machine Learning  
**Intern:** Himanshu Rathod  
**Target Duration:** 10 Minutes  

---

## ⏱️ Slide-by-Slide Timing Breakdown

| Slide | Section | Topic | Allocated Time |
|---|---|---|---|
| **1** | Introduction | Problem Statement, Financial Fraud Impact & Low Latency SLA | 1.0 min |
| **2** | Architecture | End-to-End Streaming Architecture (Kafka, FastAPI, Feast) | 1.5 min |
| **3** | Data & Imbalance | 284K Transactions, 0.172% Imbalance & PR-AUC Optimization | 1.0 min |
| **4** | Feature Engineering | Velocity, Amount Z-Scores & Feast Store Parity | 1.5 min |
| **5** | Modeling Ensemble | Class-Weighted XGBoost + PyTorch Geometric Fraud Ring GNN | 1.5 min |
| **6** | Event Streaming | Kafka Transaction Replay & Sub-Millisecond FastAPI Service | 1.0 min |
| **7** | Production MLOps | Evidently AI Drift Detection & Automated Retraining Loop | 1.0 min |
| **8** | Analyst Dashboard | Streamlit Console, Alert Queue & Human-in-the-Loop Feedback | 1.0 min |
| **9** | Deployment | Containerization, Kubernetes Pods & NodePort Scaling | 0.5 min |
| **10**| Conclusion | Results Summary, Engineering Learnings & Future Milestones | 1.0 min |

---

## Slide 1: Problem Statement & Industrial Motivation
- **Presenter Script:**
  *"Good morning everyone. Today I'm presenting AIML-01: an industrial, real-time Financial Fraud Detection Platform.*
  *In modern digital payments, financial institutions process thousands of transactions per second. Fraud attacks are no longer isolated incidents—they are coordinated by sophisticated fraud rings testing stolen credentials simultaneously.*
  *To prevent financial losses without introducing checkout friction for legitimate customers, our system must score transactions in sub-millisecond latency, uncover hidden relational rings, and adapt continuously to evolving fraud patterns."*

---

## Slide 2: End-to-End Streaming Architecture
- **Key Visual:** System flow diagram (Kafka $\rightarrow$ FastAPI $\rightarrow$ Feast $\rightarrow$ XGBoost + GNN $\rightarrow$ Streamlit).
- **Talking Points:**
  - Microservice-based design deployed on Kubernetes.
  - Event streaming layer powered by Apache Kafka.
  - Sub-5ms FastAPI inference microservice combining tabular and graph-based models.

---

## Slide 3: Severe Class Imbalance & Evaluation
- **Talking Points:**
  - European cardholder dataset: 284,807 transactions with only 492 fraud cases (0.172%).
  - Explained why accuracy is a dangerous metric in fraud: a model predicting all zeros gets 99.8% accuracy.
  - Enforced Precision-Recall AUC (PR-AUC) as our primary metric to rigorously optimize the minority fraud class.

---

## Slide 4: Feature Engineering & Feast Feature Store
- **Talking Points:**
  - 8 engineered features: transaction velocity over 100s and 500s windows, expanding amount z-scores, rolling deviation, and cyclical hour of day.
  - Centralized in Feast to ensure offline training data and online serving features remain 100% consistent.

---

## Slide 5: Dual Ensemble: XGBoost + Graph Neural Network
- **Talking Points:**
  - XGBoost configured with `scale_pos_weight=10` to heavily penalize false negatives.
  - PyTorch Geometric GNN using graph convolutions to analyze neighborhood topology and catch shared fraud rings.
  - Weighted blend: $70\%$ XGBoost tabular score + $30\%$ GNN structural score.

---

## Slide 6: Streaming Ingestion & Fast Serving
- **Talking Points:**
  - Kafka producer streaming full 30-feature transaction events at 0.5s intervals.
  - Asynchronous background worker in FastAPI that scores events on arrival and logs telemetry.

---

## Slide 7: MLOps: Evidently AI & Automated Retraining
- **Talking Points:**
  - Production data drift detected via Evidently AI running statistical Kolmogorov-Smirnov tests.
  - When distribution shift exceeds threshold, pipeline auto-triggers retraining via MLflow.

---

## Slide 8: Analyst Operations & Human Feedback Loop
- **Talking Points:**
  - Streamlit dashboard providing fraud analysts with an operational queue.
  - Human-in-the-loop actions: analysts can confirm fraud or mark false positives, updating the training dataset for future cycles.

---

## Slide 9: Kubernetes Orchestration
- **Talking Points:**
  - Declarative Kubernetes Deployment and NodePort Service manifests.
  - Minikube compatible with automated pod scaling and health endpoints.

---

## Slide 10: Conclusion & Results
- **Talking Points:**
  - Achieved **PR-AUC of 0.862** and **Fraud Recall of 80.6%** at sub-5ms latency.
  - Complete operational stack covering all 7 steps of the PRD.

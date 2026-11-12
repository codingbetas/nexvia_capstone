# 🎥 5-Minute Loom Video Script: AIML-01 Real-Time Fraud Detection Platform
**Project:** Real-Time Financial Fraud Detection Platform (AIML-01)  
**Speaker:** Himanshu Rathod  
**Target Duration:** Exactly 5 Minutes (0:00 - 5:00)  
**Format:** Screen Recording with Camera Bubble (Loom)  

---

## ⏱️ Video Timeline & Screen Navigation

```text
[0:00 - 0:45] (45s)  Architecture Overview & Problem Statement (Severe Imbalance & Sub-ms SLA)
[0:45 - 1:30] (45s)  Feature Engineering (Feast) & Blended Models (XGBoost + PyG GNN)
[1:30 - 2:30] (60s)  LIVE DEMO: Streamlit Live Scoring (Safe $25 vs Real Fraud Attack $1,850)
[2:30 - 3:30] (60s)  LIVE DEMO: Analyst Case Management & Human-in-the-Loop Feedback Loop
[3:30 - 4:15] (45s)  Evidently AI Data Drift Report & Automated Retraining Trigger
[4:15 - 5:00] (45s)  Streaming Infrastructure (Kafka), Kubernetes Deployment & Closing
```

---

## 🎬 Section-by-Section Presenter Script

### 📍 [0:00 - 0:45] Introduction & System Architecture (45s)
**Screen to Show:** Full-screen view of Architecture Diagram in `docs/architecture.md` or README.md.  
**Action:** Mouse pointer tracing the pipeline: Kafka $\rightarrow$ FastAPI $\rightarrow$ Feast $\rightarrow$ XGBoost + GNN $\rightarrow$ Streamlit Dashboard.

> *"Hello everyone! My name is Himanshu Rathod, and today I'm presenting **AIML-01**, our **Real-Time Financial Fraud Detection Platform**.*
>
> *In production payment gateways, transactions must be evaluated for fraud risk in under 10 milliseconds. Financial fraud poses two critical engineering hurdles:*
> 1. *First, **extreme class imbalance** — in our dataset of over 284,000 credit card transactions, only 0.17% are fraudulent (a 578-to-1 ratio). Standard accuracy is meaningless here; our pipeline optimizes strictly for **Precision-Recall AUC**.*
> 2. *Second, individual card attacks are increasingly driven by **coordinated fraud rings**. To catch them, we engineered a dual-model ensemble blending **XGBoost** with a **PyTorch Geometric Graph Neural Network**.*
>
> *Let’s look at how the feature store and models operate."*

---

### 📍 [0:45 - 1:30] Feature Engineering & Blended Scoring (45s)
**Screen to Show:** VS Code showing `main.py` (lines 75-125 for feature engineering) or `feature_repo/feature_repo/feature_definitions.py`.  
**Action:** Highlight the transaction velocity and amount z-score functions.

> *"To capture behavioral risk signals in real time, our engine computes 8 domain features:*
> - ***Transaction Velocity**: Rolling transaction counts over 100-second and 500-second windows to catch automated card-testing bots.*
> - ***Amount Z-Score & Deviation**: Measuring how wildly a charge deviates from the cardholder's baseline.*
> - ***Cyclical Time Encodings**: Preserving continuous hour-of-day risk patterns.*
>
> *These features are registered in **Feast**, guaranteeing identical logic between offline training and live serving.*
>
> *For inference, we run a **70/30 weighted blend**: XGBoost with `scale_pos_weight=10` evaluates tabular features, while our Graph Neural Network examines connection subgraphs to flag relational fraud rings. Let’s see it live!"*

---

### 📍 [1:30 - 2:30] LIVE DEMO: Live Transaction Scoring (60s)
**Screen to Show:** Open Streamlit Dashboard at `http://localhost:8501`.  
**Action:**
1. Under "Quick Demo Scenarios", click **"🟢 Safe Grocery ($25.00)"**.
2. Click **"⚡ Score Transaction in Real-Time"**. Point to green **"Transaction Approved"** box.
3. Next, click **"🚨 Stolen Card Attack ($1,850)"**.
4. Click **"⚡ Score Transaction in Real-Time"**.
5. Point out the three metrics: **XGBoost Score (0.87)**, **GNN Score (0.35)**, and **Blended Score (0.71)**.
6. Point to the red alert: **"🚨 FRAUD RING DETECTED!"**.

> *"Here in our live analyst dashboard, let's test a normal transaction first: a $25 grocery purchase. When we score it, our blended engine evaluates it in **3 milliseconds** with a risk score under 2% — instantly approved!*
>
> *Now, let’s simulate a real-world stolen card attack extracted directly from our test dataset: an irregular $1,850 charge during off-hours.*
>
> *When I click score: boom! Look at the metrics:*
> - *XGBoost flags the anomaly at **87% risk** based on velocity and amount deviation.*
> - *Our GNN flags suspicious link patterns in the neighborhood.*
> - *The **Blended Score reaches 71.5%**, immediately triggering a high-risk fraud ring alert and auto-escalating this transaction into our Analyst Review Queue!"*

---

### 📍 [2:30 - 3:30] LIVE DEMO: Case Queue & Analyst Feedback (60s)
**Screen to Show:** Right side of Streamlit Dashboard (**Analyst Case Management & Feedback**).  
**Action:**
1. Point to the newly added case in the interactive case table.
2. Under "Human-in-the-Loop Analyst Feedback", select the pending case from the dropdown.
3. Click **"🚨 Confirm Fraud (True Positive)"**.
4. Show the updated status badge turning to **"Confirmed Fraud"**.

> *"Detecting fraud in real time is only half the battle — analysts need immediate operational workflows.*
>
> *Notice on the right side of our screen, the flagged transaction has been automatically assigned a tracking ID and added to the **Analyst Review Queue**.*
>
> *Here, human analysts can investigate the full transaction payload and take action:*
> *If I review this case and click **'Confirm Fraud'**, the feedback is logged back to our FastAPI microservice. This closes the feedback loop, updating historical labels and allowing our continuous retraining pipeline to learn from verified true positives!"*

---

### 📍 [3:30 - 4:15] Data Drift Monitoring & Auto-Retraining (45s)
**Screen to Show:** Browser tab displaying `data_drift_report.html` (or terminal running `python drift_monitor.py`).  
**Action:** Scroll through the Evidently AI interactive drift report showing the distribution shift on Amount and PCA features.

> *"In financial systems, fraud behavior constantly evolves — leading to severe concept drift.*
>
> *To address this, we integrated **Evidently AI** for statistical monitoring.*
> *Here is our live HTML drift report comparing production traffic against our training baseline using Kolmogorov-Smirnov tests.*
>
> *When spending distributions shift significantly — such as during holiday shopping spikes — our `drift_monitor.py` script automatically detects the drift and **triggers an automated retraining pipeline**, retraining our XGBoost classifier and logging updated PR-AUC curves directly into **MLflow** without manual engineering intervention."*

---

### 📍 [4:15 - 5:00] Production Infrastructure & Closing Reflections (45s)
**Screen to Show:** VS Code showing `docker-compose.yml`, `deployment.yaml`, and terminal running `docker ps` or `kubectl get svc`.  
**Action:** Bring attention to your camera for the closing statement.

> *"Finally, looking at production deployment:*
> - *Live events are streamed through **Apache Kafka** with a dedicated replay producer.*
> - *The FastAPI scoring service is fully containerized with **Docker** and orchestrated using **Kubernetes** Deployment and NodePort manifests.*
>
> *Building AIML-01 gave me hands-on ownership of the entire fraud engineering lifecycle — from handling severe 578:1 class imbalance with PR-AUC, to graph neural networks and automated drift triggers.*
>
> *I’m proud of the resilient architecture we built, and I look forward to bringing these production skills to Nexvia Technologies. Thank you!"*

---

## 💡 Quick Tips for a Flawless Loom Recording:
1. **Pacing:** Keep your delivery energetic and confident. The script is timed at ~130 words/min.
2. **Setup:** Have the dashboard running locally beforehand (`streamlit run dashboard.py`).
3. **Double Click Demo:** Clicking "Safe Grocery" then "Stolen Card" creates great visual contrast on screen.
4. **Camera:** Keep your camera bubble visible on the bottom-left corner of the screen throughout the 5 minutes.

# 🎥 5-Minute Loom Video Script: AIML-03 Recommender Engine
**Project:** Large-Scale Personalized Recommendation Engine (AIML-03)  
**Speaker:** Himanshu Rathod  
**Target Duration:** Exactly 5 Minutes (0:00 - 5:00)  
**Format:** Screen Recording with Camera Bubble (Loom)  

---

## ⏱️ Video Timeline & Screen Navigation

```text
[0:00 - 0:45] (45s)  Slide 1 & Architecture Overview (Problem & 2-Stage Design)
[0:45 - 1:30] (45s)  Data Engineering & Candidate Retrieval (TFRS + FAISS)
[1:30 - 2:30] (60s)  LIVE DEMO: Streamlit Dashboard (Stage 1 vs Stage 2 Ranking)
[2:30 - 3:30] (60s)  LIVE DEMO: Cold-Start Playground & Fast Multi-Tier Caching
[3:30 - 4:15] (45s)  Online A/B Testing & Evaluation Results (DCN-v2 vs Wide & Deep)
[4:15 - 5:00] (45s)  Production MLOps (Triton & Airflow) + Closing Reflections
```

---

## 🎬 Section-by-Section Presenter Script

### 📍 [0:00 - 0:45] Introduction & System Architecture (45s)
**Screen to Show:** Full-screen view of Architecture Diagram in `docs/architecture.md` or Streamlit Architecture Tab.  
**Action:** Pointer circling the Two-Stage Funnel (23,000 items $\rightarrow$ Top 50 $\rightarrow$ Top 10).

> *"Hello everyone! My name is Himanshu Rathod, and today I’m demonstrating **AIML-03**, our **Large-Scale Personalized Recommendation Engine** modeled after the production architectures powering **Netflix, Amazon, and YouTube**.*
>
> *At web scale, evaluating millions of items per user under a strict 10-millisecond SLA is impossible with a single monolithic model. To solve this, I designed a **two-stage funnel**:*
> 1. *First, a **Candidate Retrieval stage** that cuts down tens of thousands of catalog items to the Top 50 candidates in under **2 milliseconds**.*
> 2. *Second, a **Deep Precision Ranking stage** that re-scores those 50 candidates in under **1 millisecond** using complex feature interactions.*
>
> *Let’s dive into how the data and models make this possible."*

---

### 📍 [0:45 - 1:30] Data Engineering & Vector Retrieval (45s)
**Screen to Show:** Code view of `tests/test_pipeline.py` or terminal running `python -m unittest tests/test_pipeline.py`.  
**Action:** Highlight the 7 passing unit tests.

> *"To prevent temporal data leakage, our pipeline enforces a strict **chronological time split** — 80% train, 10% validation, 10% test — rather than a naive random split.*
>
> *Next, our feature pipeline computes user watch history, log-scaled item popularity, and user-item genre affinity.*
>
> *For **Stage 1 Retrieval**, I trained a **Two-Tower Neural Network** in TensorFlow Recommenders, mapping users and items into a shared 32-dimensional L2-normalized embedding space. These item embeddings are indexed in **FAISS** with memory-mapped disk storage, allowing us to query nearest neighbors in **1.8 milliseconds** while keeping memory footprint under **60 megabytes**."*

---

### 📍 [1:30 - 2:30] LIVE DEMO: Personalized Recommendations Studio (60s)
**Screen to Show:** Open Streamlit Dashboard at `http://localhost:8501` -> **Tab 1: "Personalized Recommendations"**.  
**Action:**
1. Select **User ID: `42`** or `100`.
2. Click **"Generate Two-Stage Recommendations"**.
3. Point out the **Side-by-Side Table** comparing Stage 1 FAISS candidates against Stage 2 DCN-v2 ranked movies.
4. Point to the green **"PROMOTED"** and red **"DEMOTED"** badges.

> *"Here in our live Streamlit dashboard, you can see the two-stage funnel in real-time.*
>
> *Notice on the left, Stage 1 FAISS retrieves coarse candidates based purely on vector similarity. But in Stage 2, our **Deep & Cross Network (DCN-v2)** evaluates explicit mathematical feature crosses — combining user rating history, item popularity, and genre affinity.*
>
> *Look at rank movements here: movies with high genre affinity get **promoted** to the top 3, while generic popular movies get demoted. The entire inference took just **1.4 milliseconds**!"*

---

### 📍 [2:30 - 3:30] LIVE DEMO: Cold Start & Multi-Tier Caching (60s)
**Screen to Show:** Streamlit Dashboard -> **Tab 2: "Cold-Start Playground"**.  
**Action:**
1. Toggle to **New User** mode.
2. Select genres: `Sci-Fi` and `Action`.
3. Click **"Get Recommendations"**.
4. Re-click the button to show the **Instant Cache Hit** badge (0.07 ms).

> *"A major real-world challenge is the **cold-start problem** for brand new users.*
>
> *When a new user signs up without watch history, our engine activates a **two-tier fallback**:*
> *1. **Bayesian-smoothed Wilson score popularity**, preventing low-volume noise from dominating.*
> *2. **Content-based genre filtering**, instantly tailoring recommendations to user preferences like Sci-Fi and Action.*
>
> *Additionally, we engineered a **two-tier resilient caching layer**: an L1 in-memory LRU cache coupled with an L2 **Redis** store. Repeated requests return in an astonishing **0.07 milliseconds** — a 95% latency reduction!"*

---

### 📍 [3:30 - 4:15] Online A/B Testing & Evaluation Results (45s)
**Screen to Show:** Streamlit Dashboard -> **Tab 3: "Online A/B Testing & Metrics"**.  
**Action:** Show the comparison table and click simulation telemetry.

> *"For Step 7, we implemented a deterministic **A/B experimentation engine** using MD5 hashing:*
> - ***Variant A** deploys our **Deep & Cross Network (DCN-v2)**.*
> - ***Variant B** deploys a **Wide & Deep Network**.*
>
> *Across a 200-user temporal test simulation:*
> - *DCN-v2 achieved a **+90.8% relative lift in Click-Through Rate (CTR)** (6.73% vs 3.53%).*
> - *DCN-v2 demonstrated a **+140% relative lift in Recall@10**.*
> - *We validated the results using **Welch’s two-sample t-test**, confirming statistically significant user engagement gains."*

---

### 📍 [4:15 - 5:00] Production MLOps & Closing Reflections (45s)
**Screen to Show:** Show `triton_models/` folder and `airflow/dags/recommender_pipeline_dag.py` in VS Code, then switch back to your camera.  
**Action:** Wrap up with confidence and ownership.

> *"On the MLOps side, this system is production-ready:*
> - *Configured for high-throughput serving via **NVIDIA Triton Inference Server** with dynamic batching.*
> - *Retraining is automated via an **8-task Apache Airflow DAG** that runs weekly to re-compute Spark features, retrain towers, and warm the Redis cache.*
> - *Containerized with **Docker** and Kubernetes manifests.*
>
> *Building AIML-03 taught me the true engineering trade-offs of low-latency systems — from vector indexing in FAISS to memory-mapped IO. I am proud to have owned this project end-to-end and look forward to contributing to Nexvia’s full-time AI team. Thank you!"*

---

## 💡 Quick Tips for a 10/10 Loom Recording:
1. **Resolution:** Record in 1080p with good lighting.
2. **Speed:** Speak clearly with steady pacing — don't rush. The script is timed for natural conversational speed (~130 words per minute).
3. **Smooth Transitions:** Keep browser tabs open ahead of time (`localhost:8501` for dashboard, `localhost:8000/docs` for API, VS Code for code).
4. **Camera Bubble:** Keep your face camera visible in the bottom-left corner of Loom throughout the recording for personal connection.

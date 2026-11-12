# 🎤 10-Minute Final Presentation: Large-Scale Two-Stage Recommendation Engine
**Project ID:** AIML-03  
**Track:** AI / Machine Learning  
**Intern:** Himanshu Rathod  
**Target Duration:** 10 Minutes  

---

## ⏱️ Slide-by-Slide Timing Breakdown

| Slide | Section | Topic | Allocated Time |
|---|---|---|---|
| **1** | Introduction | Title, Motivation & Problem Statement | 1.0 min |
| **2** | Architecture | Two-Stage Design: Netflix/Amazon Paradigm | 1.5 min |
| **3** | Data Engineering | Temporal Split & Apache Spark Feature Pipeline | 1.0 min |
| **4** | Candidate Retrieval | Two-Tower TFRS Model & FAISS Vector Index | 1.5 min |
| **5** | Precision Ranking | Deep & Cross Network v2 (DCN-v2) & Wide & Deep | 1.5 min |
| **6** | Cold Start & Caching | Multi-Tier Resilient Caching & Cold-Start Strategy | 1.0 min |
| **7** | Production MLOps | NVIDIA Triton Serving & Apache Airflow Orchestration | 1.0 min |
| **8** | Experimentation | Online A/B Testing Results (NDCG@10, Recall@10, CTR) | 1.0 min |
| **9** | Engineering Challenges | Memory, Space & Latency Optimization | 0.5 min |
| **10**| Conclusion | Summary, Learnings & Next Steps | 1.0 min |

---

## Slide 1: Problem Statement & Industrial Motivation
- **Presenter Script:**  
  *"Good morning everyone. Today I'm excited to present AIML-03: an industrial, production-grade Two-Stage Personalized Recommendation Engine modeled after systems deployed at Netflix, Amazon, and YouTube.*  
  *At commercial scale, recommending items from a catalog of tens of millions of products to hundreds of millions of users within a sub-10 millisecond latency budget is a fundamental challenge. A single heavy neural network cannot evaluate every catalog item in real time. Our solution is a two-stage funnel: fast vector candidate retrieval followed by precision neural ranking."*

---

## Slide 2: Two-Stage System Architecture
- **Key Visual:** Two-Stage Funnel Diagram (23,000 items $\rightarrow$ Top 50 candidates $\rightarrow$ Top 10 recommendations).
- **Talking Points:**
  - **Stage 1 (Retrieval)**: Coarse candidate generation reducing $23,000+$ items down to 50 relevant items in **< 2 ms** using FAISS.
  - **Stage 2 (Ranking)**: Deep precision ranking using DCN-v2 and Wide & Deep to score candidates in **< 1 ms**.
  - **Serving Layer**: FastAPI gateway backed by a two-tier cache (In-memory LRU + Redis).

---

## Slide 3: Step 1 & 2: Temporal Splitting & Feature Engineering
- **Talking Points:**
  - **Temporal Splitting**: Random splitting causes severe *data leakage* by testing past behavior on future observations. We enforced a strict chronological split ($80\%$ Train, $10\%$ Validation, $10\%$ Test).
  - **Apache Spark & Feature Store**: Extracted user interaction sequences (`user_history`), rating variances, log-scaled item popularities, and multi-hot genre representations.
  - **Space Optimization**: Interaction tables stored in Snappy-compressed Parquet, saving $>85\%$ storage space.

---

## Slide 4: Step 3 & 4: Two-Tower Candidate Retrieval & FAISS Index
- **Talking Points:**
  - **Two-Tower Neural Network (TensorFlow Recommenders)**: Separate User Query and Item Candidate towers mapping IDs into a shared 32-dimensional L2-normalized embedding space.
  - **FAISS ANN Index**: Indexed 23,238 item vectors using `IndexFlatIP` and `IndexIVFFlat`.
  - **Performance**: High recall candidate search running in **1.8 ms** with memory-mapped storage consuming only **3.2 MB** RAM.

---

## Slide 5: Step 5: Precision Deep Ranking (DCN-v2 vs. Wide & Deep)
- **Talking Points:**
  - Re-scores Stage 1 candidates using rich feature interactions (retrieval similarity, user activity, genre affinity, item popularity).
  - **Deep & Cross Network v2 (DCN-v2)**: Uses explicit Cross Layers to calculate bounded-degree feature interactions without manual combinatorial feature engineering.
  - **Wide & Deep Network**: Serves as our comparison benchmark (Variant B) balancing wide memorization with deep generalization.
  - **Sub-Millisecond Engine**: Pure vectorized NumPy forward pass executing candidate scoring in **0.8 ms**.

---

## Slide 6: Step 6: Cold-Start Handling & Resilient Multi-Tier Caching
- **Talking Points:**
  - **Cold-Start Strategy**: Unobserved users are gracefully handled via:
    1. *Bayesian Smoothed Global Popularity* (Wilson score) for immediate high-confidence picks.
    2. *Content-Based Genre Filtering* based on onboarding selections.
  - **Multi-Tier Resilient Caching**:
    - L1 In-Memory LRU Cache (< 5 MB RAM, **0.07 ms** latency).
    - L2 Redis Cache with 1-hour TTL.
    - Zero-downtime failover if Redis is unavailable.

---

## Slide 7: Step 7: Triton Serving & Airflow Retraining Orchestration
- **Talking Points:**
  - **NVIDIA Triton Inference Server**: Production model repository supporting ONNX Runtime, dynamic batching, and GPU/CPU deployment.
  - **Triton Client Bridge**: Automatic failover to local runtime if Triton is in standby.
  - **Apache Airflow DAG**: 8 automated tasks orchestrating data ingestion, temporal splitting, Spark feature extraction, Two-Tower retraining, FAISS re-indexing, DCN ranking, and Redis cache warmup.

---

## Slide 8: Step 7: Online A/B Experimentation & Evaluation
- **Talking Points:**
  - Deterministic 50/50 traffic split via MD5 hashing.
  - Offline evaluation against future test ground truth:
    - **NDCG@10**: Evaluates ranking quality and position discounting.
    - **Recall@10 & Hit Rate@10**: Evaluates catalog discovery.
    - **CTR (Click-Through Rate)**: Live conversion tracking.
    - **Statistical Significance**: Welch's two-sample t-test ($p$-value) verifying metric lift.

---

## Slide 9: Engineering Challenges Overcome
- **Memory & Resource Management**:
  - Maintained total RAM under **60 MB** on local hardware using memory-mapped FAISS (`faiss.IO_FLAG_MMAP`) and Parquet compression.
  - Converted $O(N)$ string scans into $O(1)$ constant-time hash map lookups, slashing request latency from 50 ms to **< 2 ms**.
  - Built resilient local fallbacks for Spark, Redis, and Triton so the system runs smoothly in both resource-constrained local dev and enterprise cloud clusters.

---

## Slide 10: Conclusion & Future Work
- **Key Takeaways**:
  - Built an end-to-end, PRD-compliant two-stage recommendation engine.
  - Verified across 7 automated unit tests in **0.1 seconds**.
  - Production-ready FastAPI endpoints and interactive Streamlit executive dashboard.
- **Future Enhancements**:
  - Real-time streaming updates via Kafka.
  - Multi-task learning (MTL) predicting both clicks and watch time (MMoE / PLE architecture).

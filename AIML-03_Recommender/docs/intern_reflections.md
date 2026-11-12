# 📝 Capstone Individual Intern Reflection

**Intern Name:** Himanshu Rathod  
**Project:** AIML-03 - Large-Scale Personalized Recommendation Engine  
**Track:** AI / Machine Learning  
**Internship Organization:** Nexvia Technologies  
**Date:** October 2026  

---

## 1. What I Owned
Throughout this Capstone project, I had end-to-end technical ownership over the architecture, implementation, optimization, and validation of the two-stage recommendation system:

1. **Data Pipeline & Temporal Splitting**:
   - Designed and implemented the data cleaning and temporal train/validation/test splitting pipeline, ensuring strict chronological boundaries to eliminate future data leakage.
   - Built a comprehensive MovieLens metadata catalog mapping movie IDs to human-readable titles, genres, and release years.

2. **Feature Engineering & Feature Store**:
   - Authored the Apache Spark feature engineering jobs and created a zero-JVM vectorized engine in Pandas/PyArrow to extract user watch history, log-scaled item popularity, and user-item genre affinity.

3. **Two-Tower Candidate Retrieval (TFRS)**:
   - Formulated and trained the Two-Tower retrieval model in TensorFlow Recommenders, learning 32-dimensional L2-normalized user and item embeddings with in-batch negative loss and FactorizedTopK metrics.

4. **FAISS Vector Indexing**:
   - Engineered the candidate retrieval engine using FAISS (`IndexFlatIP` and `IndexIVFFlat`), implementing index persistence and memory-mapped disk loading to preserve low RAM overhead.

5. **Deep Ranking Models (DCN-v2 & Wide & Deep)**:
   - Implemented explicit feature-crossing layers (DCN-v2) and Wide & Deep architectures to re-score Stage 1 candidates with sub-millisecond vectorized inference.

6. **Resilient Multi-Tier Caching & Cold-Start Strategy**:
   - Designed a dual-tier cache (L1 In-Memory LRU + L2 Redis) with automatic failover, and developed Bayesian smoothed popularity and content-based genre fallback algorithms for new users.

7. **Serving, Orchestration & A/B Testing**:
   - Structured the NVIDIA Triton model repository, built the Apache Airflow 8-task retraining DAG, and created the online A/B testing framework calculating NDCG@10, Recall@10, CTR, and Welch's t-test statistical significance.
   - Designed the modern interactive Streamlit dashboard and FastAPI endpoints.

---

## 2. Key Technical Learnings

### A. The Two-Stage Funnel Paradigm
Prior to this project, I viewed recommendation primarily as a single matrix factorization or deep learning task. Building this system taught me the industrial necessity of decoupling **Candidate Generation (Retrieval)** from **Scoring (Ranking)**:
- *Candidate Generation* prioritizes high recall and sub-2ms query throughput across the entire catalog.
- *Ranking* prioritizes high precision, evaluating a smaller candidate set using explicit cross-features and deep non-linear layers.

### B. The Critical Importance of Temporal Splitting
I learned why random train/test splits are deeply flawed in recommender systems: they allow models to predict past interactions using future knowledge. Implementing strict temporal boundaries demonstrated how to evaluate realistic generalization to future user intent.

### C. Explicit Feature Interactions (DCN-v2)
Standard MLPs require high depth to approximate feature crosses. Studying and coding DCN-v2 taught me how explicit mathematical cross-layers:
$$x_{l+1} = x_0 \odot (W_l x_l + b_l) + x_l$$
directly model feature interactions (e.g., user activity level $\times$ movie genre affinity) with minimal parameter overhead.

---

## 3. Engineering Challenges & Trade-Offs Overcome

### Challenge 1: Memory & Resource Management on Developer Hardware
- **Problem**: Large recommender systems often demand tens of gigabytes of RAM for embedding lookups and PySpark JVM runtimes, risking Out-Of-Memory (OOM) crashes on local developer machines.
- **Solution**:
  - Switched interaction logs from uncompressed CSV to Snappy-compressed Parquet, cutting storage by $85\%$.
  - Used memory-mapped loading (`faiss.IO_FLAG_MMAP`) for the FAISS index so vectors are read directly from disk without bloating resident RAM.
  - Replaced $O(N)$ string searches with $O(1)$ constant-time dictionary lookups in FastAPI, dropping request latency from 50 ms to **< 2 ms**.

### Challenge 2: Resilient Cold-Start and Cache Dependencies
- **Problem**: Real-world services cannot fail if Redis crashes or when a new user arrives with zero ratings.
- **Solution**:
  - Created a dual-tier cache where an in-memory LRU cache (< 5 MB RAM) provides instant fallback if the Redis daemon is unreachable.
  - Implemented Bayesian smoothed Wilson-score popularity and onboarding genre filters, ensuring new users immediately receive relevant, high-quality recommendations.

---

## 4. Personal Takeaways & Future Goals
Building this system end-to-end solidified my interest in large-scale machine learning systems engineering (MLOps). Moving forward, I plan to explore:
- Streaming real-time feature extraction with Apache Kafka / Flink.
- Multi-Task Learning (MTL) with MMoE (Multi-gate Mixture-of-Experts) to jointly optimize for Click-Through Rate and Watch Duration.

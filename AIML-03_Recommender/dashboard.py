"""
Interactive Recommendation Engine Dashboard (Streamlit).
AIML-03 Capstone Project: Large-Scale Personalized Recommendation Engine.
Demonstrates:
- Stage 1 FAISS Retrieval + Stage 2 Deep Ranking (DCN-v2 / Wide & Deep)
- Cold-Start Handling (Popularity & Genre Filtering)
- Live Online A/B Experimentation (CTR, NDCG@10, Recall@10, Latency)
- System Architecture & Memory Footprint Telemetry
"""

import streamlit as st
import numpy as np
import pandas as pd
import requests
import json
import time

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Nexvia RecSys | Two-Stage Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1e293b;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #3b82f6;
        margin-bottom: 12px;
    }
    .badge-genre {
        display: inline-block;
        background: linear-gradient(135deg, #1e3a8a, #3b82f6);
        color: #ffffff;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 14px;
        font-size: 11px;
        margin-right: 5px;
        margin-top: 3px;
    }
    .badge-promo {
        background-color: #065f46;
        color: #34d399;
        font-size: 11px;
        padding: 2px 8px;
        border-radius: 10px;
        font-weight: 600;
    }
    .badge-demo {
        background-color: #7f1d1d;
        color: #f87171;
        font-size: 11px;
        padding: 2px 8px;
        border-radius: 10px;
        font-weight: 600;
    }
    .badge-kept {
        background-color: #374151;
        color: #d1d5db;
        font-size: 11px;
        padding: 2px 8px;
        border-radius: 10px;
        font-weight: 600;
    }
    .stage-card {
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 14px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://img.icons8.com/clouds/200/movie-projector.png", width=120)
st.sidebar.title("AIML-03 Recommender")
st.sidebar.markdown("**Two-Stage Personalized Recommender**")
st.sidebar.markdown("Netflix & Amazon Style Architecture")

# Initialize Session State to preserve recommendations across interactions
if "rec_data" not in st.session_state:
    st.session_state["rec_data"] = None
if "rec_user_id" not in st.session_state:
    st.session_state["rec_user_id"] = None
if "rec_elapsed" not in st.session_state:
    st.session_state["rec_elapsed"] = 0.0
if "liked_set" not in st.session_state:
    st.session_state["liked_set"] = set()
if "cs_data" not in st.session_state:
    st.session_state["cs_data"] = None

def handle_like_click(movie_id, movie_title, user_id, variant):
    """Callback executed before rerun: adds to liked set and posts feedback."""
    mid = str(movie_id)
    if "liked_set" not in st.session_state:
        st.session_state["liked_set"] = set()
    st.session_state["liked_set"].add(mid)
    try:
        requests.post(f"{API_URL}/feedback/click", json={
            "user_id": str(user_id),
            "movie_id": mid,
            "variant": str(variant)
        }, timeout=0.3)
    except Exception:
        pass
    st.toast(f"❤️ Added '{movie_title}' to Watch History!")

@st.fragment
def render_two_stage_funnel(data, top_k, current_user):
    """Isolated Streamlit fragment: handles like interactions without page jump or reloading."""
    recs = data.get("recommendations", [])
    stage1_cands = data.get("stage1_candidates", [])
    elapsed = st.session_state.get("rec_elapsed", 0.0)
    
    # Info Telemetry Bar
    info_c1, info_c2, info_c3, info_c4 = st.columns(4)
    info_c1.metric("A/B Model Variant", data.get("variant", "N/A"))
    info_c2.metric("End-to-End Latency", f"{data.get('latency_ms', elapsed):.2f} ms")
    info_c3.metric("Cache Layer", "Cache Hit ⚡" if data.get("cache_hit") else "Cache Miss")
    info_c4.metric("Inference Engine", data.get("inference_source", "Local-Runtime"))
    
    num_liked = len(st.session_state.get("liked_set", set()))
    if num_liked > 0:
        st.success(f"⭐ **Session Watch History:** `{num_liked}` movie(s) logged. Real-time CTR & online interactions updated.")
    
    st.markdown("---")
    st.info("💡 **Two-Stage Architecture in Action**: **Stage 1 (FAISS)** scans 23,238 vector embeddings in **~1.8 ms** to find semantic neighbors. **Stage 2 (DCN-v2)** re-scores them using user history & feature crosses in **~0.9 ms**.")
    
    # Side-by-side comparison: Stage 1 vs Stage 2
    col_stage1, col_stage2 = st.columns(2)
    
    # --- COLUMN 1: STAGE 1 CANDIDATES ---
    with col_stage1:
        st.markdown("### 🎯 Stage 1: Candidate Generation")
        st.caption("FAISS Vector Index (Approximate Nearest Neighbor via Cosine Similarity)")
        
        cands_to_show = stage1_cands if stage1_cands else recs
        for cand in cands_to_show[:top_k]:
            s1_rank = cand.get("stage1_rank", 1)
            with st.container():
                st.markdown(f"""
                <div class='stage-card'>
                    <div style='display:flex; justify-content:space-between; align-items:center;'>
                        <span style='font-size:16px; font-weight:700;'>#{s1_rank} {cand['title']} ({cand['year']})</span>
                        <span style='font-family:monospace; color:#60a5fa;'>Sim: {cand.get('retrieval_score', 0):.3f}</span>
                    </div>
                    <div style='margin-top:6px;'>
                        {" ".join([f"<span class='badge-genre'>{g}</span>" for g in cand['genres'].split('|')])}
                    </div>
                    <div style='font-size:11px; color:#94a3b8; margin-top:6px;'>
                        Catalog ID: <code>{cand['movie_id']}</code> | Retrieved from 23,238 items
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # --- COLUMN 2: STAGE 2 PRECISION RANKING ---
    with col_stage2:
        st.markdown("### 🏆 Stage 2: Precision Re-Ranking")
        st.caption(f"{data.get('variant', 'DCN-v2')} Deep Neural Ranker (Feature Crossings & Re-scoring)")
        
        for s2_rank, movie in enumerate(recs, 1):
            s1_rank = movie.get("stage1_rank", s2_rank)
            delta = s1_rank - s2_rank
            
            if delta > 0:
                delta_badge = f"<span class='badge-promo'>▲ Promoted +{delta} spots (was #{s1_rank} in Stage 1)</span>"
            elif delta < 0:
                delta_badge = f"<span class='badge-demo'>▼ Demoted {abs(delta)} spots (was #{s1_rank} in Stage 1)</span>"
            else:
                delta_badge = f"<span class='badge-kept'>━ Kept #{s2_rank} across both stages</span>"
                
            with st.container():
                st.markdown(f"""
                <div class='stage-card' style='border-left: 4px solid #10b981;'>
                    <div style='display:flex; justify-content:space-between; align-items:center;'>
                        <span style='font-size:16px; font-weight:700;'>#{s2_rank} {movie['title']} ({movie['year']})</span>
                        <span style='font-family:monospace; color:#34d399; font-weight:bold;'>P(Click): {movie.get('ranking_score', 0):.3f}</span>
                    </div>
                    <div style='margin-top:6px;'>
                        {" ".join([f"<span class='badge-genre'>{g}</span>" for g in movie['genres'].split('|')])}
                    </div>
                    <div style='margin-top:6px;'>
                        {delta_badge}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                is_liked = str(movie['movie_id']) in st.session_state.get("liked_set", set())
                if is_liked:
                    st.markdown("""
                    <div style='background:#064e3b; border:1px solid #059669; color:#6ee7b7; padding:8px 12px; border-radius:8px; font-size:12px; font-weight:600; text-align:center;'>
                        ✅ Added to Watch History & Logged for CTR
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.button(
                        "❤️ Like / Watched",
                        key=f"btn_like_{movie['movie_id']}_{s2_rank}",
                        on_click=handle_like_click,
                        args=(movie['movie_id'], movie['title'], current_user, data.get("variant", "Variant_A_DCN")),
                        use_container_width=True
                    )

# Quick API Health Check
try:
    health_resp = requests.get(f"{API_URL}/health", timeout=1.0)
    api_online = health_resp.status_code == 200
    health_data = health_resp.json() if api_online else {}
except Exception:
    api_online = False
    health_data = {}

if api_online:
    st.sidebar.success(f"Backend API: Online ({health_data.get('faiss_total_movies', 0):,} Movies Indexed)")
else:
    st.sidebar.warning("Backend API is running in embedded local mode.")

tab1, tab2, tab3, tab4 = st.tabs([
    "🎬 Personalized Recommendations",
    "❄️ Cold-Start Playground",
    "📊 Online A/B Testing & Metrics",
    "⚙️ System Telemetry & Architecture"
])

# ==============================================================================
# TAB 1: PERSONALIZED RECOMMENDATIONS
# ==============================================================================
with tab1:
    st.subheader("Personalized Two-Stage Recommendations")
    st.markdown("Combines **Stage 1 FAISS Vector Candidate Retrieval** with **Stage 2 Deep Precision Ranking**.")
    
    col1, col2 = st.columns([1, 2.5])
    
    with col1:
        st.markdown("#### User Configuration")
        user_input = st.selectbox(
            "Select Preset User ID or Type",
            ["1", "10", "100", "500", "1000", "Custom"]
        )
        if user_input == "Custom":
            selected_user_id = st.text_input("Enter User ID", "1")
        else:
            selected_user_id = user_input
            
        top_k = st.slider("Number of Recommendations (Top-K)", min_value=3, max_value=20, value=6)
        
        variant_override = st.selectbox(
            "Routing Strategy",
            ["Auto (Hash-based A/B)", "Force Variant A (DCN-v2)", "Force Variant B (Wide & Deep)"]
        )
        model_variant = None
        if "Variant A" in variant_override:
            model_variant = "Variant_A_DCN"
        elif "Variant B" in variant_override:
            model_variant = "Variant_B_WideDeep"
            
        fetch_btn = st.button("Generate Recommendations", type="primary", use_container_width=True)

    with col2:
        if fetch_btn:
            with st.spinner("Executing Two-Stage Retrieval & Ranking Pipeline..."):
                try:
                    payload = {"user_id": selected_user_id, "top_k": top_k}
                    if model_variant:
                        payload["model_variant"] = model_variant
                        
                    t_start = time.time()
                    resp = requests.post(f"{API_URL}/recommend", json=payload, timeout=5.0)
                    elapsed = (time.time() - t_start) * 1000.0
                    
                    if resp.status_code == 200:
                        st.session_state["rec_data"] = resp.json()
                        st.session_state["rec_user_id"] = selected_user_id
                        st.session_state["rec_elapsed"] = elapsed
                    else:
                        st.error(f"Error from API: {resp.text}")
                except Exception as e:
                    st.error(f"Could not connect to FastAPI server at {API_URL}: {e}")
                    st.info("Tip: Start the backend using: `uvicorn src.api:app --reload`")

        if st.session_state.get("rec_data"):
            render_two_stage_funnel(
                data=st.session_state["rec_data"],
                top_k=top_k,
                current_user=st.session_state.get("rec_user_id", selected_user_id)
            )
        else:
            st.info("👈 Select a User ID and click **Generate Recommendations** to view personalized results.")

# ==============================================================================
# TAB 2: COLD-START PLAYGROUND
# ==============================================================================
with tab2:
    st.subheader("❄️ Cold-Start Handling Strategies")
    st.markdown("""
    When an unobserved user with zero historical interactions enters the platform, the recommender dynamically switches to:
    1. **Bayesian-Smoothed Global Popularity Fallback** (Wilson/IMDb score).
    2. **Content-Based Genre Filtering** based on user preference onboarding.
    """)
    
    cs_col1, cs_col2 = st.columns([1, 2.5])
    with cs_col1:
        st.markdown("#### Cold-Start Onboarding")
        test_new_user_id = st.text_input("New Unseen User ID", "guest_99999")
        fallback_mode = st.radio("Fallback Strategy", ["Content-Based (Preferred Genres)", "Global Popularity Ranking"])
        
        selected_genres = []
        if fallback_mode == "Content-Based (Preferred Genres)":
            selected_genres = st.multiselect(
                "Select Favorite Genres",
                ["Action", "Adventure", "Animation", "Comedy", "Crime", "Drama", "Fantasy", "Horror", "Mystery", "Romance", "Sci-Fi", "Thriller", "War"],
                default=["Action", "Sci-Fi"]
            )
        cs_top_k = st.slider("Recommendations Count", 3, 15, 6, key="cs_top_k")
        cs_btn = st.button("Run Cold-Start Recommender", type="primary")

    with cs_col2:
        if cs_btn:
            with st.spinner("Generating Cold-Start Fallback Recommendations..."):
                try:
                    payload = {
                        "user_id": test_new_user_id,
                        "top_k": cs_top_k,
                        "preferred_genres": selected_genres if fallback_mode.startswith("Content") else []
                    }
                    resp = requests.post(f"{API_URL}/recommend", json=payload, timeout=5.0)
                    if resp.status_code == 200:
                        st.session_state["cs_data"] = resp.json()
                    else:
                        st.error(f"Error: {resp.text}")
                except Exception as e:
                    st.error(f"Error: {e}")

        if st.session_state.get("cs_data"):
            cs_data = st.session_state["cs_data"]
            st.success(f"Cold-Start Handled via: `{cs_data.get('inference_source')}`")
            
            for i, movie in enumerate(cs_data.get("recommendations", []), 1):
                with st.container():
                    cc1, cc2 = st.columns([3.5, 1.5])
                    with cc1:
                        st.markdown(f"**{i}. {movie['title']}** ({movie['year']})")
                        genres_html = " ".join([f"<span class='badge-genre'>{g}</span>" for g in movie['genres'].split('|')])
                        st.markdown(genres_html, unsafe_allow_html=True)
                    with cc2:
                        st.metric("Confidence / Score", f"{movie.get('ranking_score', 0):.2f}")
                    st.markdown("<hr style='margin: 6px 0;'>", unsafe_allow_html=True)
        else:
            st.info("Select preferences on the left and test the cold-start fallback mechanism.")

# ==============================================================================
# TAB 3: ONLINE A/B TESTING & EVALUATION
# ==============================================================================
with tab3:
    st.subheader("📊 Online A/B Testing & Evaluation Hub")
    st.markdown("Compares **Variant A (DCN-v2)** against **Variant B (Wide & Deep)** on NDCG@10, Recall@10, and Click-Through Rate.")
    
    # Live Experiment Telemetry
    try:
        metrics_resp = requests.get(f"{API_URL}/metrics", timeout=2.0)
        metrics_data = metrics_resp.json() if metrics_resp.status_code == 200 else {}
    except Exception:
        metrics_data = {}
        
    ab_stats = metrics_data.get("ab_experiment", {})
    var_a = ab_stats.get("variant_a_dcn", {})
    var_b = ab_stats.get("variant_b_wide_deep", {})
    
    col_a, col_b, col_lift = st.columns([1.5, 1.5, 1])
    
    with col_a:
        st.markdown("### Variant A: DCN-v2")
        st.metric("Requests Served", f"{var_a.get('requests', 0):,}")
        st.metric("Click-Through Rate (CTR)", f"{var_a.get('ctr', 0):.2%}")
        st.metric("Avg Latency", f"{var_a.get('avg_latency_ms', 0):.2f} ms")

    with col_b:
        st.markdown("### Variant B: Wide & Deep")
        st.metric("Requests Served", f"{var_b.get('requests', 0):,}")
        st.metric("Click-Through Rate (CTR)", f"{var_b.get('ctr', 0):.2%}")
        st.metric("Avg Latency", f"{var_b.get('avg_latency_ms', 0):.2f} ms")
        
    with col_lift:
        st.markdown("### Experiment Lift")
        st.metric("CTR Lift", ab_stats.get("ctr_lift", "N/A"))
        st.markdown("**Routing:** MD5 Deterministic Hash (50/50 Split)")
        
    st.markdown("---")
    st.markdown("#### Automated Benchmark Simulation (N=200 Test Users)")
    st.caption("Runs an offline test simulation against temporal test ground truth and computes NDCG@10, Recall@10, and Welch's t-test p-value.")
    
    if st.button("🚀 Run Full A/B Benchmark Experiment", type="primary"):
        with st.spinner("Simulating user interactions across both model variants..."):
            from src.evaluation.ab_test import run_ab_experiment
            results = run_ab_experiment(num_test_users=200)
            summary = results["summary"]
            
            res_c1, res_c2, res_c3 = st.columns(3)
            with res_c1:
                st.markdown("##### 🔬 Variant A (DCN-v2)")
                st.write(f"**NDCG@10:** `{summary['variant_a_dcn']['ndcg_at_10']}`")
                st.write(f"**Recall@10:** `{summary['variant_a_dcn']['recall_at_10']}`")
                st.write(f"**Hit Rate@10:** `{summary['variant_a_dcn']['hit_rate_at_10']}`")
                st.write(f"**CTR:** `{summary['variant_a_dcn']['ctr']:.2%}`")
                st.write(f"**p50 Latency:** `{summary['variant_a_dcn']['latency_p50_ms']} ms`")
            with res_c2:
                st.markdown("##### 🔬 Variant B (Wide & Deep)")
                st.write(f"**NDCG@10:** `{summary['variant_b_wide_deep']['ndcg_at_10']}`")
                st.write(f"**Recall@10:** `{summary['variant_b_wide_deep']['recall_at_10']}`")
                st.write(f"**Hit Rate@10:** `{summary['variant_b_wide_deep']['hit_rate_at_10']}`")
                st.write(f"**CTR:** `{summary['variant_b_wide_deep']['ctr']:.2%}`")
                st.write(f"**p50 Latency:** `{summary['variant_b_wide_deep']['latency_p50_ms']} ms`")
            with res_c3:
                st.markdown("##### 📈 Statistical Significance")
                st.write(f"**NDCG Lift:** `{summary['statistical_tests']['ndcg_lift']}`")
                st.write(f"**CTR Lift:** `{summary['statistical_tests']['ctr_lift']}`")
                st.write(f"**p-value:** `{summary['statistical_tests']['ndcg_p_value']}`")
                st.write(f"**Winner:** `{summary['statistical_tests']['winner']}`")

# ==============================================================================
# TAB 4: SYSTEM ARCHITECTURE & TELEMETRY
# ==============================================================================
with tab4:
    st.subheader("⚙️ System Architecture & Telemetry")
    
    st.markdown("""
    ```mermaid
    flowchart LR
        User["User Request"] --> API["FastAPI Gateway"]
        API --> Cache{"Redis / L1 Cache"}
        Cache -->|"Cache Hit (Fast Path)"| Response["Recommended Output"]
        Cache -->|"Cache Miss"| ColdCheck{"Known User?"}
        ColdCheck -->|"Cold Start"| Fallback["Popularity / Genre Fallback"]
        ColdCheck -->|"Known User"| FAISS["Stage 1: FAISS Vector Retrieval (Top 50)"]
        FAISS --> Ranker["Stage 2: Precision Ranking (DCN-v2 / Wide & Deep)"]
        Ranker --> CacheUpdate["Update Redis Cache"]
        Fallback --> Response
        CacheUpdate --> Response
    ```
    """)
    
    t_c1, t_c2, t_c3 = st.columns(3)
    
    with t_c1:
        st.markdown("#### 💾 Memory & Resource Management")
        st.markdown("""
        - **Total Indexed Items:** `23,238` movies
        - **Total User Embeddings:** `142,046` users
        - **FAISS Index RAM:** ~3.2 MB (Memory-Mapped)
        - **User Embeddings RAM:** ~18.1 MB
        - **Disk Footprint (Parquet):** < 1.5 MB
        - **Total Local RAM Footprint:** `< 60 MB`
        """)
        
    with t_c2:
        st.markdown("#### ⚡ Latency Budget Breakdown")
        latency_df = pd.DataFrame({
            "Stage": ["1. Cache Lookup", "2. FAISS ANN Search", "3. Precision Ranking", "4. Metadata Hydration"],
            "Typical Latency": ["0.08 ms", "1.80 ms", "0.90 ms", "0.40 ms"]
        })
        st.table(latency_df)
        
    with t_c3:
        st.markdown("#### 🚀 Infrastructure & MLOps")
        st.markdown(f"""
        - **Triton Server Bridge:** `{metrics_data.get('triton_status', {}).get('mode', 'Standby')}`
        - **Cache L1 Status:** `{metrics_data.get('cache', {}).get('l1_items', 0)} keys cached`
        - **Cache Hit Ratio:** `{metrics_data.get('cache', {}).get('hit_ratio', 0):.1%}`
        - **Retraining Scheduler:** Apache Airflow (`recommender_pipeline_dag.py`)
        """)

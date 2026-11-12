import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Fraud Detection Analyst Platform", page_icon="🚨", layout="wide")

st.title("🚨 Real-Time Financial Fraud Detection & Analyst Platform")
st.markdown("Blended ML (XGBoost) + Graph Neural Network (GNN) scoring with human-in-the-loop analyst feedback.")

# Real fraud case extracted from creditcard.csv (Class = 1)
REAL_FRAUD_PAYLOAD = {
    "Time": 406.0, "Amount": 1850.0,
    "V1": -2.3122, "V2": 1.9520, "V3": -1.6099, "V4": 3.9979, "V5": -0.5222,
    "V6": -1.4265, "V7": -2.5374, "V8": 1.3917, "V9": -2.7701, "V10": -2.7723,
    "V11": 3.2020, "V12": -2.8999, "V13": -0.5952, "V14": -4.2893, "V15": 0.3897,
    "V16": -1.1407, "V17": -2.8301, "V18": -0.0168, "V19": 0.4170, "V20": 0.1269,
    "V21": 0.5172, "V22": -0.0350, "V23": -0.4652, "V24": 0.3202, "V25": 0.0445,
    "V26": 0.1778, "V27": 0.2611, "V28": -0.1433
}

# Initialize session state for analyst feedback case queue
if "cases" not in st.session_state:
    st.session_state.cases = [
        {"Case ID": "CASE-101", "Type": "High Velocity", "Amount": 3450.00, "Risk Score": 0.92, "Status": "Under Review", "Feedback": "Pending"},
        {"Case ID": "CASE-102", "Type": "Card-Not-Present", "Amount": 120.50, "Risk Score": 0.08, "Status": "Resolved", "Feedback": "Legitimate"},
        {"Case ID": "CASE-103", "Type": "Fraud Ring", "Amount": 8900.00, "Risk Score": 0.88, "Status": "Escalated", "Feedback": "Confirmed Fraud"},
    ]

if "payload_to_send" not in st.session_state:
    st.session_state.payload_to_send = {"Time": 0.0, "Amount": 25.0}

# Layout: 2 Columns (Live Scoring & Analyst Review Queue)
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("💳 Transaction Live Scoring")
    st.markdown("**Quick Demo Scenarios (Click to test):**")
    
    preset_col1, preset_col2 = st.columns(2)
    with preset_col1:
        if st.button("🟢 Safe Grocery ($25.00)", use_container_width=True):
            st.session_state.payload_to_send = {"Time": 100.0, "Amount": 25.0}
    with preset_col2:
        if st.button("🚨 Stolen Card Attack ($1,850)", use_container_width=True):
            st.session_state.payload_to_send = REAL_FRAUD_PAYLOAD

    amount_val = st.number_input("Transaction Amount ($)", min_value=0.0, value=float(st.session_state.payload_to_send.get("Amount", 25.0)), step=10.0)
    st.session_state.payload_to_send["Amount"] = amount_val

    if st.button("⚡ Score Transaction in Real-Time", type="primary", use_container_width=True):
        with st.spinner("Calling FastAPI Kubernetes backend..."):
            try:
                response = requests.post("http://127.0.0.1:8000/predict", json=st.session_state.payload_to_send, timeout=5)
                data = response.json()
                
                c1, c2, c3 = st.columns(3)
                c1.metric("XGBoost Score", f"{data['xgb_score']:.4f}")
                c2.metric("GNN Score", f"{data['gnn_score']:.4f}")
                c3.metric("Blended Score", f"{data['blended_score']:.4f}")
                
                st.divider()
                if data["is_fraud"]:
                    st.error(f"🚨 **FRAUD RING DETECTED!** High Risk Alert ({data['blended_score'] * 100:.1f}%)")
                    new_case = {
                        "Case ID": f"CASE-{len(st.session_state.cases)+101}",
                        "Type": "Stolen Card Attack",
                        "Amount": amount_val,
                        "Risk Score": data["blended_score"],
                        "Status": "Under Review",
                        "Feedback": "Pending"
                    }
                    st.session_state.cases.append(new_case)
                    st.warning(f"⚠️ Action Required: Transaction blocked and added to Analyst Review Queue as **{new_case['Case ID']}**.")
                else:
                    st.success(f"✅ **Transaction Approved.** Safe purchase ({data['blended_score'] * 100:.1f}% risk).")
            except Exception as e:
                # Friendly fallback demo if docker port forward is not active
                st.warning(f"Connection notice: {e}")
                st.info("Tip: Start the port-forward or FastAPI service at http://127.0.0.1:8000")

with col_right:
    st.subheader("🕵️ Analyst Case Management & Feedback")
    df_cases = pd.DataFrame(st.session_state.cases)
    st.dataframe(df_cases, use_container_width=True)

    st.markdown("### Human-in-the-Loop Analyst Feedback")
    pending_cases = [c["Case ID"] for c in st.session_state.cases if c["Feedback"] == "Pending"]
    if pending_cases:
        selected_case = st.selectbox("Select Case to Investigate", pending_cases)
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("🚨 Confirm Fraud (True Positive)", use_container_width=True):
                for c in st.session_state.cases:
                    if c["Case ID"] == selected_case:
                        c["Status"] = "Confirmed Fraud"
                        c["Feedback"] = "True Positive"
                st.success(f"Feedback logged: {selected_case} verified as True Positive.")
                st.rerun()
        with btn_col2:
            if st.button("✅ Mark False Positive (Legitimate)", use_container_width=True):
                for c in st.session_state.cases:
                    if c["Case ID"] == selected_case:
                        c["Status"] = "Resolved"
                        c["Feedback"] = "False Positive"
                st.success(f"Feedback logged: {selected_case} marked as False Positive.")
                st.rerun()
    else:
        st.info("✅ All open cases have analyst feedback submitted!")

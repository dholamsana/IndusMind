from rag_engine import ask_maintenance_copilot
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="Industrial Maintenance Copilot",
    page_icon="⚙️",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

# --------------------------------
# LOAD MODELS AND DATA
# --------------------------------

@st.cache_resource
def load_models():
    rul_model = joblib.load(BASE_DIR / "models/rul_model.pkl")
    anomaly_model = joblib.load(BASE_DIR / "models/anomaly_model.pkl")
    feature_columns = joblib.load(BASE_DIR / "models/feature_columns.pkl")
    anomaly_features = joblib.load(BASE_DIR / "models/anomaly_features.pkl")

    return rul_model, anomaly_model, feature_columns, anomaly_features


@st.cache_data
def load_data():
    return pd.read_csv(BASE_DIR / "data/processed/engine_data.csv")


try:
    rul_model, anomaly_model, feature_columns, anomaly_features = load_models()
    df = load_data()

except Exception as e:
    st.error(f"Could not load project files: {e}")
    st.stop()


# --------------------------------
# HEADER
# --------------------------------

st.title("⚙️ Industrial Predictive Maintenance")
st.subheader("AI-powered Engine Health Monitoring System")

st.markdown(
    """
    Monitor engine sensor readings, estimate remaining useful life,
    and identify unusual sensor behaviour using machine learning.
    """
)

st.divider()

# --------------------------------
# SIDEBAR
# --------------------------------

st.sidebar.header("Engine Selection")

engine_ids = sorted(df["engine_id"].unique())

selected_engine = st.sidebar.selectbox(
    "Select Engine",
    engine_ids
)

engine_data = df[
    df["engine_id"] == selected_engine
].sort_values("cycle")

min_cycle = int(engine_data["cycle"].min())
max_cycle = int(engine_data["cycle"].max())

selected_cycle = st.sidebar.slider(
    "Select Operating Cycle",
    min_value=min_cycle,
    max_value=max_cycle,
    value=max_cycle
)

selected_row = engine_data[
    engine_data["cycle"] == selected_cycle
].iloc[[0]]

# --------------------------------
# RUL PREDICTION
# --------------------------------

X_input = selected_row[feature_columns]

predicted_rul = float(
    rul_model.predict(X_input)[0]
)

predicted_rul = float(
    np.clip(predicted_rul, 0, 125)
)

# --------------------------------
# ANOMALY DETECTION
# --------------------------------

sensor_input = selected_row[anomaly_features]

anomaly_result = anomaly_model.predict(sensor_input)[0]

anomaly_status = (
    "Anomaly Detected"
    if anomaly_result == -1
    else "Normal"
)

# --------------------------------
# METRICS
# --------------------------------

st.header(f"Engine {selected_engine} Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Operating Cycle",
        selected_cycle
    )

with col2:
    st.metric(
        "Estimated Remaining Useful Life",
        f"{predicted_rul:.1f} cycles"
    )

with col3:
    if anomaly_result == -1:
        st.error("⚠️ Anomaly Detected")
    else:
        st.success("Normal Sensor Behaviour")

st.caption(
    "RUL is capped at 125 cycles. Anomaly detection identifies unusual "
    "sensor patterns and does not confirm a mechanical failure."
)

st.divider()

# --------------------------------
# SENSOR ANALYSIS
# --------------------------------

st.header("📊 Sensor Analysis")

important_sensors = [
    sensor for sensor in
    ["sensor_11", "sensor_4", "sensor_12", "sensor_9", "sensor_7"]
    if sensor in engine_data.columns
]

st.markdown("### Sensor readings over engine lifecycle")

if important_sensors:
    sensor_history = engine_data.set_index("cycle")[important_sensors]
    st.line_chart(sensor_history)

st.markdown("### Selected cycle sensor readings")

st.dataframe(
    selected_row[anomaly_features].T.rename(
        columns={selected_row.index[0]: "Reading"}
    ),
    use_container_width=True
)

# --------------------------------
# FEATURE IMPORTANCE
# --------------------------------

st.divider()
st.header("🔍 Model Insights")

importance_df = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": rul_model.feature_importances_
}).sort_values("Importance", ascending=False)

st.markdown("### RUL Model Feature Importance")

st.bar_chart(
    importance_df.set_index("Feature")["Importance"].head(10)
)

# --------------------------------
# FOOTER
# --------------------------------

st.divider()

st.markdown(
    """
    **Project:** Industrial Predictive Maintenance & AI Maintenance Copilot

    **Dataset:** NASA C-MAPSS FD001 Turbofan Engine Degradation Simulation

    **Machine Learning:** Gradient Boosting Regressor, Isolation Forest

    **Framework:** Python, Scikit-learn, Pandas, Streamlit
    """
)


st.divider()

st.header("🤖 IndusMind AI Maintenance Copilot")

st.markdown(
    "Ask questions about engine health, sensor anomalies, "
    "degradation patterns and maintenance actions."
)

# Build context using actual ML model outputs
sensor_readings = selected_row[anomaly_features].iloc[0].to_dict()

engine_context = f"""
Engine ID: {selected_engine}
Operating Cycle: {selected_cycle}
Predicted Remaining Useful Life: {predicted_rul:.1f} cycles
Anomaly Detection Result: {anomaly_status}

Selected Sensor Readings:
{sensor_readings}
"""

# Store conversation history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display previous messages
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_question = st.chat_input(
    "Ask IndusMind about this engine..."
)

if user_question:
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_question
    })

    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Analysing engine data and retrieving knowledge..."):
            try:
                answer = ask_maintenance_copilot(
                    user_question,
                    engine_context
                )

                st.markdown(answer)

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as e:
                st.error(f"AI assistant error: {e}")

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the model
model = joblib.load("RealEstates.joblib")

# Set Streamlit config
st.set_page_config(page_title="Real Estate Dashboard", page_icon="🏠", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
        body { background-color: #f7f9fc; }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            font-size: 1rem;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .main > div {
            padding-top: 1rem;
            padding-bottom: 1rem;
            padding-left: 3rem;
            padding-right: 3rem;
        }
    </style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("""
    <h1 style='text-align: center; color: #2c3e50;'>🏠 Real Estate Price Dashboard</h1>
    <p style='text-align: center; color: #555;'>Predict Boston housing prices using machine learning</p>
""", unsafe_allow_html=True)

st.markdown("---")

# --- INPUT SECTION ---
st.subheader("🧾 Enter Housing Features")
input_left, input_right = st.columns(2)

with input_left:
    CRIM = st.number_input("CRIM", help="Per capita crime rate", value=0.1)
    ZN = st.number_input("ZN", help="Residential land zoned", value=12.5)
    INDUS = st.number_input("INDUS", help="Non-retail business acres", value=7.5)
    CHAS = st.selectbox("CHAS", [0, 1], help="Bounds Charles River")
    NOX = st.number_input("NOX", help="Nitric oxide concentration", min_value=0.0, max_value=1.0, value=0.5)
    RM = st.number_input("RM", help="Avg number of rooms", value=6.0)

with input_right:
    AGE = st.number_input("AGE", help="Proportion of old units", value=50.0)
    DIS = st.number_input("DIS", help="Distance to employment centers", value=4.0)
    RAD = st.number_input("RAD", help="Access to radial highways", value=4)
    TAX = st.number_input("TAX", help="Property-tax rate", value=300.0)
    PTRATIO = st.number_input("PTRATIO", help="Pupil-teacher ratio", value=18.0)
    B = st.number_input("B", help="1000(Bk - 0.63)^2", value=396.9)
    LSTAT = st.number_input("LSTAT", help="% lower status population", value=12.5)

# --- Prediction ---
input_data = pd.DataFrame(
    [[CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT]],
    columns=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
             'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
)

st.markdown("---")
st.subheader("🎯 Predicted House Price")

if st.button("🚀 Predict House Price"):
    prediction = model.predict(input_data)
    st.markdown(f"""
        <div style="background-color: #e8f5e9; padding: 20px; border-radius: 10px;">
            <h3 style="color: #2e7d32;">💲 Estimated Price: ${prediction[0]*1000:.2f}</h3>
        </div>
    """, unsafe_allow_html=True)

# --- VISUAL INSIGHTS SECTION ---
st.markdown("---")
st.subheader("📊 Dashboard Insights")
chart1, chart2 = st.columns(2)

with chart1:
    st.markdown("### 📌 Feature Importance")
    feature_names = input_data.columns.tolist()
    importances = model.feature_importances_
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.barh(feature_names, importances, color='#3498db')
    ax1.set_title("Feature Importance", fontsize=12)
    ax1.set_xlabel("Importance", fontsize=10)
    st.pyplot(fig1)

with chart2:
    st.markdown("### 🏘 Room Count Distribution")
    sample_rms = np.random.normal(loc=6.2, scale=0.7, size=100)
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.histplot(sample_rms, kde=True, ax=ax2, color='#f39c12')
    ax2.set_title("Distribution of Average Rooms", fontsize=12)
    ax2.set_xlabel("Number of Rooms", fontsize=10)
    st.pyplot(fig2)

# --- FOOTER ---
st.markdown("---")
st.markdown("<p style='text-align:center; color:gray;'>Made with ❤️ using Streamlit | © 2025</p>", unsafe_allow_html=True)

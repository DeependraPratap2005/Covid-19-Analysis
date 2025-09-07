import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os

# Seaborn theme
sns.set_theme(style="darkgrid")

# Custom CSS for UI
st.markdown(
    """
    <style>
    /* Main App Background */
    .stApp {
        background: linear-gradient(135deg, #141E30, #243B55);
        color: white;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Sidebar Styling - Glassmorphism */
    section[data-testid="stSidebar"] {
        background: rgba(15, 25, 35, 0.85);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 15px;
        padding: 10px;
    }

    /* Sidebar radio buttons */
    div[data-baseweb="radio"] label {
        color: #f5f5f5 !important;
        font-weight: 500;
        border-radius: 10px;
        padding: 8px 15px;
        transition: 0.3s;
    }
    div[data-baseweb="radio"] label:hover {
        background: linear-gradient(90deg, #00c6ff, #0072ff);
        color: white !important;
        cursor: pointer;
        box-shadow: 0px 0px 10px rgba(0, 183, 255, 0.7);
    }

    /* Titles */
    h1, h2, h3, h4 {
        color: #ffffff;
        text-shadow: 1px 1px 4px rgba(0,0,0,0.5);
    }

    /* Metric Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        color: white;
        backdrop-filter: blur(10px);
        margin: 10px;
        transition: 0.3s;
    }
    .metric-card:hover {
        transform: scale(1.05);
        box-shadow: 0 0 20px rgba(255,255,255,0.3);
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
    }
    .metric-label {
        font-size: 16px;
        opacity: 0.9;
    }

    /* Developer Credit */
    .dev-credit {
        font-size: 18px;
        font-weight: bold;
        color: #8B4513; /* Brown color */
        text-align: center;
        text-shadow: 2px 2px 5px #000000, 3px 3px 8px #5C3317;
        margin-top: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.title("🦠 COVID-19 Data Analysis and Prediction")
st.markdown("Analyze COVID-19 data globally with **interactive visualizations** and **predictive modeling**.")

# Add an image/banner
if os.path.exists("COVID-DATA2.JPEG"):
    st.image("COVID-DATA2.JPEG", use_column_width=True, caption="🌍 Global COVID-19 Insights")

# Sidebar Navigation
st.sidebar.title("📌 Navigation")
sections = [
    "Upload Dataset",
    "Data Overview",
    "EDA",
    "Top Countries",
    "Visualization",
    "🌎 Global Summary",
    "Prediction Model"
]
selected_section = st.sidebar.radio("Go to Section", sections)

# Upload dataset
st.sidebar.subheader("📂 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")

# Load data
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

data = None
if uploaded_file:
    try:
        data = load_data(uploaded_file)
    except Exception:
        st.error("❌ Invalid dataset format. Please upload a valid COVID-19 dataset.")

# Warning & Data Check
if not uploaded_file and selected_section != "Upload Dataset":
    st.warning("⚠ Please upload a valid COVID-19 dataset. Without data, next sections will not execute.")
    st.stop()

# Section 1: Data Overview
if selected_section == "Data Overview" and data is not None:
    st.header("📊 Data Overview")
    st.write(data.head())
    st.subheader("Missing Values")
    st.write(data.isnull().sum())
    st.subheader("Data Types")
    st.write(data.dtypes)

# Section 2: EDA
elif selected_section == "EDA" and data is not None:
    st.header("🔎 Exploratory Data Analysis")
    st.write(data.describe())

# Section 3: Top Countries
elif selected_section == "Top Countries" and data is not None:
    st.header("🏆 Top Countries by Cases")
    top_confirmed = data[['Country/Region', 'Confirmed']].sort_values(by='Confirmed', ascending=False).head(5)
    top_deaths = data[['Country/Region', 'Deaths']].sort_values(by='Deaths', ascending=False).head(5)
    st.subheader("Top 5 Countries by Confirmed Cases")
    st.dataframe(top_confirmed)
    st.subheader("Top 5 Countries by Deaths")
    st.dataframe(top_deaths)

# Section 4: Visualization
elif selected_section == "Visualization" and data is not None:
    st.header("📈 Visualizations")
    graph_type = st.selectbox("Choose a graph type", [
        "Bar Chart: Top 5 Countries by Confirmed Cases",
        "Pie Chart: Top 5 Countries by Deaths",
        "Scatter Plot: Confirmed Cases vs Deaths",
        "Line Chart: Time Series Analysis"
    ])

    if graph_type == "Bar Chart: Top 5 Countries by Confirmed Cases":
        top_confirmed = data[['Country/Region', 'Confirmed']].sort_values(by='Confirmed', ascending=False).head(5)
        fig = px.bar(top_confirmed, x="Confirmed", y="Country/Region", orientation="h",
                     color="Confirmed", text="Confirmed", color_continuous_scale="Viridis")
        st.plotly_chart(fig, use_container_width=True)

    elif graph_type == "Pie Chart: Top 5 Countries by Deaths":
        top_deaths = data[['Country/Region', 'Deaths']].sort_values(by='Deaths', ascending=False).head(5)
        fig = px.pie(top_deaths, names="Country/Region", values="Deaths", hole=0.3,
                     color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig, use_container_width=True)

    elif graph_type == "Scatter Plot: Confirmed Cases vs Deaths":
        fig = px.scatter(data, x="Confirmed", y="Deaths", size="Confirmed",
                         color="Country/Region", hover_name="Country/Region")
        st.plotly_chart(fig, use_container_width=True)

    elif graph_type == "Line Chart: Time Series Analysis":
        if 'Date' in data.columns:
            time_series_data = data.groupby('Date').sum().reset_index()
            fig = px.line(time_series_data, x="Date", y=["Confirmed", "Deaths", "Recovered"])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠ Dataset does not contain 'Date' column.")

# Section 5: Global Summary
elif selected_section == "🌎 Global Summary" and data is not None:
    st.header("🌎 Global COVID-19 Cases Summary")

    global_cases = {
        'Confirmed': data['Confirmed'].sum(),
        'Deaths': data['Deaths'].sum(),
        'Recovered': data['Recovered'].sum(),
        'Active': (data['Confirmed'].sum() - data['Deaths'].sum() - data['Recovered'].sum())
    }

    # Cards Layout
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    col1.markdown(f"<div class='metric-card'><div class='metric-value'>🦠 {global_cases['Confirmed']:,}</div><div class='metric-label'>Confirmed</div></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='metric-card'><div class='metric-value'>⚰️ {global_cases['Deaths']:,}</div><div class='metric-label'>Deaths</div></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='metric-card'><div class='metric-value'>💊 {global_cases['Recovered']:,}</div><div class='metric-label'>Recovered</div></div>", unsafe_allow_html=True)
    col4.markdown(f"<div class='metric-card'><div class='metric-value'>🔥 {global_cases['Active']:,}</div><div class='metric-label'>Active</div></div>", unsafe_allow_html=True)

    # Visualization
    summary_df = pd.DataFrame(list(global_cases.items()), columns=["Category", "Count"])
    st.subheader("📊 Global Summary Visualization")
    fig1 = px.bar(summary_df, x="Category", y="Count", color="Category", text_auto=True,
                  color_discrete_sequence=px.colors.sequential.Viridis)
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.pie(summary_df, names="Category", values="Count", hole=0.3,
                  color_discrete_sequence=px.colors.sequential.RdBu)
    st.plotly_chart(fig2, use_container_width=True)

# Section 6: Prediction
elif selected_section == "Prediction Model" and data is not None:
    st.header("🤖 Prediction Model for Future Cases")
    X = data[['Confirmed']].values
    y = data['Deaths'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.subheader("📌 Model Evaluation")
    st.write(f"**Mean Squared Error:** {mse}")
    st.write(f"**R-Squared Value:** {r2}")
    fig = px.scatter(x=X_test.flatten(), y=y_test, labels={'x': 'Confirmed Cases', 'y': 'Deaths'})
    fig.add_scatter(x=X_test.flatten(), y=y_pred, mode="lines", name="Predicted")
    st.plotly_chart(fig, use_container_width=True)

# Footer with 3D Brown Effect
st.sidebar.write("---")
st.sidebar.markdown("<div class='dev-credit'>👨‍💻 Developed by Deependra Pratap Singh</div>", unsafe_allow_html=True)

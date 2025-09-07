import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Global settings for Seaborn and Matplotlib
sns.set_theme(style="darkgrid")
plt.style.use('seaborn-darkgrid')

# Title of the App
st.title("COVID-19 Data Analysis and Prediction")
st.markdown("Analyze COVID-19 cases globally with visualizations and predictive modeling.")

# Add a banner or introductory image
st.image("COVID-DATA2.JPEG", use_column_width=True, caption="Global COVID-19 Insights")

# Sidebar for navigation
st.sidebar.title("Navigation")
sections = ["Upload Dataset", "Data Overview", "EDA", "Top Countries", "Visualization", "Global Summary", "Prediction Model"]
selected_section = st.sidebar.radio("Go to Section", sections)

# Drag-and-drop file uploader
st.sidebar.subheader("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")

# Load data
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

if uploaded_file:
    data = load_data(uploaded_file)
else:
    st.warning("Please upload a dataset to proceed.")
    st.stop()

# Section 1: Data Overview
if selected_section == "Data Overview":
    st.header("Data Overview")
    st.subheader("First 5 Rows of the Dataset")
    st.write(data.head())

    st.subheader("Missing Values")
    st.write(data.isnull().sum())

    st.subheader("Data Types")
    st.write(data.dtypes)

# Section 2: Exploratory Data Analysis (EDA)
elif selected_section == "EDA":
    st.header("Exploratory Data Analysis")
    st.subheader("Descriptive Statistics")
    st.write(data.describe())

    total_confirmed = data['Confirmed'].sum()
    total_deaths = data['Deaths'].sum()
    total_recovered = data['Recovered'].sum()

    st.subheader("Global Totals")
    st.write(f"**Confirmed Cases:** {total_confirmed}")
    st.write(f"**Deaths:** {total_deaths}")
    st.write(f"**Recovered Cases:** {total_recovered}")

# Section 3: Top Countries by Cases
elif selected_section == "Top Countries":
    st.header("Top Countries by Cases")
    top_confirmed = data[['Country/Region', 'Confirmed']].sort_values(by='Confirmed', ascending=False).head(5)
    top_deaths = data[['Country/Region', 'Deaths']].sort_values(by='Deaths', ascending=False).head(5)

    st.subheader("Top 5 Countries by Confirmed Cases")
    st.write(top_confirmed)

    st.subheader("Top 5 Countries by Deaths")
    st.write(top_deaths)

# Section 4: Visualization
elif selected_section == "Visualization":
    st.header("Visualizations")

    # Dropdown menu to select visualization type
    graph_type = st.selectbox(
        "Choose a graph type",
        [
            "Bar Chart: Top 5 Countries by Confirmed Cases",
            "Pie Chart: Top 5 Countries by Deaths",
            "Scatter Plot: Confirmed Cases vs Deaths",
            "Line Chart: Time Series Analysis (if available)"
        ]
    )

    # Visualization 1: Bar Chart for Top 5 Countries by Confirmed Cases
    if graph_type == "Bar Chart: Top 5 Countries by Confirmed Cases":
        st.subheader("Top 5 Countries by Confirmed Cases (Bar Chart)")
        top_confirmed = data[['Country/Region', 'Confirmed']].sort_values(by='Confirmed', ascending=False).head(5)

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x='Confirmed', y='Country/Region', data=top_confirmed, palette='viridis', ax=ax)
        ax.set_title("Top 5 Countries by Confirmed Cases")
        ax.set_xlabel("Confirmed Cases")
        ax.set_ylabel("Country/Region")
        st.pyplot(fig)

    # Visualization 2: Pie Chart for Top 5 Countries by Deaths
    elif graph_type == "Pie Chart: Top 5 Countries by Deaths":
        st.subheader("Top 5 Countries by Deaths (Pie Chart)")
        top_deaths = data[['Country/Region', 'Deaths']].sort_values(by='Deaths', ascending=False).head(5)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(top_deaths['Deaths'], labels=top_deaths['Country/Region'], autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
        ax.set_title("Top 5 Countries by Deaths")
        st.pyplot(fig)

    # Visualization 3: Scatter Plot for Confirmed Cases vs Deaths
    elif graph_type == "Scatter Plot: Confirmed Cases vs Deaths":
        st.subheader("Scatter Plot: Confirmed Cases vs Deaths")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(data['Confirmed'], data['Deaths'], color='purple', alpha=0.7)
        ax.set_title("Confirmed Cases vs Deaths")
        ax.set_xlabel("Confirmed Cases")
        ax.set_ylabel("Deaths")
        st.pyplot(fig)

    # Visualization 4: Line Chart for Time Series Analysis (if available)
    elif graph_type == "Line Chart: Time Series Analysis (if available)":
        st.subheader("Time Series Analysis")
        
        if 'Date' in data.columns:
            # Convert the Date column to datetime if not already done
            data['Date'] = pd.to_datetime(data['Date'])
            time_series_data = data.groupby('Date').sum().reset_index()

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(time_series_data['Date'], time_series_data['Confirmed'], label='Confirmed Cases', color='blue')
            ax.plot(time_series_data['Date'], time_series_data['Deaths'], label='Deaths', color='red')
            ax.plot(time_series_data['Date'], time_series_data['Recovered'], label='Recovered Cases', color='green')
            ax.set_title("Time Series Analysis of COVID-19 Cases")
            ax.set_xlabel("Date")
            ax.set_ylabel("Number of Cases")
            ax.legend()
            st.pyplot(fig)
        else:
            st.warning("The dataset does not contain a 'Date' column for time series analysis.")

# Section 5: Global Summary
elif selected_section == "Global Summary":
    st.header("Global COVID-19 Cases Summary")
    global_cases = {
        'Confirmed': data['Confirmed'].sum(),
        'Deaths': data['Deaths'].sum(),
        'Recovered': data['Recovered'].sum(),
        'Active': (data['Confirmed'].sum() - data['Deaths'].sum() - data['Recovered'].sum())
    }

    st.bar_chart(pd.DataFrame(global_cases, index=["Cases"]))

# Section 6: Prediction Model
elif selected_section == "Prediction Model":
    st.header("Prediction Model for Future Cases")
    X = data[['Confirmed']].values
    y = data['Deaths'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.subheader("Model Evaluation")
    st.write(f"**Mean Squared Error:** {mse}")
    st.write(f"**R-Squared Value:** {r2}")

    fig, ax = plt.subplots()
    ax.scatter(X_test, y_test, color='blue', label='Actual')
    ax.plot(X_test, y_pred, color='red', label='Predicted')
    ax.set_title("Actual vs Predicted Deaths")
    ax.set_xlabel("Confirmed Cases")
    ax.set_ylabel("Deaths")
    ax.legend()
    st.pyplot(fig)

# Footer
st.sidebar.write("---")
st.sidebar.info("Developed by Deependra Pratap Singh")


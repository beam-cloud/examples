"""
This is a simple Streamlit app that allows you to visualize NYC water consumption data.

The only part of this that is specific to Beam is the `if env.is_remote()` block, which
is used to determine if the app is running in a remote environment (e.g. on Beam) or
locally.
"""

from beam import env

if env.is_remote():
    import streamlit as st
    import pandas as pd
    import altair as alt
    import requests

# CSV endpoint for the dataset
DATA_URL = "https://data.cityofnewyork.us/resource/ia2d-e54m.csv"


def load_data():
    """
    Fetch and clean water consumption history data (CSV only).
    Returns a cleaned pandas DataFrame.
    """
    df = pd.read_csv(DATA_URL)

    rename_map = {
        "year": "year",
        "new_york_city_population": "nyc_population",
        "nyc_consumption_million_gallons_per_day": "nyc_consumption_mgd",
        "per_capita_gallons_per_person_per_day": "per_capita_gallons",
    }

    # Rename columns if they exist
    for old_col, new_col in rename_map.items():
        if old_col in df.columns:
            df.rename(columns={old_col: new_col}, inplace=True)

    # Convert columns to numeric where appropriate
    for col in ["year", "nyc_population", "nyc_consumption_mgd", "per_capita_gallons"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows without a valid 'year' (NaN) and sort by year
    df.dropna(subset=["year"], inplace=True)
    df.sort_values(by="year", inplace=True)

    return df


def main():
    st.title("A Brief History of NYC Water Consumption")
    st.markdown(
        """
        **Data Source**: [NYC Open Data](https://data.cityofnewyork.us/resource/ia2d-e54m.csv)<br>
        This dataset shows water consumption in the NYC Water Supply System, 
        plus NYC population, going back to 1979.
        """,
        unsafe_allow_html=True,
    )

    # Load data
    df = load_data()

    # Ensure required columns exist
    required_cols = {
        "year",
        "nyc_population",
        "nyc_consumption_mgd",
        "per_capita_gallons",
    }
    if not required_cols.issubset(df.columns):
        st.error(
            f"The dataset is missing one or more required columns: {required_cols}"
        )
        st.stop()

    # Sidebar filter: Year range
    st.sidebar.header("Filter by Year")
    min_year = int(df["year"].min())
    max_year = int(df["year"].max())
    year_range = st.sidebar.slider(
        "Select a range of years",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
        step=1,
    )

    # Filter data based on chosen year range
    filtered_df = df[(df["year"] >= year_range[0]) & (df["year"] <= year_range[1])]

    # Select which metrics to plot
    st.subheader("Visualize Trends Over Time")
    metrics = st.multiselect(
        "Select metrics to chart (Y-axis)",
        ["nyc_population", "nyc_consumption_mgd", "per_capita_gallons"],
        default=["nyc_consumption_mgd", "per_capita_gallons"],
    )

    if metrics:
        # Prepare data for plotting; set 'year' as index
        chart_data = filtered_df[["year"] + metrics].set_index("year")
        st.line_chart(chart_data)
    else:
        st.info("Please select at least one metric to visualize.")

    # Data preview
    st.subheader(f"Data Preview: {year_range[0]} to {year_range[1]}")
    st.write(filtered_df.head(10))
    # Quick statistics
    st.subheader("Summary Statistics (for Selected Range)")
    col1, col2 = st.columns(2)

    with col1:
        if not filtered_df["nyc_consumption_mgd"].isna().all():
            avg_consumption = filtered_df["nyc_consumption_mgd"].mean()
            st.metric("Avg Consumption (MGD)", f"{avg_consumption:,.1f}")
        if not filtered_df["per_capita_gallons"].isna().all():
            avg_per_capita = filtered_df["per_capita_gallons"].mean()
            st.metric("Avg Per Capita (GPD)", f"{avg_per_capita:,.1f}")

    with col2:
        if not filtered_df["nyc_consumption_mgd"].isna().all():
            max_consumption = filtered_df["nyc_consumption_mgd"].max()
            st.metric("Max Consumption (MGD)", f"{max_consumption:,.1f}")
        if not filtered_df["per_capita_gallons"].isna().all():
            min_per_capita = filtered_df["per_capita_gallons"].min()
            st.metric("Min Per Capita (GPD)", f"{min_per_capita:,.1f}")

    # Download button for filtered data
    st.subheader("Download Filtered Data")
    csv_data = filtered_df.to_csv(index=False)
    st.download_button(
        label="Download as CSV",
        data=csv_data,
        file_name="nyc_water_consumption_filtered.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()

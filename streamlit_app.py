"""
Streamlit Mobility Dashboard for HHtbilisi2025
============================================

This Streamlit application mirrors the functionality of the original
Dash‑based dashboard using Streamlit widgets and Plotly charts.  It
provides an interactive environment for exploring household travel
survey data.  Users can filter trips by purpose, age range and sex
and observe how these selections influence a variety of metrics and
visualisations.  The layout is organised into a sidebar for filter
controls and a main area with responsive charts.

To run the app locally install the required packages from
``requirements.txt`` and then execute:

.. code-block:: bash

   streamlit run streamlit_app.py

The app will start on a local port (typically http://localhost:8501/).

When deploying to Streamlit Cloud, point the app configuration at
``streamlit_app.py`` and ensure that the sample data CSV file
``sample_data.csv`` is present in the repository.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    """Load trip data from a CSV file and cast categorical columns.

    Parameters
    ----------
    path: str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the trips and associated person and
        household attributes.
    """
    df = pd.read_csv(path)
    cat_cols = [
        "purpose",
        "mode",
        "sex",
        "employment",
        "car_ownership",
    ]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


@st.cache_data
def prepare_household_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate trip data to a household level summary.

    Parameters
    ----------
    df: pd.DataFrame
        Trip level data.

    Returns
    -------
    pd.DataFrame
        Aggregated DataFrame with one row per household containing
        total trips, number of persons, income, car ownership and
        trips per person.
    """
    # Before grouping, ensure that the car_ownership column contains only strings.
    # In the sample data, car_ownership values may include None, which cannot
    # be compared with strings when taking the maximum. This leads to a
    # TypeError when using the "max" aggregation. To avoid this, first
    # fill missing values with a placeholder string and then use an
    # aggregation function that simply returns the first non-null value.
    df = df.copy()
    # Replace None/NaN values with a consistent string to avoid comparison
    # errors when grouping.
    if "car_ownership" in df.columns:
        # Convert categorical car_ownership values to strings before filling missing values.
        # Using fillna on a Categorical column with a new value raises an error, so we cast
        # to object/str first and then replace missing entries.  Pandas converts NaN to the
        # string "nan" when casting to str, so replace that sentinel with a consistent
        # placeholder value ("None").  This ensures the aggregation below operates on
        # standard Python strings without introducing new categories.
        df["car_ownership"] = df["car_ownership"].astype(str)
        df["car_ownership"] = df["car_ownership"].replace("nan", "None")

    # Ensure household-level numeric fields are truly numeric. In some
    # datasets these columns may be read as object or categorical due to
    # missing values or unexpected strings. Converting them to numeric
    # prevents groupby aggregations (e.g. "max") from failing on
    # non‑ordered categorical data. Invalid values will be coerced to NaN
    # and ignored by the aggregation operations.
    for col in ["num_persons", "household_income"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    grouped = (
        df.groupby("household_id").agg(
            trips_total=("trip_id", "count"),
            persons=("num_persons", "max"),
            income=("household_income", "max"),
            # Use 'first' instead of 'max' for car_ownership to avoid
            # comparisons between strings and NoneTypes. Since we filled
            # missing values, 'first' will return a representative value.
            car_ownership=("car_ownership", "first"),
        )
        .reset_index()
    )
    # Calculate trips per person; guard against division by zero
    grouped["trips_per_person"] = grouped["trips_total"] / grouped["persons"].replace({0: np.nan})
    return grouped


@st.cache_data
def compute_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the correlation matrix for selected numeric variables."""
    numeric_cols = [
        "age",
        "distance_km",
        "duration_min",
        "household_income",
    ]
    corr = df[numeric_cols].corr()
    return corr


def main() -> None:
    """Run the Streamlit dashboard."""
    st.set_page_config(page_title="Mobility Dashboard", layout="wide")
    st.title("Household Travel Survey Dashboard")

    # Load data
    DATA_PATH = "sample_data.csv"
    try:
        data = load_data(DATA_PATH)
    except FileNotFoundError:
        st.error(f"Data file '{DATA_PATH}' not found. Please ensure the CSV is present in the application directory.")
        return

    # Sidebar filters
    st.sidebar.header("Filters")
    available_purposes = sorted(data["purpose"].cat.categories.tolist())
    available_modes = sorted(data["mode"].cat.categories.tolist())
    available_sexes = sorted(data["sex"].cat.categories.tolist())

    selected_purposes = st.sidebar.multiselect(
        "Trip Purpose", options=available_purposes, default=[],
        help="Select one or more purposes to filter trips. Leave empty to include all."
    )

    # Age slider: uses a tuple for range
    min_age, max_age = int(data["age"].min()), int(data["age"].max())
    age_range = st.sidebar.slider(
        "Age Range", min_value=min_age, max_value=max_age,
        value=(min_age, max_age), step=1
    )

    selected_sexes = st.sidebar.multiselect(
        "Sex", options=available_sexes, default=[],
        help="Select one or more sex categories to filter trips."
    )

    st.sidebar.markdown("---")
    st.sidebar.write("Data set contains", len(data), "trip records.")

    # Apply filters
    df_filtered = data.copy()
    if selected_purposes:
        df_filtered = df_filtered[df_filtered["purpose"].isin(selected_purposes)]
    if age_range:
        df_filtered = df_filtered[
            (df_filtered["age"] >= age_range[0]) & (df_filtered["age"] <= age_range[1])
        ]
    if selected_sexes:
        df_filtered = df_filtered[df_filtered["sex"].isin(selected_sexes)]

    # Compute household summary for filtered data
    hh_summary = prepare_household_summary(df_filtered) if not df_filtered.empty else prepare_household_summary(data)

    # Layout: Use Streamlit columns to arrange charts in rows
    # Row 1: Trips by purpose and trips by age
    col1, col2 = st.columns(2)
    with col1:
        purpose_counts = (
            df_filtered.groupby("purpose")
            .size()
            .reindex(available_purposes, fill_value=0)
            .reset_index(name="count")
        )
        fig_purpose = px.bar(
            purpose_counts,
            x="purpose",
            y="count",
            labels={"purpose": "Trip purpose", "count": "Number of trips"},
            title="Trips by Purpose",
            color="purpose",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_purpose.update_layout(showlegend=False, xaxis_tickangle=-45)
        st.plotly_chart(fig_purpose, use_container_width=True)

    with col2:
        fig_age = px.histogram(
            df_filtered,
            x="age",
            nbins=20,
            labels={"age": "Age", "count": "Number of trips"},
            title="Trips by Age",
            color_discrete_sequence=["#2ca02c"],
        )
        fig_age.update_layout(bargap=0.1)
        st.plotly_chart(fig_age, use_container_width=True)

    # Row 2: Correlation matrix and household metrics scatter
    col3, col4 = st.columns(2)
    with col3:
        if not df_filtered.empty:
            corr_df = compute_correlation(df_filtered)
            fig_corr = px.imshow(
                corr_df,
                text_auto=True,
                color_continuous_scale="RdBu",
                zmin=-1,
                zmax=1,
                labels=dict(x="Variable", y="Variable", color="Correlation"),
                title="Correlation Matrix (Selected Data)",
            )
        else:
            # Display empty heatmap when no data matches the filters
            fig_corr = px.imshow(
                np.zeros((4, 4)),
                x=["age", "distance_km", "duration_min", "household_income"],
                y=["age", "distance_km", "duration_min", "household_income"],
                color_continuous_scale="RdBu",
                title="Correlation Matrix (No data)"
            )
        st.plotly_chart(fig_corr, use_container_width=True)

    with col4:
        fig_household = px.scatter(
            hh_summary,
            x="income",
            y="trips_per_person",
            size="persons",
            color="car_ownership",
            labels={
                "income": "Household Income",
                "trips_per_person": "Trips per Person",
                "car_ownership": "Car Ownership",
                "persons": "Household Size",
            },
            title="Household Metrics",
            hover_data=["trips_total"] if "trips_total" in hh_summary.columns else None,
        )
        fig_household.update_layout(legend_title="Car Ownership")
        st.plotly_chart(fig_household, use_container_width=True)

    # Row 3: Mode distribution and duration by mode
    col5, col6 = st.columns(2)
    with col5:
        mode_counts = (
            df_filtered.groupby("mode")
            .size()
            .reindex(available_modes, fill_value=0)
            .reset_index(name="count")
        )
        fig_mode = px.pie(
            mode_counts,
            names="mode",
            values="count",
            title="Trip Mode Distribution",
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        st.plotly_chart(fig_mode, use_container_width=True)

    with col6:
        # Show box plot only if there is data
        if not df_filtered.empty:
            fig_duration = px.box(
                df_filtered,
                x="mode",
                y="duration_min",
                labels={"mode": "Mode", "duration_min": "Duration (min)"},
                title="Trip Duration by Mode",
                color="mode",
                color_discrete_sequence=px.colors.qualitative.Set3,
            )
            fig_duration.update_layout(showlegend=False)
            st.plotly_chart(fig_duration, use_container_width=True)
        else:
            st.write("No trip data available to display duration by mode.")


if __name__ == "__main__":
    main()
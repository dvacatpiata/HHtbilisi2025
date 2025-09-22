"""
Enhanced Streamlit Mobility Dashboard for HHtbilisi2025
=======================================================

This Streamlit application mirrors the enhanced Dash dashboard.  It
loads the full household travel survey dataset from
``sample_data_full.csv`` and offers the following interactive
visualisations:

* **Trips by Purpose** – bar chart summarising the number of trips
  for each purpose category.
* **Trips by Age** – histogram of trip counts across age groups.
* **Correlation Matrix** – heatmap showing correlations between
  numeric variables (age, estimated distance, trip duration and
  household income).
* **Household Metrics** – scatter plot of trips per person versus
  household income with marker size encoding household size and colour
  indicating car ownership.
* **Mode Distribution** – pie chart of transport modes.
* **Duration by Mode** – box plot of trip durations grouped by mode.
* **Trips by Employment Status** – bar chart showing trip counts for
  each employment category.
* **Duration vs Distance** – scatter plot comparing duration and
  estimated distance with points coloured by trip purpose.

Filters in the sidebar allow users to select trip purpose(s), a range
of ages and sex categories.  All charts update accordingly.  The
sidebar also displays the number of records in the loaded dataset so
you can verify that the full survey is being used.

To run this app locally install the required packages and execute:

.. code-block:: bash

    pip install streamlit pandas plotly
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    """Load trip data from a CSV file and cast categorical columns."""
    df = pd.read_csv(path)
    cat_cols = [
        "purpose",
        "mode",
        "sex",
        "employment",
        "car_ownership",
        # Treat 'age' as a categorical variable because the full dataset uses ranges like
        # '36-45 years' and '66 and more' rather than numeric ages.
        "age",
    ]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


@st.cache_data
def prepare_household_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate trip data to a household level summary."""
    df = df.copy()
    # Ensure household-level numeric fields are numeric
    for col in ["num_persons", "household_income"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    grouped = (
        df.groupby("household_id").agg(
            trips_total=("trip_id", "count"),
            persons=("num_persons", "max"),
            income=("household_income", "max"),
            car_ownership=("car_ownership", "first"),
        )
        .reset_index()
    )
    grouped["trips_per_person"] = grouped["trips_total"] / grouped["persons"].replace({0: np.nan})
    return grouped


@st.cache_data
def compute_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the correlation matrix for selected numeric variables.

    Age is excluded from the numeric correlation because in the full survey
    dataset it is a categorical range (e.g. "36-45 years").  Including
    it would coerce categories to arbitrary codes and yield misleading
    results.  Instead we focus on the truly numeric fields.
    """
    numeric_cols = [
        "distance_km",
        "duration_min",
        "household_income",
    ]
    # Only compute correlation on the available numeric columns to avoid
    # accidental inclusion of non-numeric categories
    return df[numeric_cols].corr()


def main() -> None:
    """Run the Streamlit dashboard."""
    st.set_page_config(page_title="Mobility Dashboard", layout="wide")
    st.title("Household Travel Survey Dashboard")

    # Load data
    DATA_PATH = "sample_data_full.csv"
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

    # Age filter: use categorical age ranges instead of numeric slider
    # The 'age' column in the full survey dataset contains categories like
    # "36-45 years" or "66 and more" rather than numeric values.  A numeric
    # slider will fail when attempting to cast these strings to integers.
    # Instead, present a multiselect to choose one or more age categories.
    available_ages = sorted(data["age"].astype(str).unique().tolist())
    selected_ages = st.sidebar.multiselect(
        "Age Categories", options=available_ages, default=available_ages,
        help="Select one or more age categories to filter trips."
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
    if selected_ages:
        df_filtered = df_filtered[df_filtered["age"].isin(selected_ages)]
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
            # Do not set a specific colour sequence. Let Plotly assign colours for maximum compatibility.
        )
        fig_purpose.update_layout(showlegend=False, xaxis_tickangle=-45)
        st.plotly_chart(fig_purpose, use_container_width=True)

    with col2:
        # Histogram of age categories.  We omit `nbins` because age is
        # categorical and Plotly will automatically produce a bar for each
        # distinct category.
        fig_age = px.histogram(
            df_filtered,
            x="age",
            labels={"age": "Age", "count": "Number of trips"},
            title="Trips by Age",
            # Let Plotly choose the colour palette automatically.
        )
        fig_age.update_layout(bargap=0.1)
        st.plotly_chart(fig_age, use_container_width=True)

    # Row 2: Correlation matrix and household metrics scatter
    col3, col4 = st.columns(2)
    with col3:
        if not df_filtered.empty:
            # Compute the correlation matrix on the filtered data.  We build
            # the heatmap using graph_objects instead of px.imshow to maximise
            # compatibility with older Plotly versions where px.imshow may not
            # be available.  Axis labels are explicitly provided and a
            # colourbar is added.
            corr_df = compute_correlation(df_filtered)
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_df.values,
                x=corr_df.columns.tolist(),
                y=corr_df.index.tolist(),
                colorscale="RdBu",
                zmin=-1,
                zmax=1,
                colorbar=dict(title="Correlation"),
            ))
            fig_corr.update_layout(
                title="Correlation Matrix (Selected Data)",
                xaxis_title="Variable",
                yaxis_title="Variable",
            )
        else:
            # If no data matches the filters, show an empty correlation heatmap
            # for the numeric columns used in the correlation.  Age is
            # excluded because it is a categorical variable.
            placeholder_cols = ["distance_km", "duration_min", "household_income"]
            fig_corr = go.Figure(data=go.Heatmap(
                z=np.zeros((len(placeholder_cols), len(placeholder_cols))),
                x=placeholder_cols,
                y=placeholder_cols,
                colorscale="RdBu",
            ))
            fig_corr.update_layout(
                title="Correlation Matrix (No data)",
                xaxis_title="Variable",
                yaxis_title="Variable",
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
            # Do not set a colour sequence; use default palette for compatibility.
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
                # Use default colours; avoid specifying a discrete colour sequence.
            )
            fig_duration.update_layout(showlegend=False)
            st.plotly_chart(fig_duration, use_container_width=True)
        else:
            st.write("No trip data available to display duration by mode.")

    # Row 4: Trips by employment and duration vs distance
    col7, col8 = st.columns(2)
    with col7:
        employment_counts = (
            df_filtered.groupby("employment")
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        fig_employment = px.bar(
            employment_counts,
            x="employment",
            y="count",
            labels={"employment": "Employment status", "count": "Number of trips"},
            title="Trips by Employment Status",
            color="employment",
            # Avoid specifying a colour sequence; rely on default palette.
        )
        fig_employment.update_layout(showlegend=False, xaxis_tickangle=-45)
        st.plotly_chart(fig_employment, use_container_width=True)

    with col8:
        if not df_filtered.empty:
            fig_dur_dist = px.scatter(
                df_filtered,
                x="distance_km",
                y="duration_min",
                color="purpose",
                labels={"distance_km": "Distance (km)", "duration_min": "Duration (min)", "purpose": "Purpose"},
                title="Duration vs Distance (Filtered)",
                hover_data=["mode", "employment"],
            )
            st.plotly_chart(fig_dur_dist, use_container_width=True)
        else:
            st.write("No trip data available to display duration vs distance.")


if __name__ == "__main__":
    main()

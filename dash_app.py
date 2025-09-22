"""
Enhanced Mobility Dashboard for HHtbilisi2025
===========================================

This Dash application builds on the original mobility dashboard and
incorporates the full household travel survey dataset.  In addition to
the existing visualisations (trips by purpose, trips by age, a
correlation matrix, household metrics, mode distribution and trip
duration by mode) the dashboard now includes two extra charts:

* **Trips by Employment Status** – a bar chart showing how trip
  frequency varies across different employment categories.  This
  helps reveal whether retirees, students or full‑time workers tend to
  travel more frequently.
* **Duration vs Distance Scatter** – a scatter plot comparing the
  recorded duration of each trip (in minutes) against an estimated
  distance (km).  Points are coloured by trip purpose so you can
  identify which types of journeys are typically longer or shorter.

The application reads its data from ``sample_data_full.csv`` and
automatically adapts to the number of records it contains.  A
reduced sample of the original 1 000 records (``sample_data_backup.csv``)
is also available for reference or recovery.

To run the app locally install the required packages and execute:

.. code-block:: bash

    pip install dash dash-bootstrap-components pandas plotly
    python dash_app.py

The server will start on http://127.0.0.1:8050/ by default.
"""

import os

import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# Data loading and preparation
# -----------------------------------------------------------------------------

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
    # Convert categorical columns to category dtype for efficiency
    # Include age as a categorical column.  In the full survey dataset
    # age values are strings like "36-45 years" rather than numeric
    # integers.  Converting to a pandas Categorical type ensures that
    # downstream operations (e.g., filtering and grouping) treat age
    # values as discrete categories rather than attempting numeric
    # comparisons.
    cat_cols = [
        "purpose",
        "mode",
        "sex",
        "employment",
        "car_ownership",
        "age",
    ]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def prepare_household_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate trip data to a household level summary.

    The summary computes the total number of trips per household and the
    average number of trips per person in the household.  It also retains
    household income, car ownership and number of persons.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame returned by :func:`load_data`.

    Returns
    -------
    pd.DataFrame
        Aggregated DataFrame with one row per household.
    """
    grouped = df.groupby("household_id").agg(
        trips_total=("trip_id", "count"),
        persons=("num_persons", "max"),
        income=("household_income", "max"),
        car_ownership=("car_ownership", "first"),
    ).reset_index()
    grouped["trips_per_person"] = grouped["trips_total"] / grouped["persons"]
    return grouped


def compute_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the correlation matrix for selected numeric variables.

    Age is excluded from the numeric correlation because it is stored
    as a categorical range (e.g. "36-45 years") in the full survey
    dataset.  Including it would coerce categories to arbitrary
    ordinal codes and yield misleading results.

    Parameters
    ----------
    df: pd.DataFrame
        Filtered DataFrame of trip records.

    Returns
    -------
    pd.DataFrame
        Correlation matrix as a DataFrame.
    """
    numeric_cols = [
        "distance_km",
        "duration_min",
        "household_income",
    ]
    corr = df[numeric_cols].corr()
    return corr


# Load the dataset (full survey)
DATA_PATH = "sample_data_full.csv"
try:
    data = load_data(DATA_PATH)
except FileNotFoundError:
    raise FileNotFoundError(
        f"Data file '{DATA_PATH}' not found. Please ensure the CSV is present in the application directory."
    )

household_summary = prepare_household_summary(data)

# -----------------------------------------------------------------------------
# Dash application setup
# -----------------------------------------------------------------------------

external_stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Mobility Dashboard"

available_purposes = sorted(data["purpose"].cat.categories.tolist())
available_modes = sorted(data["mode"].cat.categories.tolist())
available_sexes = sorted(data["sex"].cat.categories.tolist())
# Extract the unique age categories as strings for the age filter.  We
# cannot treat age as numeric because the full survey encodes age in
# ranges (e.g. "36-45 years", "66 and more").  Using strings here
# allows us to populate a drop-down for multi-select filtering.
available_ages = sorted(data["age"].astype(str).unique().tolist())


def create_layout() -> html.Div:
    """Construct the Dash app layout."""
    return dbc.Container([
        dbc.Row([
            dbc.Col(html.H2("Household Travel Survey Dashboard"), width=12)
        ], className="mb-3"),
        # Filters row
        dbc.Row([
            dbc.Col([
                html.Label("Trip Purpose", className="form-label"),
                dcc.Dropdown(
                    id="purpose-filter",
                    options=[{"label": p, "value": p} for p in available_purposes],
                    value=[],  # no default selection
                    multi=True,
                    placeholder="Select purpose(s)",
                ),
            ], md=4),
            dbc.Col([
                html.Label("Age Categories", className="form-label"),
                dcc.Dropdown(
                    id="age-filter",
                    options=[{"label": a, "value": a} for a in available_ages],
                    value=[],  # no default selection; empty means all ages
                    multi=True,
                    placeholder="Select age category(ies)",
                ),
            ], md=4),
            dbc.Col([
                html.Label("Sex", className="form-label"),
                dcc.Dropdown(
                    id="sex-filter",
                    options=[{"label": s, "value": s} for s in available_sexes],
                    value=[],
                    multi=True,
                    placeholder="Select sex"
                ),
            ], md=4),
        ], className="mb-4"),
        # First row of charts
        dbc.Row([
            dbc.Col(dcc.Graph(id="trips-by-purpose"), md=6),
            dbc.Col(dcc.Graph(id="trips-by-age"), md=6),
        ], className="mb-4"),
        # Second row of charts
        dbc.Row([
            dbc.Col(dcc.Graph(id="corr-matrix"), md=6),
            dbc.Col(dcc.Graph(id="household-metrics"), md=6),
        ], className="mb-4"),
        # Third row of charts
        dbc.Row([
            dbc.Col(dcc.Graph(id="mode-distribution"), md=6),
            dbc.Col(dcc.Graph(id="duration-by-mode"), md=6),
        ], className="mb-4"),
        # Fourth row: additional charts
        dbc.Row([
            dbc.Col(dcc.Graph(id="trips-by-employment"), md=6),
            dbc.Col(dcc.Graph(id="duration-vs-distance"), md=6),
        ], className="mb-4"),
    ], fluid=True)


app.layout = create_layout()


# -----------------------------------------------------------------------------
# Callback to update charts based on filters
# -----------------------------------------------------------------------------

@app.callback(
    Output("trips-by-purpose", "figure"),
    Output("trips-by-age", "figure"),
    Output("corr-matrix", "figure"),
    Output("household-metrics", "figure"),
    Output("mode-distribution", "figure"),
    Output("duration-by-mode", "figure"),
    Output("trips-by-employment", "figure"),
    Output("duration-vs-distance", "figure"),
    Input("purpose-filter", "value"),
    # Use the age-filter dropdown instead of the numeric range slider.  This
    # receives a list of selected age categories (empty list means no filter).
    Input("age-filter", "value"),
    Input("sex-filter", "value"),
)
def update_charts(selected_purposes, selected_ages, selected_sexes):
    # Filter data by purpose, age and sex
    df_filtered = data.copy()
    # Purpose filter
    if selected_purposes:
        df_filtered = df_filtered[df_filtered["purpose"].isin(selected_purposes)]
    # Age category filter.  When one or more age categories are selected
    # filter by inclusion; if no categories are selected, retain all ages.
    if selected_ages:
        df_filtered = df_filtered[df_filtered["age"].isin(selected_ages)]
    # Sex filter
    if selected_sexes:
        df_filtered = df_filtered[df_filtered["sex"].isin(selected_sexes)]

    # Trips by purpose bar chart
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
        # Use default colours; omit explicit qualitative palette.
    )
    fig_purpose.update_layout(showlegend=False, xaxis_tickangle=-45)

    # Trips by age histogram
    # Histogram of age categories.  We omit `nbins` because age is
    # categorical and Plotly will automatically create a bar for each
    # distinct category.
    fig_age = px.histogram(
        df_filtered,
        x="age",
        labels={"age": "Age", "count": "Number of trips"},
        title="Trips by Age",
        # Do not specify a custom colour sequence for categorical ages; allow Plotly to
        # assign default colours automatically.  Explicit colour lists can
        # sometimes cause errors when the number of categories exceeds
        # the number of colours in the sequence.
    )
    fig_age.update_layout(bargap=0.1)

    # Correlation matrix heatmap
    if not df_filtered.empty:
        corr = compute_correlation(df_filtered)
        # Build the heatmap using graph_objects for maximum compatibility.
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
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
        # Empty heatmap placeholder with the numeric columns used in
        # compute_correlation (age is excluded because it is categorical)
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

    # Household metrics scatter: average trips per person vs income
    hh_df = household_summary.copy()
    # If there is a person filter, recompute household summary for the filtered set
    if not df_filtered.empty:
        hh_filtered = df_filtered.groupby("household_id").agg(
            trips_total=("trip_id", "count"),
            persons=("num_persons", "max"),
            income=("household_income", "max"),
            car_ownership=("car_ownership", "first"),
        ).reset_index()
        hh_filtered["trips_per_person"] = hh_filtered["trips_total"] / hh_filtered["persons"]
        hh_df = hh_filtered

        fig_household = px.scatter(
        hh_df,
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
        # Only include ``trips_total`` in the hover data when it exists in
        # the household summary.  Passing ``None`` when the column is
        # absent prevents Plotly from raising an error about an unknown
        # variable.
        hover_data=["trips_total"] if "trips_total" in hh_df.columns else None,
    )
    fig_household.update_layout(legend_title="Car Ownership")

    # Mode distribution
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
        # Rely on default colour cycle; avoid specifying a qualitative palette.
    )

    # Duration by mode box plot
    fig_duration_mode = px.box(
        df_filtered,
        x="mode",
        y="duration_min",
        labels={"mode": "Mode", "duration_min": "Duration (min)"},
        title="Trip Duration by Mode",
        color="mode",
        # Use default colours; omit explicit colour sequence.
    )
    fig_duration_mode.update_layout(showlegend=False)

    # Trips by employment bar chart
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
        # Default colour palette; no specific sequence set.
    )
    fig_employment.update_layout(showlegend=False, xaxis_tickangle=-45)

    # Duration vs Distance scatter plot
    fig_dur_dist = px.scatter(
        df_filtered,
        x="distance_km",
        y="duration_min",
        color="purpose",
        labels={"distance_km": "Distance (km)", "duration_min": "Duration (min)", "purpose": "Purpose"},
        title="Duration vs Distance (Filtered)",
        hover_data=["mode", "employment"],
    )

    return (
        fig_purpose,
        fig_age,
        fig_corr,
        fig_household,
        fig_mode,
        fig_duration_mode,
        fig_employment,
        fig_dur_dist,
    )


if __name__ == "__main__":
    # Run the Dash app
    port = int(os.environ.get("PORT", 8050))
    app.run_server(host="0.0.0.0", port=port, debug=False)

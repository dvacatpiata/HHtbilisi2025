# Household Travel Survey Dashboard (Streamlit)

This repository contains a prototype mobility dashboard built with [Streamlit](https://streamlit.io/) to explore household travel survey data. It was created as part of the HHtbilisi2025 project and follows contemporary design guidelines for interactive dashboards. By combining trip-level, person-level and household-level attributes, the application enables rich analysis of mobility patterns and correlations.

## Key features

- **Interactive filtering** – users can select one or more trip purposes, restrict the age range via a slider and select gender. These controls update all charts simultaneously, enabling cross-filtering across multiple views.

- **Trip analysis** – bar and histogram charts show the number of trips by purpose and by age, allowing quick comparisons across categories.

- **Correlation matrix** – a heatmap visualises the correlation between numeric variables (age, distance, duration and income) in the filtered dataset. This helps identify which variables are positively or negatively related.

- **Household metrics** – a scatter plot summarises household characteristics, plotting average trips per person against household income. Circle size reflects household size and colour encodes car ownership, helping to reveal relationships between household attributes and travel behaviour.

- **Mode distribution** – a pie chart shows how the share of trips by mode (car, public transport, walking, bicycle, other) changes with the selected filters.

- **Duration by mode** – a box plot compares trip duration across different modes, highlighting which transport modes tend to be faster or slower.

## Installation

To set up the dashboard locally, install the required packages and run the app using Streamlit:

```
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Data

The dashboard expects a CSV file with one row per trip and columns describing trip, person and household attributes. A sample dataset (`sample_data.csv`) is provided with the following columns:

| Column | Description |
| --- | --- |
| trip_id | Unique identifier for each trip. |
| person_id | Person identifier. |
| household_id | Household identifier. |
| age | Age of the traveller. |
| sex | Sex of the traveller. |
| employment | Employment status (employed, unemployed, retired, student or other). |
| purpose | Purpose of the trip (e.g., work, education, shopping, leisure, personal business, visiting friends/family). |
| mode | Mode of transport (car, public transport, walking, bicycle, other). |
| duration_min | Duration of the trip in minutes. |
| distance_km | Distance travelled in kilometres. |
| income | Annual household income in USD. |
| household_size | Number of people in the household. |
| cars | Number of cars owned by the household. |

Replace `sample_data.csv` with your own survey data using the same column structure to explore real results.

## Running the app

To launch the dashboard on your local machine, run:

```
streamlit run streamlit_app.py
```

Streamlit will open the app in your default web browser at `http://localhost:8501`. If deploying to [Streamlit Cloud](https://streamlit.io/cloud), set the entry point to `streamlit_app.py` and ensure the `sample_data.csv` file is included in the repository.

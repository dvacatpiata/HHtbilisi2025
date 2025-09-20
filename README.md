# Household Travel Survey Dashboard

This repository contains a prototype mobility dashboard built with
[Dash](https://dash.plotly.com) to explore household travel survey data.  It
was created as part of the HHtbilisi2025 project and follows
contemporary design guidelines for interactive dashboards.  By
combining trip‑level, person‑level and household‑level attributes, the
application enables rich analysis of mobility patterns and
correlations.

## Key features

* **Interactive filtering** – users can choose one or more trip purposes,
  restrict the age range via a slider and select gender.  These
  filters update all visualisations simultaneously, enabling
  cross‑filtering across multiple views【172438693224181†L465-L473】.

* **Trip analysis** – bar and histogram charts show the number of
  trips by purpose and by age, allowing quick comparisons across
  categories.

* **Correlation matrix** – a heatmap visualises the correlation between
  numeric variables (age, distance, duration and income) in the
  filtered dataset.  This helps identify which variables are
  positively or negatively related.

* **Household metrics** – a scatter plot summarises household
  characteristics, plotting average trips per person against
  household income.  Marker size reflects household size and colour
  encodes car ownership, helping to reveal relationships between
  resources and travel behaviour.

* **Mode of transport** – a pie chart shows the share of different
  transport modes, while a box plot compares trip durations across
  modes.

* **Accessible design** – the number of filters is kept manageable,
  labels are descriptive and tooltips guide the user.  These design
  decisions follow best practices for interactive dashboards【172438693224181†L420-L427】.
  Components are grouped logically and the layout adapts to
  different screen sizes【172438693224181†L598-L602】.

## Installation

Clone this repository and install the Python dependencies:

```bash
pip install -r requirements.txt
```

Alternatively install the packages manually:

```bash
pip install dash dash-bootstrap-components pandas plotly
```

## Data

The app expects a CSV file named **`sample_data.csv`** in the
repository root.  A synthetic dataset is provided for demonstration
purposes.  It includes the following columns:

| Column             | Description                                   |
|--------------------|-----------------------------------------------|
| `trip_id`          | Unique identifier for each trip               |
| `person_id`        | Identifier linking each trip to a traveller    |
| `household_id`     | Identifier linking each traveller to a household |
| `age`              | Age of the traveller                          |
| `sex`              | Gender (`M` or `F`)                           |
| `employment`       | Employment status                             |
| `purpose`          | Trip purpose (e.g. Shopping, Commuting)        |
| `mode`             | Transport mode (e.g. Car, Walking)             |
| `distance_km`      | Trip length in kilometres                     |
| `duration_min`     | Trip duration in minutes                      |
| `day_of_week`      | Day on which the trip occurred                |
| `num_persons`      | Number of people in the household             |
| `household_income` | Annual household income in local currency      |
| `car_ownership`    | Category describing car ownership              |

To use your own survey data, ensure your CSV contains the columns above.

## Running the app

Run the Dash application locally with:

```bash
python dash_app.py
```

This starts a development server accessible at `http://127.0.0.1:8050/`.  When
deployed to a production environment, you can adjust the host and
port in the `dash_app.py` script.

## References

This dashboard was informed by research on mobility analytics and
interactive visualisation.  The National Travel Survey 2023 notes
that shopping was the most common trip purpose in England with 169
trips per person in 2023, while commuting ranked second with 117
trips【846367172183715†L153-L160】.  Males and females show different travel
behaviour: males made 6% fewer trips (887 per person) than females (942
per person) but travelled 15% further, reflecting more commuting and
business trips compared with shorter shopping trips by females【846367172183715†L202-L218】.

Best‑practice recommendations for interactive dashboards also guided
the design.  Only the most relevant filters are exposed and they are


## Live app

The dashboard is deployed on Render and can be accessed at the following link:

[https://hhtbilisi2025.onrender.com](https://hhtbilisi2025.onrender.com)

Please note that the free Render instance will spin down after periods of inactivity, which may lead to a delay of about a minute when starting up.
clearly labelled to reduce cognitive load【172438693224181†L420-L427】.  When users
select values, all visualisations update together to provide coherent
feedback【172438693224181†L465-L473】.  Filters are positioned at the top of the
page and grouped logically, and the layout remains responsive on
different devices【172438693224181†L598-L602】.  Data is aggregated where
possible to keep interactions fast【172438693224181†L492-L502】.

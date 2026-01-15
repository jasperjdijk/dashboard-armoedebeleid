# Dashboard Armoedebeleid

An interactive Streamlit dashboard for visualizing and analyzing poverty policy data.

## Features

- **Overview**: Get a quick summary of all available datasets and key statistics
- **Data Explorer**: Browse and filter any dataset with customizable columns and filters
- **Municipal Analysis**: Analyze regulations by municipality with interactive charts
- **Regulations Analysis**: Deep dive into regulation types, household types, and target groups
- **Custom Charts**: Build your own visualizations with various chart types

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Running the App

Run the following command in your terminal:
```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`

## Data

The app uses data from `dataoverzicht_dashboard_armoedebeleid.xlsx` which contains information about:
- Municipal regulations
- Household types
- Target groups
- Population statistics
- And more...

## Usage Tips

- Use the sidebar to navigate between different views
- Apply filters in the Data Explorer to focus on specific subsets
- Download filtered data as CSV for further analysis
- Build custom visualizations in the Custom Charts section

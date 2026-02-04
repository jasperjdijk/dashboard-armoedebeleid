# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Streamlit dashboard for visualizing Dutch municipal income-dependent benefits (armoedebeleid). Compares benefit values across municipalities for different household types and income levels.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run dashboard_armoedebeleid.py
```

The app runs at `http://localhost:8501`

## Architecture

### Main File: `dashboard_armoedebeleid.py`

Single-file Streamlit application with this structure:

1. **Page Configuration** - Streamlit setup and custom CSS
2. **Helper Functions** - Data loading, filtering, formatting utilities
3. **Main Application** - UI components and visualizations wrapped in try/except

### Key Functions

- `filter_benefits()` - Core data filtering function. Takes municipality code, household type, income level, and various filters. Returns aggregated benefit values or detailed lists based on `result` parameter ('sum', 'ig', 'list').
- `format_dutch_currency()` - Formats numbers with Dutch conventions (dot for thousands, comma for decimals)
- `add_logo_to_figure()` - Adds watermark logo to Plotly figures

### Data Model

Data source: `dataoverzicht_dashboard_armoedebeleid.xlsx` (sheet: "Totaaloverzicht")

Key columns per household type (HH01-HH04):
- `WRD_{HH}` - Annual benefit value (divided by 12 for monthly display)
- `IG_{HH}` - Income threshold as fraction (1.0 = 100% social minimum)
- `Referteperiode_{HH}` - Required years at low income

Filter columns:
- `GMcode` - Municipality code (e.g., 'GM0363' for Amsterdam)
- `FR` - Formal regulation ('Ja'/'Nee')
- `CAV` - Health insurance discount flag
- `WB`, `BT` - Inclusion flags

### Data Format Optimization

The dashboard supports both Excel (.xlsx) and Parquet (.parquet) formats. **Parquet is strongly recommended for production** as it loads 5-10x faster than Excel, significantly improving cold start times on Cloud Run.

**Convert Excel to Parquet:**
```bash
python convert_to_parquet.py
```

This creates `dataoverzicht_dashboard_armoedebeleid.parquet` containing only the required columns and sheet.

**Using Parquet in Cloud Run:**
1. Upload the `.parquet` file to your data source
2. Update the `EXCEL_URL` environment variable to point to the `.parquet` file
3. The dashboard automatically detects the file format and uses the appropriate loader

**Benefits:**
- 5-10x faster loading
- Smaller file size
- Better cold start performance
- Only required columns are loaded

### Household Types

- `HH01` - Alleenstaande (Single person)
- `HH02` - Alleenstaande ouder met kind (Single parent with child)
- `HH03` - Paar (Couple)
- `HH04` - Paar met twee kinderen (Couple with two children)

### Visualizations (4 tabs)

1. **Huishoudtypen** - Box plot comparing all household types across municipalities
2. **Inkomensgroepen** - Line chart showing benefit decrease as income increases
3. **(In)formeel** - Stacked bar chart: formal vs informal regulations
4. **Waarden en Inkomensgrenzen** - Scatter plot: benefit value vs weighted income threshold (bubble size = population)

### Layout

- Two-column layout: graphs (2/3) + regulations table (1/3)
- Sidebar contains all filters (income slider, municipality, household type, years at low income, regulation types)
- Selected municipality highlighted in red (#d63f44), others in gray (#9f9f9f)

### Session State

Filter values stored in `st.session_state`:
- `selected_income_pct` (100-200)
- `selected_gemeente` (municipality code)
- `selected_huishouden` (HH01-HH04)
- `selected_referteperiode` (0-5)
- `regelingen_filter` (list of regulation type strings)

## Language

UI text is in Dutch. Error messages and user-facing strings should remain in Dutch.

## Deployment

### Google Cloud Run

The dashboard is deployed to Google Cloud Run at:
**https://dashboard-armoedebeleid-1097840068635.europe-west4.run.app**

**To deploy updates:**

```bash
# 1. Commit changes to git
git add .
git commit -m "Your commit message"
git push origin main

# 2. Deploy to Cloud Run (builds and deploys from source)
"C:/Program Files (x86)/Google/Cloud SDK/google-cloud-sdk/bin/gcloud.cmd" run deploy dashboard-armoedebeleid \
  --source . \
  --region europe-west4 \
  --allow-unauthenticated
```

**Notes:**
- The `--source .` flag builds the Docker image from the Dockerfile in Cloud Build
- Environment variables (EXCEL_URL, KEY_ALL, KEY_BARNEVELD, KEY_DELFT) are configured as secrets in Google Secret Manager
- Deployment takes ~5-10 minutes for the build process
- The service auto-scales from 0 to 10 instances based on traffic

### CSV Export Feature

CSV export buttons are hidden by default. Add `?export=1` to the URL to enable them:
- Normal view: `https://dashboard-armoedebeleid-1097840068635.europe-west4.run.app`
- With exports: `https://dashboard-armoedebeleid-1097840068635.europe-west4.run.app?export=1`

## Project Documentation

Implementation plans and technical documentation should be stored in the `.claude/` folder:
- Deployment plans: `.claude/google-cloud-deployment-plan.md`
- Other technical plans and documentation as needed

This keeps project-specific plans with the codebase rather than in user account folders.

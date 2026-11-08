# Kolkata Weather AI Dashboard

## Folder structure

Keep your trained project files in this structure:

```text
kolkata_weather_dashboard/
├── app.py
├── requirements.txt
├── dataset/
│   ├── kolkata_weather.csv
│   └── processed_weather.csv
├── models/
│   ├── best_weather_model.pkl
│   ├── feature_columns.pkl
│   └── model_comparison.csv
└── graphs/
```

Copy your existing CSV and model artifacts into the matching folders.

## Install

```bash
pip install -r requirements.txt
```

## Run

Open a terminal in this folder and execute:

```bash
streamlit run app.py
```

The dashboard includes Overview, Dataset Explorer, Preprocessing, Exploratory Analysis,
Model Performance and Weather Predictor pages.

The app reads the actual CSV/model artifacts. It does not fabricate model scores.

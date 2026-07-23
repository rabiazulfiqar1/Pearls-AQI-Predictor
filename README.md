# AQI Predictor

A data science project that predicts the Air Quality Index (AQI) for Karachi using historical weather and air quality data from the Open-Meteo APIs.

## Project Overview

The objective of this project is to build an end-to-end AQI prediction pipeline by:

- Collecting historical weather and air quality data
- Cleaning and preprocessing the data
- Engineering predictive features
- Training machine learning models
- Evaluating forecasting performance
- Deploying a prediction system

---

### Weather Variables

- Temperature (2m)
- Relative Humidity
- Wind Speed (10m)
- Mean Sea Level Pressure
- Precipitation
- Cloud Cover

### Air Quality Variables

- PM2.5
- PM10
- Nitrogen Dioxide (NO₂)
- Sulphur Dioxide (SO₂)
- Ozone (O₃)
- Carbon Monoxide (CO)
- US AQI

---

## Project Structure

```
AQI_predictor/
├── config/
├── scripts/
│   └── fetch_data.py
├── preprocessing/
├── models/
├── data/            
│   └── processed/
├── README.md
└── requirements.txt
```

---

## Output

The data collection pipeline generates a merged training dataset:

```
data/processed/karachi_training_data.csv
```

which serves as the input for the machine learning workflow.
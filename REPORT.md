# Pearls AQI Predictor — Final Report

**Author:** Rabia Zulfiqar
**Tag:** v1.0-submission  
**Date:** August 2026

**Live dashboard:** [aqi-predictor-pearls.streamlit.app](https://aqi-predictor-pearls.streamlit.app/)  
**Repository:** [github.com/rabiazulfiqar1/Pearls-AQI-Predictor](https://github.com/rabiazulfiqar1/Pearls-AQI-Predictor)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [System Architecture and Automation](#3-system-architecture-and-automation)
4. [Data Sources](#4-data-sources)
5. [Exploratory Data Analysis](#5-exploratory-data-analysis)
6. [Feature Engineering and Pipeline](#6-feature-engineering-and-pipeline)
7. [Model Training and Selection](#7-model-training-and-selection)
8. [Serving Layer: Dashboard and API](#8-serving-layer-dashboard-and-api)
9. [Results and Observations](#9-results-and-observations)
10. [Future Work](#10-future-work)
11. [Conclusion](#11-conclusion)

---

## 1. Executive Summary

Pearls AQI Predictor is an end-to-end forecasting system for Karachi that turns environmental measurements into multi-horizon AQI predictions. The repository combines data collection, feature engineering, model comparison, inference logic, and a public dashboard into one reproducible workflow.

The current implementation uses a feature-store-backed approach to load the latest row, generate forecasts for 24h, 48h, and 72h, and present the results in a Streamlit-based interface. The project is designed to be practical, transparent, and easy to extend for future deployment and monitoring improvements.

---

## 2. Problem Statement

Karachi’s air quality varies significantly over time and directly affects daily decision-making for residents, commuters, and health-sensitive groups. A real-time AQI reading answers the question “what is the air quality now,” but a forecast answers the more useful question “what can I expect in the next day or two?”

This project addresses that gap by building a forecasting workflow that can generate multi-day AQI predictions from publicly available environmental data and present them through a user-friendly dashboard.

---

## 3. System Architecture and Automation

The system is organized into modular stages:

1. Data ingestion from public APIs
2. Feature engineering and feature-store preparation
3. Model training and evaluation
4. Forecast generation from the latest feature row
5. Dashboard presentation for end users

The main code paths are implemented in the repository folders [app](app), [models](models), [pipelines](pipelines), [utils](utils), and [config](config).

The dashboard uses the inference logic in [models/predict.py](models/predict.py), while the public UI is implemented in [app/dashboard.py](app/dashboard.py).

<img src="architecture.png" alt="System architecture" width="800" />

---

## 4. Data Sources

The project relies on environmental data from Open-Meteo, including air-quality and weather signals. The feature pipeline combines pollutant data such as PM2.5, PM10, NO₂, O₃, SO₂, and CO with meteorological variables such as temperature, humidity, wind, and pressure.

The resulting time-series data is engineered into supervised learning features so the model can learn both short-term persistence and longer-term seasonal structure.

---

## 5. Exploratory Data Analysis

The repository contains several EDA artifacts in [EDA-outputs](EDA-outputs). These images were used to understand both the target behavior and the important variables for forecasting.

### 5.1 Target distribution

![AQI target distribution](EDA-outputs/target_dist.png)

This plot shows the spread of AQI values in the dataset and supports the decision to model the target as a continuous variable rather than a simple categorical alert flag.

### 5.2 Feature correlation

![Feature correlation](EDA-outputs/feature_corr.png)

This figure highlights the strongest relationships between AQI and the engineered variables, helping identify the most informative signals for prediction.

### 5.3 AQI trend over time

![AQI trend](EDA-outputs/aqi_trend.png)

The time-series view confirms that AQI shows temporal structure and persistence, which is a key motivation for using lag-based and rolling features.

### 5.4 SHAP-based importance

![SHAP importance for the 24h horizon](EDA-outputs/shap_importance_24h.png)

The SHAP output helps explain which variables contributed most strongly to the model’s decision-making and supports the feature-selection process used in the project.

---

## 6. Feature Engineering and Pipeline

The project uses a structured feature-engineering workflow to transform raw environmental observations into model-ready rows.

### 6.1 Engineered features

The feature pipeline includes:

- lag-based AQI features
- rolling statistics and short-term change measures
- temporal variables such as month and hour-based structure
- weather and pollutant features that reflect the physical drivers of air quality

These features are designed to capture both persistence and short-term dynamics in the AQI signal.

### 6.2 Feature-store integration

The latest feature rows are loaded from the repository’s feature-store-based workflow, and the prediction pipeline uses those rows as the basis for inference. This keeps the training and serving paths aligned with the same feature definitions.

---

## 7. Model Training and Selection

Several model families were evaluated in the repository to compare predictive behavior. The reported evaluation summary shows that the project compared a naive baseline with linear, tree-based, and gradient-boosted approaches.

### 7.1 Holdout comparison summary

| Model | 24h RMSE | 48h RMSE | 72h RMSE |
|---|---:|---:|---:|
| Naive | 13.16 | 17.89 | 19.89 |
| Ridge | 10.44 | 14.30 | 15.30 |
| Random Forest | 11.32 | 14.53 | 17.81 |
| XGBoost | 10.95 | 14.30 | 15.48 |

These results indicate that the forecasting task is most reliable at shorter horizons and that the linear baseline performed competitively compared with the more complex models on the repository’s evaluation setup.

### 7.2 LSTM experiment results

The repository also contains a dedicated LSTM experiment workflow in [models/lstm.py](models/lstm.py) and the saved results in [results/LSTM_results.md](results/LSTM_results.md). Several architecture configurations were tested:

- baseline_64_32
- wider_128_64
- deeper_64_64_32
- bidirectional_64_32
- regularized_96_48_lowlr

The best-performing LSTM configuration was wider_128_64, which achieved the lowest validation loss and the strongest validation metrics across all horizons. Its reported results were:

| Horizon | RMSE | MAE | R² |
|---|---:|---:|---:|
| 24h | 15.35 | 11.60 | 0.400 |
| 48h | 17.45 | 13.10 | 0.232 |
| 72h | 18.82 | 13.97 | 0.135 |

On the independent holdout set, the same configuration achieved:

| Horizon | RMSE | MAE | R² |
|---|---:|---:|---:|
| 24h | 11.22 | 7.83 | 0.376 |
| 48h | 12.72 | 9.16 | 0.141 |
| 72h | 13.24 | 9.66 | -0.009 |

These results show that the LSTM was competitive at short horizons but that the longer-horizon task remained more difficult.

### 7.3 Model choice

The repository’s modeling workflow is built around comparing several approaches while keeping the feature definitions consistent. That makes it possible to judge whether a simpler model is sufficient or whether a more expressive model is justified.

---

## 8. Serving Layer: Dashboard and API

The dashboard is implemented in [app/dashboard.py](app/dashboard.py), and the prediction logic is implemented in [models/predict.py](models/predict.py).

### 8.1 Dashboard screenshots

![Dashboard screenshot 1](dashboard_ss/dashboard1.png)

![Dashboard screenshot 2](dashboard_ss/dashbaord2.png)

The dashboard presents:

- a current-condition summary
- a three-day prediction table
- a forecast trajectory chart
- AQI category-based visual cues

### 8.2 Prediction workflow

At inference time, the pipeline:

1. loads the latest feature row,
2. selects the appropriate model for each horizon,
3. generates the 24h/48h/72h forecast values,
4. returns the results to the interface for display.

The repository also includes a FastAPI backend in [backend/main.py](backend/main.py) with health and prediction endpoints under [backend](backend). The Streamlit dashboard loads models directly from Hopsworks and caches them on first run, so it does not require a separate backend deployment to function in the local workflow.

---

## 9. Results and Observations

The project produced a complete workflow that is useful for both experimentation and presentation.

### 9.1 What the repository demonstrates

- A working end-to-end AQI prediction pipeline
- Reproducible feature engineering from environmental inputs
- A model-comparison framework with documented metrics
- A public-facing dashboard experience
- Clear integration between training-time logic and serving-time inference

### 9.2 Practical takeaway

The results show that AQI forecasting is feasible with a modest project structure and publicly available data, especially for short-horizon prediction. The repository is therefore a strong proof-of-concept and a solid base for future extensions.

---

## 10. Future Work

Several improvements are natural next steps:

- add stronger uncertainty estimates and interval reporting
- expand the out-of-time evaluation framework
- improve deployment robustness and monitoring
- add richer alerting and user-facing features
- extend the system to more locations or more granular forecasting horizons

---

## 11. Conclusion

Pearls AQI Predictor is a complete and practical AQI forecasting project for Karachi. It brings together data, modelling, inference, and visualization into one cohesive workflow and shows how a public-facing air-quality forecast can be built from repository-based ML components.

The project is now positioned not just as an experimental notebook workflow, but as a usable forecasting application with clear documentation, visuals, and a dashboard interface.

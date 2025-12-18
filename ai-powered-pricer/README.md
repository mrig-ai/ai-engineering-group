# 🥑 AI-Powered Avocado Pricer

An end-to-end machine learning project that predicts avocado prices using historical sales data and gradient-boosted models.  
The project demonstrates data analysis, feature engineering, model training, evaluation, and interactive exploration in Jupyter.


## 📌 Overview

This project builds an **XGBoost regression model** to estimate avocado prices based on:
- historical sales volumes
- regions
- temporal features (time, seasonality)
- product attributes

The core workflow is implemented in a Jupyter Notebook and is designed for **experimentation and interpretability**, including interactive visualizations.


## 📂 Repository Structure

```text
ai-powered-pricer/
├── data/                        # Avocado dataset(s)
├── Avocado-Pricer-meetup.ipynb  # Main analysis, modeling & visualization notebook
├── xgb_all.joblib               # Trained XGBoost model artifact
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation

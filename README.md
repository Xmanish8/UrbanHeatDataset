# 🌡️ Urban Heat Island (UHI) Prediction — Pune, India

An AI/ML-based geospatial system to identify urban heat stress 
hotspots, quantify key drivers of urban heating, and generate 
optimized cooling interventions.

Built for ISRO Hackathon — Problem Statement 1

## 🎯 Objective
- Identify Urban Heat Hotspots
- Analyze Drivers of Urban Heating
- Model Heat Dynamics using ML
- Generate Cooling Scenarios

## 📡 Data Sources
| Dataset | Source | Features |
|---|---|---|
| Land Surface Temperature | Landsat 8 C2 L2 | LST (°C) |
| Spectral Indices | Sentinel-2 | NDVI, NDBI, NDWI |
| Meteorological | ERA5-Land | AirTemp, WindSpeed |
| Terrain | SRTM | Elevation, Slope |
| Population | WorldPop | Population Density |

## 🤖 ML Model
- **Algorithm:** Random Forest Regressor
- **Target:** Land Surface Temperature (LST)
- **R²:** 0.998
- **RMSE:** < 0.1 °C
- **Grid Resolution:** 100m × 100m
- **Total Grid Cells:** 42,895

## 📊 Features Used
- NDVI (Vegetation)
- NDBI (Built-up Areas)
- AirTemp (Atmospheric Temperature)
- Elevation (Terrain)
- Slope (Terrain Morphology)
- Population Density

## 🗺️ Outputs
- Heat Stress Zone Map (5 classes)
- Continuous LST Prediction Map
- Interactive Folium Maps
- SHAP Explainability Analysis
- Cooling Scenario Simulator

## 🚀 How to Run

### 1. Install dependencies
pip install -r requirements.txt

### 2. Add your data
Place UrbanHeatDataset.csv in the data/ folder

### 3. Train model
python train.py

### 4. Run dashboard
streamlit run app.py

## 📁 Project Structure
UHI-Prediction/
├── data/                  # Input CSV from GEE
├── models/                # Saved ML models
├── outputs/               # Charts and plots
├── maps/                  # Interactive HTML maps
├── notebooks/             # Jupyter notebooks
├── train.py               # Training pipeline
├── evalution.py           # Evaluation + SHAP
├── app.py                 # Streamlit dashboard
└── requirements.txt

## 🏆 Hackathon
ISRO Space Applications Centre
Problem Statement 1 — Urban Heat Mitigation

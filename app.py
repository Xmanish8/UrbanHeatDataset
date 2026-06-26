import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import joblib
import os

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="UHI Prediction — Pune",
    page_icon="🌡️",
    layout="wide"
)

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    df = pd.read_csv('data/UrbanHeatDataset.csv')
    df.drop(columns=['.geo', 'system:index'],
            inplace=True, errors='ignore')
    df.dropna(inplace=True)
    return df

@st.cache_resource
def load_model():
    return joblib.load("models/uhi_xgboost_model.pkl")
    
df    = load_data()
model = load_model()

FEATURES = [
    'NDVI',
    'NDBI',
    'NDWI',
    'Albedo',
    'AirTemp',
    'WindSpeed',
    'Elevation',
    'Slope',
    'PopDensity'
]
df['LST_Predicted'] = model.predict(df[FEATURES])

p20 = df['LST_Predicted'].quantile(0.20)
p40 = df['LST_Predicted'].quantile(0.40)
p60 = df['LST_Predicted'].quantile(0.60)
p80 = df['LST_Predicted'].quantile(0.80)

def classify_uhi(lst):
    if lst <= p20:   return 'Cool Zone'
    elif lst <= p40: return 'Mild Zone'
    elif lst <= p60: return 'Moderate Zone'
    elif lst <= p80: return 'Warm Zone'
    else:            return 'Hot Zone (UHI)'

df['UHI_Zone'] = df['LST_Predicted'].apply(classify_uhi)

zone_colors = {
    'Cool Zone'      : '#313695',
    'Mild Zone'      : '#74add1',
    'Moderate Zone'  : '#fee090',
    'Warm Zone'      : '#f46d43',
    'Hot Zone (UHI)' : '#a50026'
}

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/"
    "b/bd/Indian_Space_Research_Organisation_Logo.svg/"
    "200px-Indian_Space_Research_Organisation_Logo.svg.png",
    width=120
)
st.sidebar.title("🌡️ UHI Prediction")
st.sidebar.markdown("**Pune City Core — 2023**")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview",
     "🗺️ Heat Map",
     "📊 Driver Analysis",
     "🌿 Scenario Simulator",
     "🤖 AI Recommendations"]
)

# ============================================================
# PAGE 1 — OVERVIEW
# ============================================================

if page == "🏠 Overview":
    st.title("🌡️ Urban Heat Island Prediction System")
    st.markdown("### Pune City Core | ISRO Hackathon 2025")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Grid Cells", f"{len(df):,}", "100m × 100m")
    col2.metric("Mean LST", f"{df['LST_Predicted'].mean():.1f} °C")
    col3.metric("Max LST", f"{df['LST_Predicted'].max():.1f} °C")
    col4.metric("Model R²", "0.94")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("UHI Zone Distribution")
        zone_counts = df['UHI_Zone'].value_counts()
        ordered = ['Cool Zone', 'Mild Zone', 'Moderate Zone',
                   'Warm Zone', 'Hot Zone (UHI)']
        counts = [zone_counts.get(z, 0) for z in ordered]
        colors = [zone_colors[z] for z in ordered]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(ordered, counts, color=colors, edgecolor='white')
        ax.set_ylabel('Grid Cells')
        ax.set_title('UHI Zone Distribution')
        plt.xticks(rotation=20, ha='right')
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.subheader("LST Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df['LST_Predicted'], bins=50,
                color='tomato', edgecolor='white')
        ax.axvline(df['LST_Predicted'].mean(),
                   color='black', linestyle='--', label='Mean')
        ax.set_xlabel('Predicted LST (°C)')
        ax.set_ylabel('Count')
        ax.set_title('LST Distribution')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("Data Sources")
    st.table(pd.DataFrame({
        'Dataset'  : ['Landsat 8 C2 L2', 'Sentinel-2',
                      'ERA5-Land', 'SRTM', 'WorldPop'],
        'Features' : ['LST (°C)', 'NDVI, NDBI, NDWI',
                      'AirTemp, WindSpeed',
                      'Elevation, Slope', 'Population Density'],
        'Resolution': ['30m', '10m', '9km', '30m', '100m']
    }))

# ============================================================
# PAGE 2 — HEAT MAP
# ============================================================

elif page == "🗺️ Heat Map":
    st.title("🗺️ Urban Heat Map — Pune")
    st.markdown("---")

    map_type = st.radio(
        "Select Map Type",
        ["Heatmap", "UHI Zones", "Continuous LST"],
        horizontal=True
    )

    df_sample = df.sample(n=3000, random_state=42)
    center    = [df['latitude'].mean(), df['longitude'].mean()]

    if map_type == "Heatmap":
        m = folium.Map(location=center,
                       zoom_start=12, tiles='CartoDB positron')
        heat_data = [[r['latitude'], r['longitude'],
                      r['LST_Predicted']]
                     for _, r in df_sample.iterrows()]
        HeatMap(heat_data, min_opacity=0.5,
                radius=15, blur=10).add_to(m)
        st_folium(m, width=1100, height=550)

    elif map_type == "UHI Zones":
        import json
        features = []
        for _, row in df_sample.iterrows():
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [row['longitude'], row['latitude']]
                },
                "properties": {
                    "zone" : row['UHI_Zone'],
                    "lst"  : round(row['LST_Predicted'], 2),
                    "color": zone_colors[row['UHI_Zone']]
                }
            })
        m = folium.Map(location=center,
                       zoom_start=12, tiles='OpenStreetMap')
        folium.GeoJson(
            {"type": "FeatureCollection", "features": features},
            marker=folium.CircleMarker(radius=4),
            style_function=lambda f: {
                'fillColor'  : f['properties']['color'],
                'color'      : f['properties']['color'],
                'fillOpacity': 0.8,
                'weight'     : 0
            },
            tooltip=folium.GeoJsonTooltip(
                fields=['zone', 'lst'],
                aliases=['Zone:', 'LST (°C):'])
        ).add_to(m)
        st_folium(m, width=1100, height=550)

    else:
        fig, ax = plt.subplots(figsize=(12, 8))
        sc = ax.scatter(df['longitude'], df['latitude'],
                        c=df['LST_Predicted'],
                        cmap='RdYlBu_r', s=2, alpha=0.8)
        plt.colorbar(sc, ax=ax, label='LST (°C)')
        ax.set_title('Predicted LST — Pune City Core')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        plt.tight_layout()
        st.pyplot(fig)

# ============================================================
# PAGE 3 — DRIVER ANALYSIS
# ============================================================

elif page == "📊 Driver Analysis":
    st.title("📊 Driver Analysis — What Causes UHI?")
    st.markdown("---")

    st.subheader("SHAP Feature Importance")
    if os.path.exists('shap_importance.png'):
        st.image('shap_importance.png', use_column_width=True)
    else:
        st.warning("Run SHAP analysis first to generate this plot.")

    st.subheader("SHAP Summary — Feature Impact")
    if os.path.exists('shap_summary.png'):
        st.image('shap_summary.png', use_column_width=True)

    st.markdown("---")
    st.subheader("SHAP Driver Ranking")
    if os.path.exists('shap_mean_importance.csv'):
        shap_df = pd.read_csv('shap_mean_importance.csv',
                              header=None,
                              names=['Feature', 'SHAP Value'])
        shap_df = shap_df.sort_values(
            'SHAP Value', ascending=False)
        st.dataframe(shap_df, use_container_width=True)

    st.markdown("---")
    st.subheader("SHAP Dependence Plots")
    col1, col2 = st.columns(2)
    for feat, col in zip(['NDVI', 'NDBI', 'AirTemp', 'Elevation'],
                         [col1, col2, col1, col2]):
        path = f'shap_dependence_{feat}.png'
        if os.path.exists(path):
            col.image(path, caption=f'SHAP — {feat}',
                      use_column_width=True)

    st.markdown("---")
    st.subheader("Feature Importance (Random Forest)")
    importances = model.feature_importances_
    feat_df = pd.DataFrame({
        'Feature'    : FEATURES,
        'Importance' : importances
    }).sort_values('Importance', ascending=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(feat_df['Feature'], feat_df['Importance'],
            color='steelblue', edgecolor='white')
    ax.set_xlabel('Importance')
    ax.set_title('Random Forest Feature Importance')
    plt.tight_layout()
    st.pyplot(fig)

# ============================================================
# PAGE 4 — SCENARIO SIMULATOR
# ============================================================

elif page == "🌿 Scenario Simulator":
    st.title("🌿 Cooling Scenario Simulator")
    st.markdown("---")

    st.markdown("### Adjust Intervention Intensity")

    col1, col2 = st.columns(2)
    with col1:
        ndvi_increase = st.slider(
            "🌳 Urban Greening — NDVI Increase",
            0.0, 0.30, 0.15, 0.01)
        ndbi_decrease = st.slider(
            "🏠 Cool Roofs — NDBI Decrease",
            0.0, 0.20, 0.10, 0.01)
    with col2:
        coverage = st.slider(
            "📍 Target Coverage (% hottest cells)",
            10, 50, 30, 5)
        st.markdown("---")
        st.markdown(f"**Targeting:** top {coverage}% hottest cells")
        st.markdown(f"**Cells affected:** "
                    f"{int(len(df) * coverage / 100):,}")

    if st.button("▶ Run Simulation", type="primary"):
        X_sim  = df[FEATURES].copy()
        cutoff = df['LST_Predicted'].quantile(1 - coverage/100)
        mask   = df['LST_Predicted'] >= cutoff

        baseline = df['LST_Predicted'].mean()

        # Greening
        X_g = X_sim.copy()
        X_g.loc[mask, 'NDVI'] = (
            X_g.loc[mask, 'NDVI'] + ndvi_increase).clip(upper=0.9)
        lst_g = model.predict(X_g).mean()

        # Cool roofs
        X_r = X_sim.copy()
        X_r.loc[mask, 'NDBI'] = (
            X_r.loc[mask, 'NDBI'] - ndbi_decrease).clip(lower=-1.0)
        lst_r = model.predict(X_r).mean()

        # Combined
        X_c = X_sim.copy()
        X_c.loc[mask, 'NDVI'] = (
            X_c.loc[mask, 'NDVI'] + ndvi_increase).clip(upper=0.9)
        X_c.loc[mask, 'NDBI'] = (
            X_c.loc[mask, 'NDBI'] - ndbi_decrease).clip(lower=-1.0)
        lst_c = model.predict(X_c).mean()

        st.markdown("---")
        st.subheader("Simulation Results")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Baseline LST",
                    f"{baseline:.2f} °C")
        col2.metric("After Greening",
                    f"{lst_g:.2f} °C",
                    f"{lst_g - baseline:.3f} °C")
        col3.metric("After Cool Roofs",
                    f"{lst_r:.2f} °C",
                    f"{lst_r - baseline:.3f} °C")
        col4.metric("Combined",
                    f"{lst_c:.2f} °C",
                    f"{lst_c - baseline:.3f} °C")

        # Bar chart
        results = pd.DataFrame({
            'Scenario': ['Baseline', 'Greening',
                         'Cool Roofs', 'Combined'],
            'LST'     : [baseline, lst_g, lst_r, lst_c]
        })
        fig, ax = plt.subplots(figsize=(8, 4))
        colors  = ['#d73027', '#1a9850', '#4575b4', '#313695']
        ax.bar(results['Scenario'], results['LST'],
               color=colors, edgecolor='white')
        ax.axhline(y=baseline, color='red',
                   linestyle='--', lw=2)
        ax.set_ylabel('Mean LST (°C)')
        ax.set_title('Cooling Scenario Comparison')
        plt.tight_layout()
        st.pyplot(fig)

# ============================================================
# PAGE 5 — AI RECOMMENDATIONS
# ============================================================

elif page == "🤖 AI Recommendations":
    st.title("🤖 AI-Based Intervention Recommendations")
    st.markdown("---")

    hotspots = df[df['UHI_Zone'] == 'Hot Zone (UHI)']
    st.metric("Total Hotspot Cells", f"{len(hotspots):,}")
    st.metric("Hotspot Area (approx)",
              f"{len(hotspots) * 0.01:.1f} km²")

    st.markdown("---")
    st.subheader("Top 10 Hottest Locations")
    top10 = df.nlargest(10, 'LST_Predicted')[
        ['latitude', 'longitude', 'LST_Predicted',
         'NDVI', 'NDBI', 'UHI_Zone']
    ].reset_index(drop=True)
    top10.index += 1
    st.dataframe(top10, use_container_width=True)

    st.markdown("---")
    st.subheader("Recommended Interventions")
    st.markdown("""
| Intervention | Target Zone | Expected Reduction | Priority |
|---|---|---|---|
| 🌳 Urban Greening | Hot Zone (UHI) | −0.635 °C | 🔴 High |
| 🏠 Cool Roofs | Hot + Warm Zone | −0.479 °C | 🔴 High |
| 💧 Water Bodies | Hot Zone (UHI) | −0.478 °C | 🟡 Medium |
| 🌿 Combined Strategy | All Hot Zones | −0.746 °C | 🔴 High |
    """)

    st.markdown("---")
    st.subheader("Intervention Priority Map")
    if os.path.exists('outputs/intervention_priority_map.png'):
        st.image('outputs/intervention_priority_map.png',
                 use_column_width=True)

    st.markdown("---")
    st.subheader("Before vs After Intervention")
    if os.path.exists('outputs/intervention_map.png'):
        st.image('outputs/intervention_map.png',
                 use_column_width=True)

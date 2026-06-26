import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from sklearn.ensemble import RandomForestRegressor
import folium
from folium.plugins import HeatMap
import joblib

# ============================================================
# 1. LOAD NEW CSV
# ============================================================

df = pd.read_csv('UrbanHeatDataset.csv')
df.drop(columns=['.geo', 'system:index'], inplace=True, errors='ignore')
df.dropna(inplace=True)

print(f"Rows loaded: {len(df)}")

FEATURES = ['NDVI', 'NDBI', 'AirTemp', 'Elevation', 'Slope', 'PopDensity']
TARGET   = 'LST'

X = df[FEATURES]
y = df[TARGET]

# ============================================================
# 2. TRAIN & PREDICT
# ============================================================

rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
rf.fit(X, y)
joblib.dump(rf, 'uhi_rf_model.pkl')

df['LST_Predicted'] = rf.predict(X)

# ============================================================
# 3. UHI ZONES
# ============================================================

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

print("Zone distribution:")
print(df['UHI_Zone'].value_counts())

# ============================================================
# 4. STATIC MATPLOTLIB MAP
# ============================================================

plt.figure(figsize=(12, 10))
for zone, color in zone_colors.items():
    subset = df[df['UHI_Zone'] == zone]
    plt.scatter(subset['longitude'], subset['latitude'],
                c=color, label=zone, s=2, alpha=0.6)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Urban Heat Island Zones — Pune (2023)')
plt.legend(loc='lower right', markerscale=5)
plt.tight_layout()
plt.savefig('uhi_zone_map.png', dpi=200, bbox_inches='tight')
plt.show()
print("Saved → uhi_zone_map.png")

# ============================================================
# 5. HEATMAP (sampled — folium can't handle 42k points well)
# ============================================================

# Sample 5000 points for folium performance
df_sample = df.sample(n=5000, random_state=42)

center_lat = df['latitude'].mean()
center_lon = df['longitude'].mean()

m1 = folium.Map(location=[center_lat, center_lon],
                zoom_start=12, tiles='CartoDB positron')

heat_data = [[r['latitude'], r['longitude'], r['LST_Predicted']]
             for _, r in df_sample.iterrows()]

HeatMap(heat_data, min_opacity=0.5,
        radius=15, blur=10).add_to(m1)

m1.save('uhi_heatmap.html')
print("Saved → uhi_heatmap.html")

# ============================================================
# 6. INTERACTIVE ZONE MAP (sampled)
# ============================================================

m2 = folium.Map(location=[center_lat, center_lon],
                zoom_start=12, tiles='OpenStreetMap')

for _, row in df_sample.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=4,
        color=None,
        fill=True,
        fill_color=zone_colors[row['UHI_Zone']],
        fill_opacity=0.75,
        popup=folium.Popup(
            f"<b>Zone:</b> {row['UHI_Zone']}<br>"
            f"<b>LST:</b> {row['LST_Predicted']:.2f} °C<br>"
            f"<b>NDBI:</b> {row['NDBI']:.3f}<br>"
            f"<b>NDVI:</b> {row['NDVI']:.3f}",
            max_width=200)
    ).add_to(m2)

legend_html = """
<div style="position:fixed; bottom:30px; left:30px; z-index:1000;
     background:white; padding:14px; border-radius:8px;
     border:2px solid #aaa; font-size:13px; line-height:24px;">
  <b>UHI Zone</b><br>
  <span style="color:#313695;font-size:18px;">&#9632;</span> Cool Zone<br>
  <span style="color:#74add1;font-size:18px;">&#9632;</span> Mild Zone<br>
  <span style="color:#fee090;font-size:18px;">&#9632;</span> Moderate Zone<br>
  <span style="color:#f46d43;font-size:18px;">&#9632;</span> Warm Zone<br>
  <span style="color:#a50026;font-size:18px;">&#9632;</span> Hot Zone (UHI)<br>
</div>
"""
m2.get_root().html.add_child(folium.Element(legend_html))
m2.save('uhi_zone_map.html')
print("Saved → uhi_zone_map.html")

# ============================================================
# 7. SATELLITE MAP
# ============================================================

m3 = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=12,
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attr='Esri Satellite')

lst_min = df['LST_Predicted'].min()
lst_max = df['LST_Predicted'].max()
cmap    = cm.get_cmap('RdYlBu_r')

for _, row in df_sample.iterrows():
    norm_val  = (row['LST_Predicted'] - lst_min) / (lst_max - lst_min)
    hex_color = mcolors.to_hex(cmap(norm_val))
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=4,
        color=None,
        fill=True,
        fill_color=hex_color,
        fill_opacity=0.8,
        popup=folium.Popup(
            f"<b>LST:</b> {row['LST_Predicted']:.2f} °C<br>"
            f"<b>Zone:</b> {row['UHI_Zone']}",
            max_width=150)
    ).add_to(m3)

m3.save('uhi_satellite_map.html')
print("Saved → uhi_satellite_map.html")

print("\n✅ All maps saved. Open in browser:")
print("   → uhi_heatmap.html")
print("   → uhi_zone_map.html")
print("   → uhi_satellite_map.html")
print("   → uhi_zone_map.png  (static)")
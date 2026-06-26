# ============================================================
# UHI Prediction — ML Training Pipeline
# Author: Manish Marathe
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv('UrbanHeatDataset.csv')
print(f"Shape: {df.shape}")
print(df.head())
print(df.describe())

# ============================================================
# 2. PREPROCESSING
# ============================================================

# Drop rows with any null values (masked pixels)
df.dropna(inplace=True)
df.drop(columns=['.geo', 'system:index'], inplace=True, errors='ignore')
print(f"Shape after dropping nulls: {df.shape}")

# Define features and target
FEATURES = ['NDVI', 'NDBI', 'AirTemp', 'Elevation', 'Slope']
TARGET   = 'LST'

X = df[FEATURES]
y = df[TARGET]

# ============================================================
# 3. EXPLORATORY DATA ANALYSIS
# ============================================================

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df[FEATURES + [TARGET]].corr(),
            annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=150)
plt.show()

# LST distribution
plt.figure(figsize=(8, 4))
sns.histplot(y, bins=50, kde=True, color='tomato')
plt.title('LST Distribution (°C)')
plt.xlabel('LST (°C)')
plt.savefig('lst_distribution.png', dpi=150)
plt.show()

# ============================================================
# 4. TRAIN / TEST SPLIT + SCALING
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler  = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")

# ============================================================
# 5. TRAIN MODELS
# ============================================================

models = {
    'Random Forest'       : RandomForestRegressor(
                              n_estimators=200,
                              max_depth=None,
                              min_samples_split=5,
                              random_state=42,
                              n_jobs=-1),
    'Gradient Boosting'   : GradientBoostingRegressor(
                              n_estimators=200,
                              learning_rate=0.05,
                              max_depth=5,
                              random_state=42),
    'XGBoost'             : xgb.XGBRegressor(
                              n_estimators=200,
                              learning_rate=0.05,
                              max_depth=6,
                              subsample=0.8,
                              colsample_bytree=0.8,
                              random_state=42,
                              verbosity=0)
}

results = {}

for name, model in models.items():
    # Tree models don't need scaling; use raw splits
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    results[name] = {'RMSE': rmse, 'MAE': mae, 'R2': r2}
    print(f"\n{name}")
    print(f"  RMSE : {rmse:.3f} °C")
    print(f"  MAE  : {mae:.3f} °C")
    print(f"  R²   : {r2:.4f}")

# ============================================================
# 6. CROSS-VALIDATION ON BEST MODEL (XGBoost)
# ============================================================

cv_scores = cross_val_score(
    models['XGBoost'], X, y,
    cv=5, scoring='r2', n_jobs=-1
)
print(f"\nXGBoost 5-Fold CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ============================================================
# 7. FEATURE IMPORTANCE
# ============================================================

best_model = models['XGBoost']
importances = best_model.feature_importances_

feat_df = pd.DataFrame({
    'Feature'   : FEATURES,
    'Importance': importances
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(data=feat_df, x='Importance', y='Feature', palette='viridis')
plt.title('XGBoost Feature Importance — UHI Prediction')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.show()

print("\nFeature Importance Ranking:")
print(feat_df.to_string(index=False))

# ============================================================
# 8. PREDICTED vs ACTUAL PLOT
# ============================================================

y_pred_best = best_model.predict(X_test)

plt.figure(figsize=(7, 7))
plt.scatter(y_test, y_pred_best, alpha=0.3, s=10, color='steelblue')
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual LST (°C)')
plt.ylabel('Predicted LST (°C)')
plt.title('XGBoost — Predicted vs Actual LST')
plt.tight_layout()
plt.savefig('predicted_vs_actual.png', dpi=150)
plt.show()

# ============================================================
# 9. SAVE MODEL AND SCALER
# ============================================================

import joblib

joblib.dump(best_model, 'uhi_xgboost_model.pkl')
joblib.dump(scaler,     'uhi_scaler.pkl')
print("\nModel saved → uhi_xgboost_model.pkl")
print("Scaler saved → uhi_scaler.pkl")

# ============================================================
# 10. RESULTS SUMMARY TABLE
# ============================================================

results_df = pd.DataFrame(results).T
print("\n===== MODEL COMPARISON =====")
print(results_df.to_string())
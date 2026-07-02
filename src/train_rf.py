"""
Tire Wear Prediction from CAN Bus Data — Random Forest + Feature Engineering
=============================================================================
Predicts remaining tire tread depth (mm) from CAN bus signals, enabling
predictive maintenance alerts before safety limits are reached.
Author: Bharath Kanaiah Parthiban
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import os, json

np.random.seed(333)
sns.set_theme(style="whitegrid", font_scale=1.05)
PLOT_DIR = os.path.join(os.path.dirname(__file__), '..', 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

# ── Synthetic CAN bus dataset (tire life simulation) ──
N = 2000
rng = np.random.default_rng(333)

total_km        = rng.uniform(0, 60000, N)        # odometer km
avg_speed_kmh   = rng.uniform(30, 130, N)         # average speed
aggressive_braking_cnt = rng.integers(50, 2000, N) # ABS events / 1000 km
cornering_g_avg = rng.uniform(0.05, 0.45, N)      # avg lateral g
ambient_temp    = rng.uniform(-15, 45, N)         # °C
tire_pressure_fl= rng.uniform(1.8, 2.8, N)        # bar
tire_pressure_fr= rng.uniform(1.8, 2.8, N)        # bar
vehicle_mass    = rng.uniform(1200, 2500, N)       # kg
front_axle_load = rng.uniform(0.45, 0.65, N)      # % of total mass
avg_accel       = rng.uniform(0.5, 3.5, N)        # m/s²

INITIAL_TREAD = 8.0  # mm
# Physics-informed wear model
wear = (
    total_km * (
        0.00008
        + 0.00003 * (avg_speed_kmh / 100)
        + 0.000015 * aggressive_braking_cnt / 500
        + 0.0001 * cornering_g_avg
        + 0.000005 * (vehicle_mass / 1500)
        - 0.000005 * (tire_pressure_fl - 2.2) * 2
    )
    + rng.normal(0, 0.08, N)
)
tread_depth = np.clip(INITIAL_TREAD - wear, 1.5, INITIAL_TREAD)

X = np.column_stack([total_km, avg_speed_kmh, aggressive_braking_cnt, cornering_g_avg,
                     ambient_temp, tire_pressure_fl, tire_pressure_fr,
                     vehicle_mass, front_axle_load, avg_accel])
feature_names = ['Total km', 'Avg Speed (km/h)', 'Aggressive Braking Events',
                 'Cornering G (avg)', 'Ambient Temp (°C)', 'Tire Pressure FL (bar)',
                 'Tire Pressure FR (bar)', 'Vehicle Mass (kg)',
                 'Front Axle Load Ratio', 'Avg Acceleration (m/s²)']
y = tread_depth

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestRegressor(n_estimators=300, max_depth=None, min_samples_leaf=2,
                            random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

r2   = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
cv   = cross_val_score(rf, X, y, cv=5, scoring='r2')
print(f"Test R²={r2:.4f}  RMSE={rmse:.4f}mm  MAE={mae:.4f}mm")
print(f"CV R²: {cv.mean():.4f} ± {cv.std():.4f}")
json.dump({"R2": round(r2,4), "RMSE_mm": round(rmse,4), "CV_R2": round(cv.mean(),4)},
          open(os.path.join(PLOT_DIR,'..','metrics.json'),'w'), indent=2)

# Fig 1 — Predicted vs Actual + Residuals
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Tire Wear Prediction — Random Forest Performance', fontsize=14, fontweight='bold')
ax = axes[0]
ax.scatter(y_test, y_pred, alpha=0.4, s=18, color='#E65100', edgecolors='none')
lims = [y_test.min()-0.1, y_test.max()+0.1]
ax.plot(lims, lims, 'r--', lw=2)
ax.axhline(1.6, color='red', ls=':', lw=1.5, label='Legal limit (1.6mm)')
ax.set_xlabel('Actual Tread Depth (mm)', fontsize=11); ax.set_ylabel('Predicted Tread (mm)', fontsize=11)
ax.set_title(f'Predicted vs Actual  |  R² = {r2:.4f}', fontsize=11)
ax.legend(); ax.set_xlim(lims); ax.set_ylim(lims)
ax = axes[1]
residuals = y_test - y_pred
ax.hist(residuals, bins=40, color='#E65100', alpha=0.85, edgecolor='white')
ax.axvline(0, color='black', lw=2, ls='--')
ax.axvline(residuals.mean(), color='blue', lw=1.5, label=f'Mean={residuals.mean():.3f}mm')
ax.set_xlabel('Residual (mm)', fontsize=11); ax.set_ylabel('Count', fontsize=11)
ax.set_title(f'Residual Distribution  |  RMSE={rmse:.3f} mm', fontsize=11); ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR,'fig1_rf_performance.png'), dpi=150, bbox_inches='tight')
plt.close(); print("Saved fig1")

# Fig 2 — Feature Importance
fi = rf.feature_importances_; si = fi.argsort()[::-1]
fig, ax = plt.subplots(figsize=(10, 5))
ax.barh([feature_names[i] for i in si[::-1]], fi[si[::-1]],
        color=['#BF360C' if fi[i]==fi.max() else '#FF8A65' for i in si[::-1]], edgecolor='white')
ax.set_xlabel('Feature Importance (Gini)', fontsize=11)
ax.set_title('Feature Importance — Tire Wear Prediction', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR,'fig2_feature_importance.png'), dpi=150, bbox_inches='tight')
plt.close(); print("Saved fig2")

# Fig 3 — Wear vs Mileage scatter (colored by cornering intensity)
fig, ax = plt.subplots(figsize=(9, 6))
sc = ax.scatter(total_km, tread_depth, c=cornering_g_avg, cmap='hot_r', alpha=0.4, s=12)
cbar = plt.colorbar(sc, ax=ax); cbar.set_label('Avg Cornering G', fontsize=10)
ax.axhline(1.6, color='red', ls='--', lw=2, label='Legal limit (1.6 mm)')
ax.axhline(3.0, color='orange', ls='--', lw=1.5, label='Recommended replace (3 mm)')
ax.set_xlabel('Total Mileage (km)', fontsize=11); ax.set_ylabel('Tread Depth (mm)', fontsize=11)
ax.set_title('Tire Tread Depth vs Mileage — Cornering Intensity', fontsize=13, fontweight='bold')
ax.legend(); plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR,'fig3_wear_vs_mileage.png'), dpi=150, bbox_inches='tight')
plt.close(); print("Saved fig3")

# Fig 4 — Maintenance alert threshold analysis
tread_bins = np.linspace(1.5, 8.0, 100)
pct_critical = [(tread_depth < t).mean()*100 for t in tread_bins]
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(tread_bins, pct_critical, color='#D32F2F', lw=2.5)
ax.fill_between(tread_bins, 0, pct_critical, alpha=0.15, color='#D32F2F')
ax.axvline(1.6, color='black', ls=':', lw=2, label='Legal limit (1.6 mm)')
ax.axvline(3.0, color='orange', ls='--', lw=1.5, label='Early warning (3.0 mm)')
ax.set_xlabel('Tread Depth Threshold (mm)', fontsize=11)
ax.set_ylabel('% Fleet Below Threshold', fontsize=11)
ax.set_title('Predictive Maintenance Alert Analysis', fontsize=13, fontweight='bold')
ax.legend(); plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR,'fig4_maintenance_analysis.png'), dpi=150, bbox_inches='tight')
plt.close(); print("Saved fig4 — Done!")

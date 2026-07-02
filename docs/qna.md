# Interview Q&A — Tire Wear Prediction from CAN Bus Data — Random Forest
## Random Forest Regressor (300 trees) | Predictive Maintenance & Fleet Management

### Q1: Why Random Forest for tire wear prediction?

Random Forest is robust to outliers, handles feature interactions naturally (e.g., high speed × low pressure), requires minimal preprocessing, and is highly interpretable via feature importance. For tabular CAN data where features are known physical quantities, RF consistently outperforms linear models and competes with gradient boosting. The ensemble structure also provides variance estimates useful for uncertainty-aware maintenance alerts.

---

### Q2: How do you collect training data for this in practice?

Real training data is collected via: (1) Instrumented test vehicles with both CAN logging and manual tread gauge measurements at regular intervals (every 5,000 km), (2) Dealership inspection records linked to vehicle OBD history (if OEM has telemetry), (3) Fleet operator data (trucks, taxis, rental cars) with high measurement frequency. Continental, Bridgestone, and Michelin all have internal programs for connected tire monitoring.

---

### Q3: What is TPMS and how does it relate?

Tire Pressure Monitoring System (TPMS) is mandatory in all new EU and US vehicles since 2014 (EU Regulation 661/2009). It monitors inflation pressure using in-wheel sensors (direct TPMS) or wheel speed differences (indirect TPMS). A tread depth model could be embedded in the TPMS ECU at negligible cost, extending its value beyond pressure monitoring.

---

### Q4: What are the key physical drivers of tire wear?

The main drivers are: (1) Total distance — primary wear driver; (2) Speed — higher speed = higher temperature = faster wear; (3) Braking intensity — longitudinal slip generates heat and wear; (4) Cornering force — lateral grip wears the contact patch edge; (5) Inflation pressure — under-inflation concentrates load on shoulders; (6) Wheel alignment (camber, toe) — misalignment creates uneven wear patterns.

---

### Q5: How would you handle the variability between tire models/brands?

A production model would include tire SKU as a categorical feature (one-hot encoded), or train separate models per tire family. Transfer learning approaches fine-tune a base model on specific tire data. The challenge is that wear rate varies significantly between, e.g., a performance summer tire and an all-season touring tire at the same operating conditions.

---


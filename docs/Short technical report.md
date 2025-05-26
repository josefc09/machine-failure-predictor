**SHORT TECHNICAL REPORT: PREDICTIVE MAINTENANCE SYSTEM V1.0**

**Version:** 1.0 **Date:** May 25, 2025 **Author:** José Alberto Fuentes Carbajal **Status:** Final

---

**1\. INTRODUCTION / EXECUTIVE SUMMARY**

This report summarizes the development phase of the Predictive Maintenance System. The project aimed to predict machine failures within a 24-hour window using telemetry and error data. We successfully developed and deployed a solution leveraging an XGBoost model served via a FastAPI-based API. This report outlines the key results, decisions made, lessons learned, and proposed next steps.

---

**2\. KEY RESULTS OBTAINED**

* **Model Performance:**  
  * Successfully trained an XGBoost classifier to predict failures 24 hours in advance.  
  * Initial offline evaluation shows promising results, particularly in **Recall** for the 'failure' class, which is critical for minimizing missed failures. Precision and F1-Score indicate a viable model, though further tuning is expected.  
  * Generated evaluation artifacts, including classification reports, confusion matrices, and feature importance plots, providing insights into model behavior.  
* **Feature Engineering:**  
  * Implemented a robust data preprocessing pipeline that generates time-series features (24-hour rolling means, std deviations, and error counts). These features proved to be significant predictors according to the feature importance analysis.  
* **API Service:**  
  * Deployed a functional FastAPI service capable of receiving machine data and returning failure risk predictions in near real-time.  
  * The API includes data validation using Pydantic schemas and provides interactive documentation via Swagger UI.  
* **Batch Prediction:**  
  * Developed a standalone script (`predict.py`) enabling offline, batch predictions on new datasets.

---

**3\. KEY DECISIONS & TRADE-OFFS**

* **Model Choice (XGBoost):**  
  * **Decision:** Selected XGBoost for its proven high performance on tabular data and speed.  
  * **Trade-off:** Sacrificed some model interpretability compared to simpler models like Logistic Regression but gained significant predictive power.  
* **Deployment Framework (FastAPI):**  
  * **Decision:** Choose FastAPI for its high performance, async capabilities, and Pydantic integration.  
  * **Trade-off:** While excellent for APIs, it required focusing on API-centric deployment rather than a simpler batch-only system initially.  
* **Imbalance Handling (`scale_pos_weight`):**  
  * **Decision:** Used XGBoost's built-in `scale_pos_weight` parameter for simplicity and effectiveness.  
  * **Trade-off:** Avoided the complexity and potential artifacts of data-level methods like SMOTE, though those might offer different performance profiles.  
* **Feature Window (24h):**  
  * **Decision:** Focused on a 24-hour rolling window to support preventive maintenance by capturing recent signals that may indicate upcoming failures.  
  * **Trade-off:** This provides relevant short-term context but may miss longer-term degradation patterns that could be captured with larger windows or different feature types.  
* **Technology Stack:**  
  * **Decision:** Utilized Pandas, Scikit-learn, XGBoost, FastAPI, and Joblib for a unified Python-based workflow.  
  * **Trade-off:** Enabled rapid development and ease of integration, but limited the ability to leverage specialized tools from other ecosystems.

---

**4\. LESSONS LEARNED**

* **Data is Key:** The quality and representativeness of the input data are paramount. **Rare Events are Hard:** Predicting failures (a rare event) requires careful handling of class imbalance and focusing on appropriate metrics (like Recall) beyond just Accuracy.  
* **Feature Engineering Matters:** Simple rolling statistics provided good predictive power, but a deeper domain understanding could unlock even better features.  
* **MLOps is Essential:** While building the model was successful, moving to production highlights the need for robust MLOps practices: versioning, monitoring, automated retraining, and CI/CD.  
* **FastAPI is Fast (to Develop & Run):** The framework significantly accelerated API development and provided good performance.  
* **Configuration Management:** Using `config.yaml` early on helped manage paths and parameters effectively.
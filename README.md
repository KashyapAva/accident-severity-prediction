# 🚦 Context-Aware Accident Severity Prediction

This project aims to build predictive models for accident severity using the **US Accidents** dataset, factoring in **weather**, **seasonality**, and **geographic hotspots**. By segmenting data contextually and training specialized models, we improve the accuracy of severity classification — aiding proactive traffic safety planning.

## 🧭 Project Goals
- Predict the severity of traffic accidents using environmental, spatial, and temporal features.
- Address data imbalance and contextual heterogeneity via tailored modeling pipelines.
- Develop separate models for **winter** vs. **non-winter** seasons for finer accuracy.

## 📚 Dataset
- Source: [US Accidents (2016–2023) by Sobhan Moosavi](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)
- Size: 2.8M+ records
- Variables: Weather, location, time, road conditions, etc.

## 🧪 Methodology

### 🔹 Data Cleaning
- Filled missing values using location-based mapping (e.g., timezone by state).
- Dropped columns with >50% missing data or low correlation with target.
- Forward and backward filling for temporal consistency.

### 🔹 Feature Engineering
- Created `Is_Holiday`, `Is_Rush_Hour`, and `Accident_Duration`.
- Derived temporal features from timestamps (month, hour, weekday).
- Converted boolean strings to binary types and used label encoding.

### 🔹 Hotspot Detection (Geo-Spatial)
- Created spatial indices using longitude-latitude.
- Computed proximity-based accident clustering to generate a `Hotspot_Level` feature.

### 🔹 Contextual Splits
- Divided dataset into **Winter** and **Rest-of-Year**.
- Applied undersampling using `ClusterCentroids` to balance classes.

## ⚙️ Models Trained

| Category | Algorithms Explored |
|----------|---------------------|
| Baseline | KNN, Decision Tree  |
| Ensemble | Random Forest, XGBoost, AdaBoost, Bagged Trees |
| Deep Learning | Feed-Forward Neural Network |

### 🔍 Evaluation Metric: **Weighted Severity Error (WSE)**  
Custom metric to penalize misclassification more heavily for high-severity accidents.

---

## 📊 Results

### Winter Dataset

| Model                      | WSE   |
|---------------------------|--------|
| K-Nearest Neighbors       | 171.2  |
| Decision Tree             | 17.4   |
| Random Forest             | 7.4    |
| XGBoost                   | 12.6   |
| AdaBoost                  | 13.2   |
| Bagged Trees              | 12.4   |
| **Feed-Forward Neural Net** | **2.82** |

### Rest of Seasons Dataset

| Model                      | WSE   |
|---------------------------|--------|
| K-Nearest Neighbors       | 246.06 |
| Decision Tree             | 21.77  |
| Random Forest             | 17.64  |
| XGBoost                   | 19.66  |
| AdaBoost                  | 16.83  |
| Bagged Trees              | 18.36  |
| **Feed-Forward Neural Net** | **1.70** |

---

## 🧠 Key Insights

- **Feed-forward neural networks** significantly outperformed ensemble and tree models.
- **Seasonal modeling** improved performance by isolating weather effects.
- **Hotspot-based geospatial features** contributed to accuracy by modeling local density.

## 🛠 Tools Used
- Python (Pandas, Scikit-learn, TensorFlow, XGBoost)
- GeoPandas for spatial indexing
- Matplotlib & Seaborn for visualization

## 📁 Repository Structure
- `Final_Project_Report.pdf`: Full project report with plots, results, and model tuning
- `Data_Mining_EDA`: Python scripts for data cleaning, modeling, and geospatial processing

## 👥 Team
- Kashyap Ava  
- Sashwath Krishnamoorty Giribabu

---

*Completed as final project for CS412: Data Mining course at UIUC.*

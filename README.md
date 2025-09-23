# Airbnb Success Predictor  

A machine learning project that helps real estate investors identify **profitable Airbnb properties** by predicting nightly price, occupancy likelihood, and success drivers.  
This work was completed as a three-part academic assignment focusing on **data science pipelines, unsupervised learning, and supervised learning**.  

---

## Project Overview  
Investing in short-term rentals can be risky. This project builds a **data-driven decision support tool** for investors considering an Airbnb property.  
The system leverages **Airbnb listing data**, **U.S. Census data**, and **Walk Score data** to predict:  

- **Nightly Price** (Regression)  
- **Occupancy Demand** (Classification)  
- **Success Drivers** (Feature Importance Analysis)  
- **Golden Cluster Properties** (Clustering + Classification)  

---


---

## 📊 Assignments Breakdown  

### 1️⃣ Assignment 1 – Project Plan & Research  
- Defined **business purpose**: help investors reduce risk by predicting profitability.  
- Data sources:  
  - [Inside Airbnb](https://insideairbnb.com/get-the-data/) – listing data (price, amenities, reviews, etc.)  
  - [U.S. Census Bureau](https://data.census.gov/) – socioeconomic context  
  - [Walk Score](https://www.walkscore.com/professional/walk-score-data.php) – neighborhood walkability  
- Literature review on: **dynamic pricing**, **guest experience factors**, and **local regulations**.  

---

### 2️⃣ Assignment 2 – Unsupervised Learning (Clustering)  
- Applied **K-Means & Fractal Clustering** to segment ~27,000 listings.  
- Defined a **Golden Cluster** of properties balancing profitability and guest satisfaction.  
- Outcome: Created a binary target `is_golden_cluster` for supervised learning.  

---

### 3️⃣ Assignment 3 – Supervised Learning (Classification)  
- Amalgamated Airbnb data with Census & Walk Score data using **reverse geocoding** and **spatial joins**.  
- Ran multiple models to predict `is_golden_cluster`:  
  - **Random Forest** → F1: **0.9798**, AUC: **0.9952**  
  - **XGBoost** → F1: **0.9738**, AUC: **0.9992**  
  - **KNN** → F1: 0.8855, AUC: 0.9741  
  - **SVM** → F1: 0.7585, AUC: 0.9645  
- Insight: **Ensemble models** benefited most from enriched datasets.  

---

## 📈 Key Results  
- Identified **hidden gem neighborhoods** outside obvious tourist centers.  
- Built a **Success Score (0–100)** combining price, demand probability, and review scores.  
- Demonstrated that **data enrichment + ensemble models** give the most accurate predictions.  

---

## How to Run  
1. Clone this repo:  
   ```bash
   git clone https://github.com/<your-username>/Airbnb-Success-Predictor.git
   cd Airbnb-Success-Predictor


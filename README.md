#  Airbnb Success Predictor

A comprehensive machine learning project designed to help real estate investors identify profitable short-term rental opportunities by predicting Airbnb listing success using data-driven decision support.

**Project Type:** Academic ML/Data Science Pipeline (3-part assignment)
**Completed By:** [@ananya101001](https://github.com/ananya101001)

[IEEE paper](!Predicting Airbnb Success_ A Data-Driven Investment Framework for the Los Angeles Market.pdf)

---

##  Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Project Structure](#project-structure)
- [Data Sources](#data-sources)
- [Methodology](#methodology)
- [Key Results](#key-results)
- [Technologies](#technologies)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Key Findings](#key-findings)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)

---

##  Overview

This repository contains an academic project that develops a data-driven framework for predicting the success of Airbnb listings. The system leverages Airbnb listing data, U.S. Census data, and Walk Score data to predict nightly price, occupancy likelihood, and success drivers for short-term rental properties.

**Why This Matters:**
- Short-term rental investing is risky without proper analysis
- Market conditions vary significantly by location and property type
- Data-driven insights can help investors make informed decisions
- Success prediction combines multiple factors: price, demand, location, reviews

---

##  Problem Statement

Real estate investors face critical challenges when considering Airbnb investments:

1. **Pricing Uncertainty:** What nightly rate will maximize revenue without losing bookings?
2. **Demand Prediction:** Will the property attract enough guests to be profitable?
3. **Location Impact:** How do socioeconomic factors and walkability affect success?
4. **Competitive Analysis:** Which properties outperform competitors and why?
5. **Risk Reduction:** How can investors reduce investment risk through data analysis?

This project builds a **machine learning framework** to answer these questions.

---

##  Project Structure

```
Airbnb-Success-Predictor/
├── README.md
├── data/
│   ├── listings_data.csv              # Airbnb listing features
│   ├── census_data.csv                # U.S. Census socioeconomic data
│   ├── walk_score_data.csv            # Walkability scores by location
│   └── processed_data/
│       └── merged_enriched_dataset.csv # Final integrated dataset
│
├── notebooks/
│   ├── 1_exploratory_analysis.ipynb         # Part 1: EDA & Data Understanding
│   ├── 2_clustering_analysis.ipynb          # Part 2: Unsupervised Learning
│   │   ├── K-Means Clustering
│   │   ├── Fractal Clustering
│   │   └── Golden Cluster Definition
│   ├── 3_success_prediction.ipynb           # Part 3: Supervised Learning
│   │   ├── Random Forest
│   │   ├── XGBoost
│   │   ├── KNN
│   │   └── SVM
│   └── 4_feature_importance_analysis.ipynb  # Success Driver Analysis
│
├── src/
│   ├── data_preprocessing.py          # Data cleaning & integration
│   ├── clustering.py                  # Clustering algorithms
│   ├── model_training.py              # Model implementations
│   ├── evaluation.py                  # Model evaluation metrics
│   └── utilities.py                   # Helper functions
│
├── results/
│   ├── clustering_results/            # Cluster assignments & visualizations
│   ├── model_results/                 # Performance metrics
│   ├── visualizations/                # EDA plots & charts
│   └── reports/                       # Summary reports
│
└── requirements.txt                   # Python dependencies
```

---

##  Data Sources

### 1. **Airbnb Listing Data** 
**Source:** [Inside Airbnb](https://insideairbnb.com/get-the-data/) – listing data (price, amenities, reviews, etc.)
- **Records:** ~27,000 listings
- **Features:** Price, amenities, room type, capacity, reviews, host info, description
- **Target Variables:** Nightly price, occupancy rate, review scores

### 2. **U.S. Census Data**
**Source:** [U.S. Census Bureau](https://data.census.gov/) – socioeconomic context
- **Features:** Income levels, employment rates, education, demographics
- **Purpose:** Understand neighborhood economic factors affecting rental demand
- **Integration:** Spatial join with listing locations

### 3. **Walk Score Data**
**Source:** [Walk Score](https://www.walkscore.com/professional/walk-score-data.php) – neighborhood walkability
- **Features:** Walkability scores (0-100), transit access, bike score
- **Purpose:** Assess neighborhood accessibility and lifestyle factors
- **Impact:** High walkability often correlates with higher nightly rates

---

##  Methodology

### **Part 1: Data Integration & Exploration**
- Merged Airbnb data with Census and Walk Score using reverse geocoding
- Performed spatial joins to match listings with census tracts
- Conducted EDA to understand feature distributions and correlations

### **Part 2: Unsupervised Learning (Clustering)**

Applied K-Means & Fractal Clustering to segment ~27,000 listings.

**Clustering Objectives:**
- Identify natural groupings of similar properties
- Find high-performing clusters balancing profitability and satisfaction
- Create business-meaningful segments (e.g., "luxury," "budget," "tourist-focused")

**Golden Cluster Definition:**
Defined a Golden Cluster of properties balancing profitability and guest satisfaction.

Criteria for "Golden Cluster":
- Higher than median occupancy rate
- Higher than median nightly price
- Higher than median review scores
- Consistent performance metrics

**Outcome:** Created a binary target `is_golden_cluster` for supervised learning.

### **Part 3: Supervised Learning (Classification)**

Amalgamated Airbnb data with Census & Walk Score data using reverse geocoding and spatial joins.

**Model Development:**

Ran multiple models to predict `is_golden_cluster`:
- Random Forest → F1: 0.9798, AUC: 0.9952
- XGBoost → F1: 0.9738, AUC: 0.9992
- KNN → F1: 0.8855, AUC: 0.9741
- SVM → F1: 0.7585, AUC: 0.9645

**Models Evaluated:**
1. **Random Forest:** Tree-based ensemble, strong feature importance
2. **XGBoost:** Gradient boosting, best predictive power
3. **KNN:** Instance-based learning, captures local patterns
4. **SVM:** Support Vector Machine, handles non-linear boundaries

**Hyperparameter Tuning:**
- Grid search / Random search for optimal parameters
- Cross-validation (5-fold) for robust evaluation
- Class balancing (weighted loss) for imbalanced data

---

##  Key Results

### **Model Performance**

| Model | F1-Score | AUC-ROC | Accuracy | Precision | Recall |
|-------|----------|---------|----------|-----------|--------|
| **XGBoost** | 0.9738 | **0.9992** | 0.9654 | 0.9712 | 0.9765 |
| **Random Forest** | **0.9798** | 0.9952 | 0.9701 | 0.9821 | 0.9775 |
| **KNN** | 0.8855 | 0.9741 | 0.8934 | 0.8923 | 0.8787 |
| **SVM** | 0.7585 | 0.9645 | 0.7821 | 0.7654 | 0.7516 |

**Winner:** Random Forest (F1: 0.9798) and XGBoost (AUC: 0.9992) tied as best performers

### **Key Insights**

Insight: Ensemble models benefited most from enriched datasets.

1. **Data Enrichment Matters:** Adding Census & Walk Score data improved model performance by 12-15%
2. **Ensemble Models Excel:** Random Forest and XGBoost outperform simple classifiers
3. **High Precision:** Models correctly identify successful properties 97%+ of the time
4. **Feature Importance:** Amenities, location metrics, and review scores are top predictors

### **Business Outcomes**

Identified hidden gem neighborhoods outside obvious tourist centers.

Built a Success Score (0–100) combining price, demand probability, and review scores.

Key Discoveries:
- Properties with high walkability (70+) earn 18% more on average
- Certain neighborhoods have 40%+ higher occupancy rates despite lower visibility
- Specific amenity combinations increase success likelihood by 25-35%

---

## 💻 Technologies

### **Programming Languages**
- Python 3.x

### **Data Processing & Analysis**
- `pandas` – Data manipulation and aggregation
- `numpy` – Numerical computing
- `scikit-learn` – Machine learning algorithms
- `xgboost` – Gradient boosting
- `geopy` – Reverse geocoding

### **Visualization**
- `matplotlib` – Static plots
- `seaborn` – Statistical visualizations
- `folium` – Geographic visualizations
- `plotly` – Interactive charts

### **Spatial Analysis**
- `geopandas` – Spatial data operations
- `shapely` – Geometric operations

### **Development**
- Jupyter Notebook – Interactive analysis
- Git – Version control

---

##  Installation & Setup

### **Prerequisites**
- Python 3.7+
- pip or conda package manager
- Git

### **Step 1: Clone Repository**
```bash
git clone https://github.com/ananya101001/Airbnb-Success-Predictor.git
cd Airbnb-Success-Predictor
```

### **Step 2: Create Virtual Environment**
```bash
# Using Python venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n airbnb python=3.9
conda activate airbnb
```

### **Step 3: Install Dependencies**
```bash
pip install -r requirements.txt
```

### **Step 4: Download Data**
```bash
# Download from Inside Airbnb, Census Bureau, and Walk Score
# Place CSV files in data/ directory
# See Data Sources section for links
```

### **Step 5: Run Analysis**
```bash
# Launch Jupyter
jupyter notebook

# Open notebooks in order:
# 1. 1_exploratory_analysis.ipynb
# 2. 2_clustering_analysis.ipynb
# 3. 3_success_prediction.ipynb
```

---

## 📖 Usage

### **For Data Scientists/Analysts**

1. **Explore Data:**
   ```python
   import pandas as pd
   from src.data_preprocessing import load_and_preprocess
   
   df = load_and_preprocess('data/listings_data.csv')
   df.head()
   ```

2. **Train Models:**
   ```python
   from src.model_training import train_random_forest, train_xgboost
   
   model_rf = train_random_forest(X_train, y_train)
   model_xgb = train_xgboost(X_train, y_train)
   ```

3. **Make Predictions:**
   ```python
   predictions = model_xgb.predict(X_test)
   success_probabilities = model_xgb.predict_proba(X_test)
   ```

### **For Real Estate Investors**

1. **Assess Property Success:**
   - Input property details (price, amenities, location)
   - Model predicts success probability (0-100 score)
   - Compare against similar properties in the area

2. **Identify Opportunities:**
   - Use clustering results to find high-performing neighborhoods
   - Analyze feature importance to understand success drivers
   - Target properties with specific amenity combinations

3. **Price Optimization:**
   - Use price prediction models to find optimal nightly rate
   - Avoid overpricing (lose bookings) or underpricing (lose revenue)
   - Adjust based on occupancy probability predictions

---

##  Key Findings

### **Success Drivers (Feature Importance)**

Top 10 factors predicting successful listings:
1. **Amenities:** Kitchen, Wi-Fi, Air Conditioning, Washer/Dryer
2. **Location:** Walk Score, proximity to transit, neighborhood clustering
3. **Price:** Competitively priced relative to similar properties
4. **Reviews:** Quantity and sentiment of guest reviews
5. **Host Experience:** Number of listings, years on platform
6. **Room Type:** Entire homes > private rooms > shared rooms
7. **Capacity:** Optimal guest capacity for location
8. **Availability:** Calendar availability and cancellation policies
9. **Description:** Length and detail of listing description
10. **Photos:** Number and quality of listing photos

### **Hidden Gems & Opportunities**

Demonstrated that data enrichment + ensemble models give the most accurate predictions.

- Neighborhoods outside tourist centers often have less competition
- High walkability properties command 15-20% price premiums
- Specific amenity combinations increase bookings by 25%+
- New hosts can achieve success with data-backed strategies

---

## 🚀 Future Enhancements

### **Short-term**
- [ ] Deploy as web app (Flask/Streamlit) for investor access
- [ ] Add real-time price prediction API
- [ ] Implement time-series analysis for seasonal trends
- [ ] Create investment ROI calculator

### **Medium-term**
- [ ] Expand to multiple cities/countries
- [ ] Add sentiment analysis of guest reviews
- [ ] Incorporate high-resolution property photos (CNN features)
- [ ] Build recommendation system for property improvements

### **Long-term**
- [ ] Develop deep learning models (LSTM for time-series)
- [ ] Integrate with Airbnb API for live data
- [ ] Create mobile app for investors
- [ ] Build investment portfolio optimizer

---

##  Literature & References

**Key Research Areas:**
- Dynamic pricing in peer-to-peer markets
- Guest experience factors affecting booking decisions
- Location-based valuation models
- Feature importance in real estate prediction

**Related Papers:**
- Airbnb Pricing Analysis & Optimization
- Short-term Rental Market Dynamics
- Spatial Analysis in Real Estate

---

##  Contributing

Contributions are welcome! 

**To contribute:**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Commit changes (`git commit -am 'Add improvement'`)
5. Push to branch (`git push origin feature/improvement`)
6. Submit a Pull Request

**Areas for Contribution:**
- Bug fixes and code improvements
- Additional models or algorithms
- Performance optimizations
- Documentation enhancements
- Data source integrations

---

##  License

This project is provided for educational purposes. 

---

##  Contact & Questions

**Project Author:** [@ananya101001](https://github.com/ananya101001)

For questions, suggestions, or collaboration:
- Open an Issue on GitHub
- Submit a Pull Request
- Contact via GitHub profile

---

##  Academic Context

**Completed As:** 3-part academic assignment
- **Part 1:** Exploratory Data Analysis
- **Part 2:** Unsupervised Learning (Clustering)
- **Part 3:** Supervised Learning (Classification & Prediction)

**Demonstrates Skills:**
- Data integration from multiple sources
- Feature engineering and domain knowledge
- Clustering and segmentation
- Classification modeling and evaluation
- Business acumen and decision support
- ML pipeline development

---

##  Key Statistics

- **Listings Analyzed:** ~27,000
- **Features Engineered:** 50+
- **Clusters Created:** Multiple (K-Means, Fractal)
- **Models Trained:** 4 classifiers
- **Best AUC Score:** 0.9992 (XGBoost)
- **Data Sources:** 3 (Airbnb, Census, Walk Score)

---

##  Quick Start

```bash
# Clone and setup
git clone https://github.com/ananya101001/Airbnb-Success-Predictor.git
cd Airbnb-Success-Predictor
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run analysis
jupyter notebook
# Open: 1_exploratory_analysis.ipynb
```


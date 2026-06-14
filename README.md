# Crop & Fertilizer Recommendation System

A machine learning-based web application that recommends the most suitable **crop** and **fertilizer** based on soil nutrient levels and environmental conditions. Built with Flask and scikit-learn.

## Overview

Farmers often lack data-driven guidance on which crops to plant and which fertilizers to apply given their specific soil and climate conditions. This project addresses that gap by training classification models on agricultural datasets and serving predictions through a simple web interface.

**Given** soil nutrients (N, P, K), temperature, humidity, pH, and rainfall → the system recommends a **crop** (out of 22 types) and a **fertilizer** (out of 7 types).

## Tech Stack

- **Language:** Python 3
- **ML Framework:** scikit-learn
- **Web Framework:** Flask
- **Frontend:** HTML, CSS
- **Model Serialization:** Pickle

## Datasets

| Dataset | Records | Features | Target |
|---------|---------|----------|--------|
| `Crop_recommendation.csv` | 2,200 | N, P, K, Temperature, Humidity, pH, Rainfall | Crop label (22 classes) |
| `Fertilizer.csv` | 99 | Nitrogen, Potassium, Phosphorous | Fertilizer Name (7 classes) |

### Supported Crops
Rice, Maize, Jute, Cotton, Coconut, Papaya, Orange, Apple, Muskmelon, Watermelon, Grapes, Mango, Banana, Pomegranate, Lentil, Blackgram, Mungbean, Mothbeans, Pigeonpeas, Kidneybeans, Chickpea, Coffee

### Supported Fertilizers
Urea, DAP, 14-35-14, 28-28, 17-17-17, 20-20, 10-26-26

## Models & Results

### Crop Recommendation — Naive Bayes
- Trained on 7 features (N, P, K, temperature, humidity, pH, rainfall)
- Predicts one of 22 crop classes

### Fertilizer Recommendation — Random Forest Classifier
- Trained on 3 features (Nitrogen, Potassium, Phosphorous)
- Hyperparameter tuning via GridSearchCV (n_estimators, max_depth, min_samples_split)
- **Results after tuning:**

| Metric | Score |
|--------|-------|
| Accuracy | 100% |
| Precision | 1.00 |
| Recall | 1.00 |
| F1-Score | 1.00 |

A Neural Network (MLP Classifier) was also evaluated for comparison:

| Metric | Random Forest | Neural Network |
|--------|--------------|----------------|
| Accuracy | 98.76% | 95.00% |
| Training Time | 0.88s | 0.24s |
| Log Loss | 0.032 | 0.168 |

Random Forest was selected as the production model due to superior accuracy and lower log loss.

## Project Structure

```
CROP/
├── model/
│   ├── naive_bayes_model.pkl      # Crop recommendation model
│   └── random_forest_model.pkl    # Fertilizer recommendation model
├── static/
│   └── style.css                  # Application styling
├── templates/
│   └── index.html                 # Web interface
├── app.py                         # Flask application
├── Crop_recommendation.csv        # Crop dataset (2,200 records)
├── Fertilizer.csv                 # Fertilizer dataset (99 records)
├── Recommendation_Fertilizer.ipynb # Model training notebook
└── README.md
```

## How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/Zan-n/Crop-Fertilizer-Recommendation.git
   cd Crop-Fertilizer-Recommendation
   ```

2. **Install dependencies**
   ```bash
   pip install flask numpy scikit-learn
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

4. **Open in browser**
   Navigate to `http://127.0.0.1:5000`

## How It Works

1. User enters soil and environmental parameters through the web form
2. The Naive Bayes model predicts the optimal crop based on all 7 features
3. The Random Forest model predicts the best fertilizer based on N, P, K values
4. Both recommendations are displayed on the same page

## Methodology

The Jupyter notebook (`Recommendation_Fertilizer.ipynb`) covers the full ML pipeline:

1. **Exploratory Data Analysis** — distribution plots, box plots, correlation heatmaps
2. **Preprocessing** — StandardScaler normalization, LabelEncoder for target variables, 70/30 and 80/20 train-test splits
3. **Model Training** — Random Forest and Neural Network (MLP) classifiers
4. **Hyperparameter Tuning** — GridSearchCV with 3-fold cross-validation
5. **Evaluation** — accuracy, precision, recall, F1-score, confusion matrix, log loss
6. **Model Comparison** — Random Forest vs Neural Network on accuracy, training time, and memory usage
7. **Deployment** — model export via Pickle, Flask web application

## License

This project is open source and available for educational purposes.

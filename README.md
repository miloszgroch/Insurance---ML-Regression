# Insurance Cost Prediction – Machine Learning Regression

This project focuses on predicting individual medical insurance charges using machine learning techniques.  
It demonstrates a complete workflow including data preprocessing, exploratory analysis, model training, evaluation, and deployment through a Streamlit application.

---

## Project Overview

The objective is to build regression models capable of predicting insurance costs based on demographic and lifestyle features such as:

- Age
- Sex
- BMI
- Number of children
- Smoking status
- Region

The project covers:

- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering
- Baseline regression modeling
- Neural network modeling
- Model evaluation and comparison
- Deployment via a web application

---

## Repository Structure
```
Insurance-ML-Regression/
│
├── app/ # Streamlit web application
│ └── app.py
│
├── notebooks/ # Exploratory analysis and experimentation
│ └── Insurance.ipynb
│
├── data/ # Dataset
│ └── insurance.csv
│
├── models/ # Saved trained model
│ └── model.pkl
│
├── requirements.txt
└── README.md
```

---

## Model Development

The modeling workflow includes:

1. Data preprocessing and encoding of categorical variables
2. Feature scaling where required
3. Training baseline models (e.g., Linear Regression)
4. Training an advanced Neural Network model
5. Performance evaluation using:

   - Mean Squared Error (MSE)
   - Root Mean Squared Error (RMSE)
   - R² Score

The neural network improves predictive performance by capturing nonlinear relationships between smoking status, BMI, and insurance charges.

---

## Results Summary

Model Performance

| Model | MAE | MSE | R² |
|------|------|------|------|
| SVR | 8590.78 | 175,486,326.07 | -0.10 |
| Linear Regression | 4213.57 | 72,142,778.11 | 0.55 |
| Random Forest | 2612.88 | 19,194,314.28 | 0.88 |
| Gradient Boosting | **2448.14** | **16,404,983.95** | **0.90** |

- Gradient Boosting achieved the best performance across all evaluation metrics 
and was selected as the final model for deployment in the Streamlit application.

Key findings from the analysis:

- Smoking status is the strongest predictor of insurance costs.
- BMI significantly affects charges, especially for smokers.
- Linear Regression provides a solid and interpretable baseline.
- The Neural Network model reduces prediction error and better captures nonlinear feature interactions.

The final model demonstrates strong generalization performance on unseen test data and provides reliable cost estimates.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/miloszgroch/Insurance---ML-Regression.git
cd Insurance---ML-Regression
```
Install dependencies:

```bash
pip install -r requirements.txt
```

Running the Notebook

```bash
jupyter notebook notebooks/Insurance.ipynb
```

---

## Training and Saving the Model

Ensure that the trained model is saved to the models/ directory as model.pkl.
You can export the trained model from the notebook using:

```python
import joblib
joblib.dump(model, "models/model.pkl")
```

## Running the Streamlit Application

```bash
streamlit run app/app.py
```

This will launch a local web interface allowing users to input parameters and obtain predicted insurance charges.


![Streamlit Demo](https://raw.githubusercontent.com/miloszgroch/Insurance---ML-Regression/main/app/streamlit_demo.png)

---

## Future Improvements

- Hyperparameter optimization
- Cross-validation pipeline
- Model explainability using SHAP
- Docker containerization
- Cloud deployment

---
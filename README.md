# Customer Churn Prediction

## Project Overview

This project uses Machine Learning to predict customer churn in the telecommunications industry.

Customer churn refers to customers discontinuing a company's services. The objective is to identify customers who are more likely to leave so that businesses can take proactive retention measures.

The project uses the IBM Telco Customer Churn dataset and compares three classification algorithms.

## Objectives

- Perform data cleaning and preprocessing.
- Explore customer churn patterns through data analysis.
- Build and evaluate multiple Machine Learning models.
- Compare model performance using classification metrics.
- Develop an interactive prediction application using Streamlit.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib
- Streamlit
- Jupyter Notebook

## Machine Learning Models

Three classification algorithms were implemented:

1. Logistic Regression
2. Random Forest Classifier
3. Extra Trees Classifier

The models were evaluated using Accuracy, Precision, Recall, F1-score, and ROC-AUC.

## Model Performance

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 75.76% | 53.25% | 72.19% | 61.29% | 82.47% |
| Logistic Regression | 72.57% | 49.01% | 79.68% | 60.69% | 83.51% |
| Extra Trees | 74.70% | 51.87% | 66.84% | 58.41% | 81.33% |

Random Forest was selected as the final model based on its F1-score.

## Project Structure

```text
customer-churn-prediction/
│
├── data/
├── images/
├── models/
├── notebooks/
├── reports/
├── src/
│   └── train_model.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd customer-churn-prediction
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

The project uses the IBM Telco Customer Churn dataset.

Download the CSV file and place it inside the `data/` directory with this exact filename:

`Telco-Customer-Churn.csv`

Dataset source:

https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

## Train the Model

```bash
python src/train_model.py
```

The training script generates model evaluation reports, visualizations, and the trained model.

## Run the Streamlit Application

```bash
streamlit run app.py
```

If the Streamlit launcher has an environment issue, use:

```bash
python -m streamlit run app.py
```

## Evaluation and Visualizations

The project generates:

- Customer churn distribution
- Churn by contract type
- Customer tenure analysis
- Confusion matrix
- ROC curve
- Model comparison report

## Limitations

- Model performance depends on the dataset and selected features.
- Predictions represent estimated churn probabilities, not guaranteed outcomes.
- The selected classification threshold is a demonstration setting and may require business-specific calibration.
- Further improvements could include hyperparameter tuning, cross-validation, and threshold optimization.

## Future Improvements

- Hyperparameter optimization.
- Cross-validation.
- Feature importance analysis.
- Model explainability using SHAP.
- Improved Streamlit dashboard.
- Deployment to a cloud platform.

## Author

**Muhammad Aqib**

Software Engineering Graduate | Machine Learning & Data Science Learner

---

*This project was developed for educational and portfolio purposes.*
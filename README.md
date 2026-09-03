# 🛡️ Credit Card Fraud Detection System

## 📌 Project Overview

The **Credit Card Fraud Detection System** is a Machine Learning project that identifies whether a credit card transaction is **legitimate or fraudulent**. The project uses Python and Scikit-learn to preprocess transaction data, handle class imbalance, train a classification model, and evaluate its performance using multiple metrics.

## 🎯 Objective

The main objective of this project is to build a machine learning model that can detect potentially fraudulent credit card transactions and reduce the risk of financial loss.

## 📊 Dataset

The project uses the **Credit Card Fraud Detection Dataset**.

* Total transactions: **284,807**
* Fraudulent transactions: **492**
* Legitimate transactions: **284,315**
* Target column: `Class`
* `0` → Legitimate transaction
* `1` → Fraudulent transaction

The dataset contains anonymized features `V1` to `V28`, along with `Time` and `Amount`.

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Imbalanced-learn
* Joblib
* Streamlit

## 🔄 Project Workflow

1. Load the dataset
2. Perform data cleaning
3. Check missing values and duplicates
4. Perform Exploratory Data Analysis (EDA)
5. Analyze fraudulent and legitimate transactions
6. Handle class imbalance
7. Split data into training and testing sets
8. Perform feature scaling
9. Train a Logistic Regression model
10. Evaluate the model
11. Save the trained model
12. Build a Streamlit web application

## 🤖 Machine Learning Model

The primary model used in this project is **Logistic Regression**.

Since fraudulent transactions are much fewer than legitimate transactions, `class_weight="balanced"` is used to give more importance to the minority fraud class.

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    class_weight="balanced",
    max_iter=500,
    solver="liblinear"
)

model.fit(X_train, y_train)
```

## 📈 Model Evaluation

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* PR-AUC
* Confusion Matrix

For fraud detection, **Precision, Recall, F1-Score and PR-AUC** are especially important because the dataset is highly imbalanced.

## 🌐 Streamlit Application

A Streamlit web application is included to allow users to enter transaction details and receive a fraud prediction.

The application displays:

* Fraud probability
* Transaction classification
* Legitimate/Fraud alert

## 📁 Project Structure

```text
Fraud_Detection_Project/
│
├── data/
│   └── creditcard.csv
│
├── models/
│   └── fraud_model.joblib
│
├── notebooks/
│   └── fraud_detection.ipynb
│
├── train.py
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## ⚙️ Installation

Clone the repository:

```bash
git clone <your-github-repository-url>
cd Fraud_Detection_Project
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment on Windows:

```bash
venv\Scripts\activate
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

## ▶️ Train the Model

Place `creditcard.csv` inside the `data` folder and run:

```bash
python train.py
```

The trained model will be saved inside the `models` folder.

## 🚀 Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your browser.

## 💡 Key Learning Outcomes

Through this project, I learned:

* Data preprocessing
* Exploratory Data Analysis
* Binary classification
* Handling imbalanced datasets
* Logistic Regression
* Model evaluation
* Precision, Recall and F1-score
* Model persistence using Joblib
* Building a Machine Learning web application using Streamlit

## ⚠️ Disclaimer

This project is created for **educational and portfolio purposes**. It is not intended to be used as a production financial fraud detection system.

## 👨‍💻 Author

**Suhas Angadi**

Machine Learning | Python | Data Science

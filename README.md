## **Machine Learning Classification Models – Streamlit Deployment**
**a. Problem Statement**

This project aims to build, evaluate, and deploy multiple Machine Learning classification models using a single dataset.
It demonstrates a complete end-to-end ML workflow, including:

Data preprocessing

Training six ML classification models

Evaluating models using multiple performance metrics

Building an interactive Streamlit web application

Deploying the app on Streamlit Community Cloud

The goal is to compare the performance of traditional and ensemble models and derive insights from their behavior on the same classification problem.

**b. Dataset Description**

Dataset Name: Breast Cancer Wisconsin (Diagnostic) – WDBC (UCI)
Type: Binary Classification
Instances: 569
Features: 30 numerical features (+ ID column)
Target Column: diagnosis

B = Benign

M = Malignant

Source: UCI Machine Learning Repository

The raw file wdbc.data was converted to a structured CSV file using:

python convert_wdbc_to_csv.py

**c. Models Used and Evaluation Metrics**

The following six classification models were implemented:

Logistic Regression

Decision Tree Classifier

K-Nearest Neighbors (KNN)

Naive Bayes Classifier

Random Forest Classifier (Ensemble)

XGBoost Classifier (Ensemble)

Evaluation Metrics

Each model was evaluated using:

Accuracy

AUC Score

Precision

Recall

F1 Score

Matthews Correlation Coefficient (MCC)

Comparison Table of Model Performance

**d. Observations on Model Performance**

A model-wise interpretation describing why each model performed the way it did is included in the observations table.
(Insert image-1.png in final PDF submission)

**e. Streamlit Web Application**

An interactive application was developed using Streamlit to allow users to:

Upload CSV test datasets

Select any of the six models via dropdown

View evaluation metrics (Accuracy, Precision, Recall, F1, AUC, MCC)

View Confusion Matrix visualization

View Classification Report

This enables hands-on comparison of model behavior in a user-friendly interface.

**f. Deployment Details**

Platform: Streamlit Community Cloud

Deployment Method: GitHub → Streamlit integration

Branch: main

Entry File: app.py

The application launches an interactive frontend when the deployment link is accessed.

**g. Tools and Technologies Used**

Python

Streamlit

Scikit-learn

XGBoost

Pandas

NumPy

Matplotlib

Seaborn

GitHub for version control

**h. Execution Environment (BITS Virtual Lab)**

All scripting, model execution, and testing were performed on the BITS Virtual Lab environment.

A screenshot of the environment is included in the final PDF submission.

▶ How to Run the Project
1. Install Dependencies
pip install -r requirement.txt

2. Convert Dataset
python convert_wdbc_to_csv.py

3. Train Models
python model/train_and_save.py --data data/dataset.csv --target diagnosis

4. Run Streamlit Application
python -m streamlit run app.py

**Final Notes**

All six models were trained and evaluated on the same dataset.

The Streamlit app was successfully deployed and tested.

The repository includes:

Full source code

Requirements

Trained model scripts

Streamlit UI

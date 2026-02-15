Machine Learning Classification Models – Streamlit Deployment
a. Problem Statement
The objective of this assignment is to build, evaluate, and deploy multiple machine learning classification models on a single dataset. The project demonstrates an end-to-end machine learning workflow including data preprocessing, model training, evaluation using multiple performance metrics, development of an interactive Streamlit web application, and deployment on Streamlit Community Cloud.

The goal is to compare the performance of traditional machine learning models and ensemble models on the same classification problem and provide insights into their behavior.

b. Dataset Description 

The dataset used for this project is a classification dataset obtained from a public repository (Kaggle/UCI).

Dataset Name: Breast Cancer Wisconsin (Diagnostic) – WDBC (UCI)
Type: Binary Classification
Number of Instances: 569
Number of Features: 30 numeric features (+ ID column)
Target Column: diagnosis (B = Benign, M = Malignant)
Source: UCI Machine Learning Repository
Dataset Conversion: wdbc.data converted to dataset.csv using convert_wdbc_to_csv.py

c. Models Used and Evaluation Metrics
The following six classification models were implemented and evaluated using the same dataset:

Logistic Regression
Decision Tree Classifier
K-Nearest Neighbors (KNN) Classifier
Naive Bayes Classifier
Random Forest Classifier (Ensemble Model)
XGBoost Classifier (Ensemble Model)

Each model was evaluated using the following performance metrics:
Accuracy
AUC Score
Precision
Recall
F1 Score
Matthews Correlation Coefficient (MCC)

Comparison Table of Model Performance
(image.png)

d. Observations on Model Performance
(image-1.png)

e. Streamlit Web Application
An interactive Streamlit web application was developed and deployed on Streamlit Community Cloud.

Features of the App:
Upload CSV test dataset
Dropdown menu to select classification model
Display of evaluation metrics:
Accuracy
Precision
Recall
F1 Score
AUC Score
MCC
Confusion Matrix visualization
Classification Report display


f. Deployment Details
Platform: Streamlit Community Cloud
Deployment Method: GitHub integration
Branch: main
Entry file: app.py
The deployed application opens an interactive frontend when accessed via the live link.

g. Tools and Technologies Used
Python
Streamlit
Scikit-learn
XGBoost
Pandas
NumPy
Matplotlib
Seaborn
GitHub

h. Execution Environment
The entire assignment, including model execution and testing, was performed on the BITS Virtual Lab. A screenshot of the execution environment has been included in the final PDF submission as proof.
(image-2.png)
How to Run in BITS Virtual Lab
1. Install dependencies:

2. Convert dataset:
   python convert_wdbc_to_csv.py

3. Train models:
   python model/train_and_save.py --data data/dataset.csv --target diagnosis

4. Run Streamlit App:
   python -m streamlit run app.py


Final Notes
All models were trained and evaluated on the same dataset.
The application was successfully deployed and tested.
The repository includes complete source code, dependencies, and documentation.
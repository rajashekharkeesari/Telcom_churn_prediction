Telco Customer Churn Analysis - Exploratory Data Analysis (EDA)
Overview
This Jupyter Notebook contains an exploratory data analysis (EDA) of the Telco Customer Churn dataset. The dataset includes information about telecom customers and whether they churned (left the service) in the last month. The goal of this analysis is to understand the factors contributing to customer churn and identify patterns that can help predict and reduce churn rates.

Dataset Information
Dataset Name: WA_Fn-UseC_-Telco-Customer-Churn.csv

Rows: 7043

Columns: 21

Target Variable: Churn (Yes/No)

Key Features
Customer Demographics: gender, SeniorCitizen, Partner, Dependents

Account Information: tenure, Contract, PaperlessBilling, PaymentMethod

Services: PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies

Charges: MonthlyCharges, TotalCharges

Analysis Steps
Data Loading and Initial Inspection

Load the dataset and inspect the first few rows.

Check data types and descriptive statistics.

Data Cleaning

Handle missing values (TotalCharges column).

Create tenure groups for easier analysis.

Remove unnecessary columns (customerID, tenure).

Exploratory Data Analysis

Visualize the distribution of the target variable (Churn).

Analyze the relationship between customer attributes and churn.

Examine numerical features (MonthlyCharges, TotalCharges) and categorical features (Contract, PaymentMethod, etc.).

Key Insights

The dataset is imbalanced with 73.5% of customers not churning and 26.5% churning.

TotalCharges had 11 missing values, which were dropped due to their minimal impact.

Tenure was grouped into bins (e.g., 1-12 months, 13-24 months) for better analysis.

Dependencies
Python 3.x

Libraries: pandas, numpy, seaborn, matplotlib

How to Run
Ensure the dataset file (WA_Fn-UseC_-Telco-Customer-Churn.csv) is in the same directory as the notebook.

Install required libraries using pip install pandas numpy seaborn matplotlib.

Open and run the Jupyter Notebook (Churn Analysis - EDA.ipynb).

Results
The EDA highlights key factors influencing churn, such as tenure, contract type, and service usage.

Visualizations provide clear insights into customer behavior patterns.

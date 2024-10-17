# Credit Card Fraud Detection Project

## Overview
Credit card fraud is a major concern in the financial sector, costing businesses billions every year. Early and accurate detection of fraudulent transactions can significantly reduce these losses. This project applies machine learning algorithms to detect fraudulent transactions from a dataset of credit card activities.

## Dataset
The dataset used in this project is the [Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) from Kaggle. It contains transactions made by European cardholders in September 2013.

- **Rows:** 284,807 transactions
- **Fraudulent transactions:** 492 (0.17%)
- **Non-fraudulent transactions:** 284,315 (99.83%)
- **Features:** 30 numerical features including 'Time', 'Amount', and anonymized features V1 to V28. The 'Class' label denotes whether a transaction is fraudulent (1) or non-fraudulent (0).

## Project Structure
- **Data Preprocessing:** Scaling of features such as 'Amount' and 'Time', handling of class imbalance using undersampling.
- **Exploratory Data Analysis:** Correlation analysis, data distribution, and visualization of fraud vs non-fraud.
- **Modeling:** Applied several machine learning models including:
  - XGBoost Classifier
  - Random Forest Classifier
  - K-Nearest Neighbors (KNN)
- **Evaluation Metrics:** Precision, Recall, F1-score, and F2-score.

## Data Preprocessing
1. **Feature Scaling:** 
   - 'Amount' and 'Time' features are scaled using `RobustScaler` to account for their different ranges.
   
2. **Handling Class Imbalance:** 
   - Due to the severe imbalance between fraudulent and non-fraudulent transactions, we used undersampling to balance the training data, ensuring the model does not become biased toward non-fraudulent transactions.

3. **Outlier Removal:**
   - Outliers were removed from highly correlated features using the Interquartile Range (IQR) method, focusing on features V10, V12, and V14.

## Models
We experimented with multiple machine learning classifiers to detect fraudulent transactions:
1. **XGBoost Classifier**: Gradient boosting algorithm, tuned to handle class imbalance.
2. **Random Forest Classifier**: A robust ensemble learning method for classification.
3. **K-Nearest Neighbors (KNN)**: A simple algorithm that classifies a data point based on the majority class of its neighbors.

Each model was evaluated using cross-validation on the following metrics:
- **Precision**: How many of the identified frauds were actually fraudulent.
- **Recall**: How many actual frauds were correctly identified.
- **F1 Score**: Harmonic mean of precision and recall.
- **F2 Score**: Weighted harmonic mean of precision and recall, favoring recall more.


**Best Performing Model: XGBoost Classifier**
- The XGBoost Classifier outperformed others with a weighted F2 score of 0.9815, balancing precision and recall effectively, which is important in fraud detection.


# Heart Disease Prediction with Ensemble Learning

This repository contains a heart disease prediction model developed using ensemble learning techniques. The goal is to predict the likelihood of heart disease in patients based on various health-related parameters such as age, cholesterol levels, and other clinical features. The model leverages a combination of multiple classifiers to enhance prediction accuracy and robustness.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Modeling](#modeling)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

Heart disease is one of the leading causes of death worldwide. Early diagnosis can significantly reduce mortality rates. This project aims to create a machine learning model using ensemble learning to predict the presence of heart disease based on patient data. Ensemble learning combines multiple models to improve predictive performance compared to individual models.

The ensemble methods used in this project include:
- **KNN**
- **SVM**
- **Naive Bayes**
- **Decision tree**
- **Random forest**
- **Adaboost**
- **Gradient boost**
- **XGBoost**
- **Stacking**

The goal is to compare these methods and select the most optimal model for heart disease prediction.

## Dataset

The dataset used in this project is the **Heart Disease UCI Dataset**, which is publicly available on the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Heart+Disease). The dataset contains 14 clinical features for 303 patients, such as:
- Age
- Sex
- Chest pain type
- Resting blood pressure
- Serum cholesterol
- Maximum heart rate achieved
- Exercise-induced angina
- ST depression induced by exercise
- Number of major vessels colored by fluoroscopy

Target variable:
- `0`: No heart disease
- `1`: Presence of heart disease

## Modeling

Ensemble learning techniques used in this project include:
- **Random Forest Classifier**: An ensemble method that constructs multiple decision trees and outputs the mode of the classes for classification tasks.
- **Gradient Boosting Classifier**: Sequentially builds models to correct errors made by previous models, boosting the accuracy of predictions.
- **Stacking Classifier**: Combines the predictions of multiple base models (e.g., Random Forest, Gradient Boosting) and trains a meta-model (usually a logistic regression) on their predictions.


## Results

The project evaluates the performance of each ensemble method based on the following metrics:
- **Accuracy**: Overall performance of the classifier.
- **Precision**: Proportion of positive identifications that were actually correct.
- **Recall**: Proportion of actual positives identified correctly.
- **F1-Score**: Weighted average of precision and recall.

| Model             | Accuracy | Precision | Recall  | F1 Score |
|-------------------|----------|-----------|---------|----------|
| KNN               | 0.644737 | 0.638889  | 0.621622| 0.630137 |
| SVM               | 0.672131 | 0.833333  | 0.468750| 0.600000 |
| Naive Bayes       | 0.836066 | 0.892857  | 0.781250| 0.833333 |
| Decision Tree     | 0.754098 | 0.793103  | 0.718750| 0.754098 |
| Random Forest     | 0.803279 | 0.857143  | 0.750000| 0.800000 |
| AdaBoost          | 0.836066 | 0.892857  | 0.781250| 0.833333 |
| Gradient Boosting | 0.868852 | 0.870968  | 0.843750| 0.857143 |
| XGBoost           | 0.885246 | 0.900000  | 0.843750| 0.871795 |
| Stacking          | 0.754098 | 0.774194  | 0.750000| 0.761905 |
## A Comparative Analysis of Machine Learning Models: SVM, Logistic Regression, k-NN, K-Means Clustering, and PCA

## Overview

This project provides a comprehensive analysis and comparison of several machine learning models, including Support Vector Machine (SVM), Logistic Regression, k-Nearest Neighbors (k-NN), K-Means Clustering, and Principal Component Analysis (PCA). The goal is to evaluate the performance of these models on a dataset and provide insights into their strengths and weaknesses.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Data Pre-processing](#data-pre-processing)
- [Model Implementations](#model-implementations)
  - [Support Vector Machine (SVM)](#support-vector-machine-svm)
  - [Logistic Regression](#logistic-regression)
  - [k-Nearest Neighbors (k-NN)](#k-nearest-neighbors-knn)
  - [K-Means Clustering](#k-means-clustering)
  - [Principal Component Analysis (PCA)](#principal-component-analysis-pca)
- [Comparative Analysis](#comparative-analysis)
- [Conclusion and Recommendations](#conclusion-and-recommendations)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)

## Introduction

The purpose of this project is to compare the performance of various supervised learning models (SVM, Logistic Regression, k-NN) and unsupervised learning models (K-Means Clustering, PCA) on a given dataset. The analysis includes data pre-processing, model training, hyperparameter tuning, and evaluation using various performance metrics.

## Dataset

The dataset used in this analysis consists of multiple files related to car models. These files are combined into a single DataFrame and pre-processed to handle missing values and scale the features.

Dataset files:
- `opel_corsa_01.csv`
- `opel_corsa_02.csv`
- `peugeot_207_01.csv`
- `peugeot_207_02.csv`
- `Country-data.csv`

## Data Pre-processing

- **Handling Missing Values:** Rows with missing values are dropped.
- **Feature Scaling:** Features are scaled using `StandardScaler`.
- **Train-Test Split:** The dataset is split into training and test sets.

## Model Implementations

### Support Vector Machine (SVM)

- **Hyperparameter Tuning:** Using `GridSearchCV` to find the best parameters.
- **Evaluation:** Classification report and confusion matrix.

### Logistic Regression

- **Hyperparameter Tuning:** Using `GridSearchCV` to find the best regularization strength (`C` value).
- **Evaluation:** Classification report and confusion matrix.

### k-Nearest Neighbors (k-NN)

- **Hyperparameter Tuning:** Using `GridSearchCV` to find the best number of neighbors.
- **Evaluation:** Classification report and confusion matrix.

### K-Means Clustering

- **Implementation:** Training the K-Means model.
- **Evaluation:** Metrics such as inertia and silhouette score.

### Principal Component Analysis (PCA)

- **Dimensionality Reduction:** Applying PCA to reduce the dimensionality of the dataset.
- **Visualization:** Visualizing the principal components.

## Comparative Analysis

Performance of the models is compared based on accuracy, precision, recall, F1-score, and other metrics. Visualizations such as confusion matrices and classification reports are used to illustrate the results.

## Conclusion and Recommendations

The findings from the comparative analysis are summarized, and recommendations are provided based on the performance of different models.

## Installation

To run the code in this repository, you need to have Python installed along with the following libraries:
- pandas
- scikit-learn
- seaborn
- matplotlib
- numpy

You can install the required libraries using the following command:
```bash
pip install pandas scikit-learn seaborn matplotlib numpy
```

## Usage

1. Clone the repository:
```bash
git clone https://github.com/saky-semicolon/SVM-Logistic-Regression-k-NN-K-Means-Clustering-and-PCA.git
```

2. Navigate to the project directory:
```bash
cd SVM-Logistic-Regression-k-NN-K-Means-Clustering-and-PCA
```

3. Run the Jupyter Notebook to see the analysis:
```bash
jupyter notebook "A Comparative Analysis of Machine Learning Models.ipynb"
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

<b> <i> Thank you for reading! </i> </b>
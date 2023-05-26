# Arvato Financial Services Customer Segmentation Report

This project aims to analyze the demographics data of a mail-order sales company in Germany and compare it against the general population demographics. The goal is to create a customer segmentation report using unsupervised learning techniques and identify the characteristics of the company's core customer base. Additionally, the project involves predicting potential customers for a marketing campaign using a separate dataset.

## Project Overview
The project is based on real-life data provided by Bertelsmann Arvato Analytics. It involves four datasets:

1. Udacity_AZDIAS_052018.csv: Demographics data for the general population of Germany.
2. Udacity_CUSTOMERS_052018.csv: Demographics data for the company's customers.
3. Udacity_MAILOUT_052018_TRAIN.csv: Demographics data for individuals targeted in a marketing campaign (train set).
4. Udacity_MAILOUT_052018_TEST.csv: Demographics data for individuals targeted in a marketing campaign (test set).

The project consists of two main parts:

## Part 1: Customer Segmentation Report
In this part, the project analyzes the relationship between the demographics of the company's customers and the general population. The steps involved are as follows:

1. Data Preprocessing: The datasets are loaded and preprocessed to handle missing values, convert data types, and drop irrelevant columns.
2. Dimensionality Reduction: Principal Component Analysis (PCA) is used to reduce the dimensionality of the datasets while retaining important information.
3. Clustering: K-means clustering algorithm is applied to the PCA-transformed datasets to group similar data points together. The optimal number of clusters is determined using the elbow method.
4. Comparison: The cluster labels of the customer dataset and the general population dataset are compared to identify similarities and differences, providing insights into the characteristics of the company's customer base.

## Part 2: Predictive Modeling
In this part, the project uses the insights gained from Part 1 to predict potential customers for the marketing campaign. The steps involved are as follows:

1. Data Preprocessing: The train dataset is preprocessed in a similar manner as in Part 1.
2. Feature Selection: Features from the train dataset that are most relevant for predicting customer conversion are identified.
3. Model Training and Evaluation: A machine learning model is trained on the selected features and evaluated using the test dataset. The model predicts which individuals are most likely to become customers for the company.

## Repository Contents
- `Data` folder: Contains the four datasets provided for the project.
- `Arvato Project Workbook.ipynb`: Jupyter Notebook containing the code and analysis for the project.
- `README.md`: Readme file providing an overview of the project and its contents.

## Dependencies
The following libraries are used in the project:

- numpy
- pandas
- matplotlib
- seaborn
- sklearn

Please ensure that these libraries are installed before running the code.

## How to Use
To run the project, follow these steps:

1. Clone the repository to your local machine.
2. Install the required dependencies mentioned above.
3. Open the `Arvato Project Workbook.ipynb` file using Jupyter Notebook or any other compatible environment.
4. Run the notebook cells sequentially to perform the data analysis, segmentation, and predictive modeling tasks.

Note: The code assumes that the data files are located in the `Data` folder within the project directory. Please make sure to update the file paths if the data files are stored elsewhere.

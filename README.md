# Credit_Risk_modeling
Project on segmenting customers according to their previous loan records


This is a real time data from a leading bank in India. There are two datasets. One is general information of customers and another is a report from Credit rating agency in India CIBIL which tracks the loan records of Customers including any previous and present OD line. I have preprocessed the data understanding all the features. I have performed record deletion for null value record. Using hypothese testing I understood that few of the features can be removed as I found there is a existance of multi-collinearity using Anova test and Chi-2 test.

Also for modeling I have used multiple algorithms but found XGBoost more accurate for the dataset. Hence I have created the pickle file for XGBoost after performing the hyper-parameter tuning with over 720 combinations.

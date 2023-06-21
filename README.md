# Customer Churn Prediction Project
![customer_churn](customer_churn.png)
## Introduction

Customer churn, also known as customer attrition, refers to the phenomenon where customers cease their relationship with a business or service. It is a critical challenge for companies across various industries, including banking. Being able to predict customer churn can provide valuable insights to companies, allowing them to take proactive measures to retain customers and improve customer satisfaction.

In this Jupyter notebook project, we will explore a dataset called "Bank Customer Churn Prediction." This dataset contains information about bank customers and their characteristics, such as credit score, country, gender, age, tenure, balance, and more. We will leverage this dataset to build a predictive model that can identify potential churners.

The project will be divided into the following sections:

1. **Data Loading and Exploration**: We will begin by loading the dataset into our notebook and gaining an understanding of its structure and content. We will explore the different variables, check for missing values, and perform any necessary data cleaning steps.

2. **Data Visualization**: Visualizations can provide valuable insights into the relationships between different variables and help identify patterns and trends. We will create visualizations using libraries like Matplotlib or Seaborn to explore the data further.

3. **Data Preprocessing**: Before training a predictive model, we need to preprocess the data. This step involves handling categorical variables, scaling numerical features, and splitting the dataset into training and testing sets.

4. **Model Training and Evaluation**: We will select a suitable machine learning algorithm and train it on the preprocessed data. We will evaluate the model's performance using appropriate metrics and make any necessary adjustments to improve its predictive capabilities.

5. **Customer Churn Prediction**: Using the trained model, we will make churn predictions on new, unseen data. We will discuss the importance of interpretability and explore methods to interpret the model's predictions.

By the end of this project, you will have gained hands-on experience in reading, cleaning, visualizing, and predicting customer churn using machine learning techniques. So, let's get started!

## 1. Data Loading and Exploration

The dataset consists of information about 10,000 bank customers. Here are some key findings:

- The dataset contains various attributes such as credit score, country, gender, age, tenure, balance, products number, credit card status, active membership, estimated salary, and churn.
- The dataset is complete, with no missing values in any of the columns.
- Customers have an average credit score of 650.53, with a standard deviation of 96.65.
- The average age of customers is approximately 38.92 years, with a standard deviation of 10.49.
- Customers have an average tenure of 5.01, indicating the average duration of their relationship with the bank.
- The average balance across all customers is approximately 76,485.89 USD, with a standard deviation of 62,397.41 USD.
- Most customers have one product, with an average of 1.53 products per customer.
- Around 70.55% of customers have a credit card, while approximately 51.51% are active members.
- The estimated salary of customers ranges from 11.58 USD to 199,992.48 USD, with an average of 100,090.24 USD.
- The churn rate is 20.37%, suggesting that approximately one-fifth of the customers have churned.

## 2. Data Visualization
![customer_churn](graphs/churned_or_not_churned.png)

In this dataset, there is an imbalance between the number of retained customers and the number of customers who left the bank. Retained customers are over-represented, comprising approximately 20.4% of the dataset, while customers who left the bank account for the remaining portion. This imbalance can potentially introduce bias in the modeling process.

![customer_churn](graphs/dist_continuous_variables.png)


- **credit_score:** On average, not churned customers have a slightly higher credit score (651.85) compared to churned customers (645.35).
- **age:** The average age of churned customers (44.84) is higher than that of not churned customers (37.41).
- **tenure:** There is a negligible difference in the average tenure between not churned (5.03) and churned (4.93) customers.
- **balance:** Churned customers have a higher average balance (91108.54) compared to not churned customers (72745.30).
- **products_number:** The average number of products is slightly lower for churned customers (1.48) compared to not churned customers (1.54).
- **estimated_salary:** The average estimated salary is slightly higher for churned customers (101465.68) compared to not churned customers (99738.39).



![customer_churn](graphs/dist_categorical_variables.png)


- **Gender:** The churned percentage for female customers is 25.07%, while for male customers it is 16.46%. This suggests that female customers have a higher likelihood of churning compared to male customers.
- **Country:** Among the countries in the dataset, Germany has the highest churned percentage at 32.44%, followed by Spain at 16.67% and France at 16.15%. This indicates that customers from Germany are more likely to churn compared to customers from the other countries.
- **Credit Card:** Both categories, 0 and 1, have similar churned percentages of approximately 20.18% and 20.81% respectively. This suggests that the presence of a credit card does not significantly impact the likelihood of churn.
- **Active Member:** Customers who are active members (category 1) have a lower churned percentage of 14.27%, while non-active members (category 0) have a higher churned percentage of 26.85%. This indicates that being an active member contributes to higher customer retention.


## 3. Data Preprocessing
### Feature Analysis

The aim of this part of the code is to perform a comprehensive analysis of the dataset's features and their relationships. By calculating the correlation matrix, we aim to uncover any linear associations between continuous variables, which helps us understand how changes in one variable may impact another. Additionally, conducting the chi-square analysis allows us to examine the dependence between categorical variables and identify any significant associations. This information is crucial for gaining insights into the underlying patterns, dependencies, and potential predictors within the dataset. Ultimately, this analysis aids in feature selection, identifying key variables, and understanding the factors that may influence the target variable or outcome of interest.

**Correlation Matrix:**
![customer_churn](graphs/corr_matrix.png)

Looking at the correlation matrix of the continuous variables, we can observe the following:

- Credit Score: It shows very weak correlation with all other variables, indicating that it may not have a significant impact on the other variables or the churn rate. It could be a candidate for dropping.
- Age: There is no strong correlation with any other variable. However, age is often considered an important factor in predicting churn, so it may still be valuable to retain.
- Tenure: It has a very weak positive correlation with estimated salary. This suggests that customers with longer tenure may have slightly higher estimated salaries, but the correlation is not strong enough to conclude a significant impact on churn.
- Balance: There is a weak negative correlation with products number, indicating that customers with higher balances may have fewer products. However, the correlation is not strong enough to make a definite conclusion about its impact on churn.
- Products Number: It has a weak negative correlation with balance, suggesting that customers with more products may have lower balances. However, the correlation is not strong enough to make a conclusive judgment about its impact on churn.
- Estimated Salary: It shows no strong correlation with any other variable, indicating that it may not have a significant impact on the other variables or the churn rate. It could be a candidate for dropping.

Based on these observations, the variables that could potentially be dropped are "credit_score" and "estimated_salary" as they exhibit weak correlations with other variables. However, it is important to further analyze these variables and consider other factors before making a final decision on variable selection.


**Chi-square test:**

```        
         Variable  Chi-square       p-value
3        country  301.255337  3.830318e-66
0  active_member  242.985342  8.785858e-55
2         gender  112.918571  2.248210e-26
1    credit_card    0.471338  4.923724e-01
```

**Based on the chi-square test results, we can conclude the following:**

- Country: The country variable shows a significant association with the churn rate, as indicated by the high chi-square value and very low p-value. Therefore, the country variable has an impact on the churn rate.
- Active Member: The active_member variable also demonstrates a significant association with the churn rate, as evidenced by the high chi-square value and very low p-value. Hence, the active_member variable affects the churn rate.
- Gender: The gender variable exhibits a moderate association with the churn rate, as indicated by a relatively high chi-square value and low p-value. Therefore, the gender variable has an impact on the churn rate.
- Credit Card: The credit_card variable shows no significant association with the churn rate, as the chi-square value is relatively low and the p-value is above the typical significance level of 0.05. Thus, the credit_card variable does not appear to have an impact on the churn rate.

In summary, country, active_member, and gender variables have an impact on the churn rate, while the credit_card variable does not. 


**Based on the two previous analysis, the variables  "credit_card", "credit_score" and "estimated_salary" will be dropped**

### Encoding of categorical descriptors

Encoding variables is important to represent categorical data in a format that can be effectively used by machine learning algorithms. In the case of gender, encoding it as numerical values (e.g., 0 for male, 1 for female) allows algorithms to interpret and process the data. This conversion enables the algorithms to identify patterns and relationships in the data that can be used for making predictions or classifications. By encoding variables, we make the data more accessible and compatible with machine learning techniques, improving the overall accuracy and effectiveness of the models.

### Min-Max scaling
Scaling is important in machine learning to ensure variables are on a similar scale. Min-Max scaling, or normalization, transforms data to a fixed range (e.g., 0-1). It preserves relative relationships between data points, benefits non-Gaussian distributions, and improves algorithm performance and convergence. Scaling eliminates bias from variables with larger magnitudes and facilitates accurate predictions.


**In a real-world scenario, it is often beneficial to engineer new variables by combining existing variables to capture more complex relationships and improve predictive power. However, for the purpose of this demonstration project, we will focus on utilizing the variables that are already available in the dataset without introducing additional engineered variables.**

## Model Training and Evaluation

### Logistic regression model

#### Results
![confusion_matrix_LR](graphs/confusion_matrix_LR.png)

```
precision    recall  f1-score   support

           0       0.83      0.96      0.89      1607
           1       0.56      0.20      0.30       393

    accuracy                           0.81      2000
   macro avg       0.70      0.58      0.59      2000
weighted avg       0.78      0.81      0.77      2000
```


- Test Accuracy: The model achieved an accuracy of 81.20% on the test set, indicating that it correctly classified 81.20% of the samples.
- Precision: For the positive class (churned customers), the precision is 0.56, suggesting that out of all the predicted churned customers, only 56% were actually churned.
- Recall: The recall for the positive class is 0.20, indicating that the model was able to correctly identify only 20% of the actual churned customers.
- F1-Score: The F1-score is a harmonic mean of precision and recall. For the positive class, the F1-score is 0.30, reflecting a trade-off between precision and recall.
- Confusion Matrix: the matrix shows that the model correctly predicted 1544 non-churned customers (true negatives) and 80 churned customers (true positives). However, it also misclassified 63 non-churned customers as churned (false positives) and 313 churned customers as non-churned (false negatives).


### Support Vector Machine (SVM)
#### Results
![confusion_matrix_SVM](graphs/confusion_matrix_SVM.png)

```
             precision    recall  f1-score   support

           0       0.87      0.98      0.92      1607
           1       0.80      0.40      0.53       393

    accuracy                           0.86      2000
   macro avg       0.83      0.69      0.73      2000
weighted avg       0.85      0.86      0.84      2000
```

- Test Accuracy: The model achieved an accuracy of 86.20% on the test set, indicating that it correctly classified 86.20% of the samples.
- Precision: For the positive class (churned customers), the precision is 0.80, suggesting that out of all the predicted churned customers, 80% were actually churned.
- Recall: The recall for the positive class is 0.40, indicating that the model was able to correctly identify 40% of the actual churned customers.
- F1-Score: The F1-score is a harmonic mean of precision and recall. For the positive class, the F1-score is 0.53, reflecting a trade-off between precision and recall.
- Confusion Matrix: the matrix shows that the model correctly predicted 1567 non-churned customers (true negatives) and 157 churned customers (true positives). However, it misclassified 40 non-churned customers as churned (false positives) and 236 churned customers as non-churned (false negatives).



## Conclusion

In this project, we developed and evaluated two models, a Support Vector Machine (SVM) model and a logistic regression model, for predicting bank customer churn. Here is a comparison of their performance and suggestions for improvement:

```
SVM Model:
    Accuracy: 86.20%
    Precision (Churned Customers): 0.80
    Recall (Churned Customers): 0.40
    F1-Score (Churned Customers): 0.53

Logistic Regression Model:
    Accuracy: 81.20%
    Precision (Churned Customers): 0.56
    Recall (Churned Customers): 0.20
    F1-Score (Churned Customers): 0.30
```

The SVM model outperforms the logistic regression model in terms of accuracy and performance for customer churn prediction.
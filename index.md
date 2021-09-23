# Rayhan Ali's Data Science Portfolio

## Project 1: HOUSE PRICE PREDICTION

### Objective: Using 79 explanatory variables, predict the final house price of each home.






# Project 2: CHURN ANALYSIS AND CLASSIFICATION

### Objective: 
- To analyze the characteristics of the customers that discontinued a service (churned). 
- To draw insights from the data in order to reduce the churn rate.
- Train a classification model to predict whether or not a customer will churn

### Background of the data: The data set includes information about
- Customers who left within the last month – the column is called Churn
- Services that each customer has signed up for – phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies
- Customer account information – how long they’ve been a customer, contract, payment method, paperless billing, monthly charges, and total charges
- Demographic info about customers – gender, age range, and if they have partners and dependents

## Part 1: Data Exploration

After importing the necessary libraries and the data, .info() method is called to explore the datatype of features

![](https://github.com/syednuman42/Syed-Numan-Portfolio/blob/main/Project%202/images/df.head.PNG?raw=true)

![](https://github.com/syednuman42/Syed-Numan-Portfolio/blob/main/Project%202/images/dfinfo.PNG?raw=true)

There are 7032 rows of data with 21 columns

We can see that there are a lot of categorical features so we will later have to create dummy variables for them.

The statistical summary of numerical data :

![](https://github.com/syednuman42/Syed-Numan-Portfolio/blob/main/Project%202/images/dfdescribe.PNG?raw=true)

We can confirm that there is no missing data by checking for null values


Checking the balance of the class label(Churn) with a countplot:

![](https://github.com/syednuman42/Syed-Numan-Portfolio/blob/main/Project%202/images/churn%20countplot.png?raw=true)


The class label looks slightly imbalanced. However, there are quite a few instances of both the categories.



The feature - 'Contract type' has 3 categories: month-to-month, one year and two year contract type

We wish to explore the distribution of TotalCharges per Contract type and separate them based on whether they churned or not

![](https://github.com/syednuman42/Syed-Numan-Portfolio/blob/main/Project%202/images/tenure%20cohort%20vs%20contract.png?raw=true)


We can see that for contract type month to month, the distribution of customers who have churned is similar to that of those that did not churn. One possible reason for this could be that it is likely that **a customer that chooses month to month service does not plan to stay for a long period of time**.

What's interesting is that for the customers who have churned in the one/two year contract type the total charge's median is more than that of the people who did not churn.
This implies that **one/two year contract customers are more likely to churn if they had more total charge**

Assuming we cannot do much about the customers in the month-to-month category, our main objective now is to reduce churn rate for one and two year contract.
To do this we have to look deeper into the data to find out how to lower the charges for one/two year contract

The correlation between the features and the target label(churn):

![](https://github.com/syednuman42/Syed-Numan-Portfolio/blob/main/Project%202/images/feature%20correlation.png?raw=true)



## PART 2: CHURN ANALYSIS:

Exploring the count of customers with different tenure with a histogram:

![](https://github.com/syednuman42/Syed-Numan-Portfolio/blob/main/Project%202/images/tenure%20countplot.png?raw=true)

We can see that there are a lot of customers with a small tenure i.e. customers who have just started and are maybe on a month to month contract

We can also see that there are a lot of customers with a high tenure value. These are the customers who have been using the service for a long period of time and will most likely continue to use it.

Moreover, there are spikes around 12,24,48 months mark. It is likely that these customers had a one/two year contract who churned after their contract was over.

Comparing tenure of customers based on contract and churn where columns represent contract type and rows represent churn

![](https://github.com/syednuman42/Syed-Numan-Portfolio/blob/main/Project%202/images/tenure%20displot.png?raw=true)

We can see that customers with one/two year contract type are less likely to churn than the month-to-month contract type

An interesting thing here is that there are a lot of customers with a high tenure yet they are still on month-to-month contract type. We know that a month-to-month contract type customer is more likely to churn. Therefore, if we can switch these high tenure month-to-month contract type customer to one/two year contract then maybe we could reduce their churn rate. This could be done by offering some discount on yearly plans.

Exploring Monthly charges vs Total Charges with a Scatterplot:

![](https://github.com/syednuman42/Syed-Numan-Portfolio/blob/main/Project%202/images/total%20charges%20vs%20monthly%20charges%20scatterplot.png?raw=true)

This implies that a lot of people tend to churn if their monthly charges are higher. The company could then reduce monthly charges as from the company's perspective the goal is to maximize total charges.



## Part 3: Predictive Modelling:

1. Explanatory and dependant variables are separated. 
2. Dummies are created for the categorical variables.
3. The column 'customerID' is dropped since it has no pattern.
4. A train test split is performed using a test size of 10%

### Decision Tree Classifier

A decision tree model is initiated, fitted and trained. 

Then model predictions are calculated and test metrics are evaluated using a confusion matrix.

![](https://github.com/syednuman42/Syed-Numan-Portfolio/blob/main/Project%202/images/conf%20matrix%20dt.png?raw=true)

The confusion matrix shows that a decision tree classifier is performing much better on no churn than yes churn.
Almost 50% of the customers were wrongly classified to not churn.
Since our main objective is to correctly identify the people that did churn, the model is not perfomingly satisfactory!

Barplot showing the Importance of various features according to Decision Tree Classifier:

![](https://github.com/syednuman42/Syed-Numan-Portfolio/blob/main/Project%202/images/feat%20imp%20dt.png?raw=true)

It was not clear from earlier data exploration that internet service fiber optics would be such an important feature!

### Random Forest Classifier

![](https://github.com/syednuman42/Syed-Numan-Portfolio/blob/main/Project%202/images/conf%20matrix%20rf.png?raw=true)


It looks like a random forest model with default values is performing worse than a decision tree classifier.
To improve this model we can do a grid search to tune the hyperparameters.

### Adaboost Classifier

![](https://github.com/syednuman42/Syed-Numan-Portfolio/blob/main/Project%202/images/conf%20matrix%20ada.png?raw=true)

AdaBoost is performing slightly better than a decision tree model


### Logistic Regression

![](https://github.com/syednuman42/Syed-Numan-Portfolio/blob/main/Project%202/images/conf%20matrix%20log.png?raw=true)


### Support Vector Classifier

![](https://github.com/syednuman42/Syed-Numan-Portfolio/blob/main/Project%202/images/conf%20matrix%20svm.png?raw=true)


### Conclusion: Since all the methods are performing kind of similar. This might be the best possible outcome for this task. In addition to this, we could do some feature engineering or a grid search to improve the model.


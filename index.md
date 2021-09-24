## Project 1: HOUSE PRICE PREDICTION

### Objective: Using 79 explanatory variables, predict the final house price of each home.

### Part 1: Exploratory Data Analysis

Initial look at the dataset:

![](https://github.com/rayhan-ali/Rayhan_Portfolio/blob/gh-pages/Project%201:%20House%20Price%20Prediction/images/train_head.png?raw=true)

Note: Number of training features = Number of test features.

Even though the number of features are the same, we must also check if the features themselves are exactly the same. I found no mismatching columns across the two datasets.

### Missing Values

**Note that it is important to deal with training and test sets during imputation SEPERATELY to avoid DATA LEAKAGE**

Plot percentage of missing data for both test and train sets:

IMAGE_ missing value_train

Image_mising_test

Both the training and test datasets have 4 features (Fence, Alley, MiscFeature and PoolQc) with more than 60% of missing data. As a rule of thumb, these features are removed from both sets.

### Missing Values Imputation: Training data

Overview of the features with missing data:

IMAGE_missing_training

- I look over the description file to study the meaning of these features. For most features, values of NaN are probably there because the particular feature does not exist since it is rare that a garage or a basement actually exists but someone decided to not include it during data collection. So a value of 0 or NA or none would be meaningful to replace for the features that represent basement, garage, masonry veneer type/area.
- As per the description "LotFrontage" represents the linear feet of street connected to property. The best way to impute it is to replace it with the median street length in the particular neighbourhood of that house.
- "Electrical" has just 1 missing value and can be replaced with the mode.
- 'FireplaceQu' represents the quality of fireplace. Most likely, the missing values correspond to the fact that there is no fireplace. I checked this intuition and found it to be true. So replaced missing values in Fireplacequ with NA.

This resolves all the missing value issues for the training data. 

### Missing Values Imputation: Training data

- Most features with missing values are the same as those we found earlier in train data. So applying similar imputation techniques for them should work.
- The features with missing values that were not found in train data have very few (<10) missing values. I replaced these missing values with the corresponding mode value.

### Splitting the data into Categorical and Quantitative
Numerical features : 40
Categorical features : 36

### Part 2: Feature Engineering

### Skewness: 
Checking the skew/kurtosis of the numerical features (for the train set)

IMAGE skew

It seems that a lot of features are highly skewed and should be transformed. I use log-transform on features that have a skew of more than 0.75 or less than -0.75 (rule of thumb). 

### One-hot Encoding Categorical Variables

The proper way to handle this issue is to fit a one-hot encoder on training dataset and then call transform function on both training and test datasets, separately, to get dummy variables. This ensures no mismatch between dummy variables in training and test datasets. It also guarantees no leakage issue.

### Feature Selection

- A crude method is to look for correlation between each of the explanatory variables and the target variable. 
- Pearson's correlation only measures linear association, while Spearman's correlation can be used to check for any monotonic relationship. Creating a heatmap of features that have a Spearman correlation of more than 0.4:

IMAGE_Heatmap

- To avoid the problem of multicollinearity, features that are highly correlated with each other are dropped. Three sets of features highly correlated: "TotalBsmtSF, 1stFlrSF", "GarageArea, GarageCars" and "TotRmsAbvGrd, GrLivArea".
- Looking at the description of the features, it is obvious that dropping any one feature from each group is fine since they represent essentially the same thing. We drop 1stFlrSF, GarageCars and TotRmsAbvGrd
- Temporal Features: Features like 'Year-Built' and 'Garage Year Built' seem to be highly correlated. To avoid auto-correlation, 'YearRemodAdd' and 'GarageYrBlt' are dropped and only 'YearBuilt' is kept.

### Feature Selection: Wrapper Method (LASSO)

Using the 'lasso' regression technique, we can drop some features. I used a value of alpha so as to obtain roughly features. 
Lasso regression found the following features to be the most important:

'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'BsmtFinSF1',
'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '2ndFlrSF', 'GrLivArea',
'BsmtFullBath', 'FullBath', 'Fireplaces', 'GarageArea', 'WoodDeckSF',
'OpenPorchSF', 'EnclosedPorch', 'ScreenPorch', 'MiscVal', 'YrSold'

### Feature Selection: Iterative (Decision Tree Regressor)

Recursive feature elimination using the decision tree regressor was again used to select 20 most important features:

'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt',
'MasVnrArea', 'BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF', '2ndFlrSF',
'GrLivArea', 'FullBath', 'HalfBath', 'GarageArea', 'WoodDeckSF',
'OpenPorchSF', 'EnclosedPorch', 'MoSold', 'YrSold', 'hasfireplace'

Analysing the features selected by the above 3 methods and going through the features description file, I select the following features:

'LotFrontage', 'LotArea', 'OverallQual', 'YearBuilt','OverallCond','TotalBsmtSF', 'FullBath',
'GarageArea', 'YrSold','OpenPorchSF','GrLivArea','FullBath'


### Part 3: Modelling

### Hyper-parameter Tuning using Pipeline, GridSearchCV

After combining the categorical and numerical data back together, pipelines were created each for these four algorithms: Lasso, Ridge, Elasticnet, SVR.
It is important to construct pipelines while doing a GridSearch for the hyper-parameters. This ensures that the preprocessing steps don't cause **information leakage** during the cross-validation stage.

Calculating rmse scores for each model:
- Ridge: 0.1248 
- Lasso: 0.1232 
- Elastic Net: 0.1231 
- SVR: 0.1238 
- GBR: 0.1246 

### Ensembling

Depending upon the scores above, an ensemble (blend) of all the models was created. The respective weights for each model was decided based on their performance in the individual performance above. Since the models performed relatively similarly, they were all given roughly the same weight. 

The ensembled model performed relatively better than any of the individual model achieving an **rmse score of 0.0916**.  


## [Project 2: CHURN ANALYSIS AND CLASSIFICATION](https://github.com/rayhan-ali/Rayhan_Portfolio/blob/main/Project%202:%20CHURN%20ANALYSIS%20AND%20CLASSIFICATION/Project%202-%20CHURN%20ANALYSIS%20AND%20CLASSIFICATION.ipynb)

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

![](https://github.com/rayhan-ali/Rayhan_Portfolio/blob/gh-pages/Project%202/images/df.head.PNG?raw=true)

![](https://github.com/rayhan-ali/Rayhan_Portfolio/blob/gh-pages/Project%202/images/dfinfo.PNG?raw=true)

There are 7032 rows of data with 21 columns

We can see that there are a lot of categorical features so we will later have to create dummy variables for them.

The statistical summary of numerical data :

![](https://github.com/rayhan-ali/Rayhan_Portfolio/blob/gh-pages/Project%202/images/dfdescribe.PNG?raw=true)

We can confirm that there is no missing data by checking for null values


Checking the balance of the class label(Churn) with a countplot:

![](https://github.com/rayhan-ali/Rayhan_Portfolio/blob/gh-pages/Project%202/images/churn%20countplot.png?raw=true)


The class label looks slightly imbalanced. However, there are quite a few instances of both the categories.



The feature - 'Contract type' has 3 categories: month-to-month, one year and two year contract type

We wish to explore the distribution of TotalCharges per Contract type and separate them based on whether they churned or not

![](https://github.com/rayhan-ali/Rayhan_Portfolio/blob/gh-pages/Project%202/images/tenure%20cohort%20vs%20contract.png?raw=true)


We can see that for contract type month to month, the distribution of customers who have churned is similar to that of those that did not churn. One possible reason for this could be that it is likely that **a customer that chooses month to month service does not plan to stay for a long period of time**.

What's interesting is that for the customers who have churned in the one/two year contract type the total charge's median is more than that of the people who did not churn.
This implies that **one/two year contract customers are more likely to churn if they had more total charge**

Assuming we cannot do much about the customers in the month-to-month category, our main objective now is to reduce churn rate for one and two year contract.
To do this we have to look deeper into the data to find out how to lower the charges for one/two year contract

The correlation between the features and the target label(churn):

![](https://github.com/rayhan-ali/Rayhan_Portfolio/blob/gh-pages/Project%202/images/feature%20correlation.png?raw=true)



## PART 2: CHURN ANALYSIS:

Exploring the count of customers with different tenure with a histogram:

![](https://github.com/rayhan-ali/Rayhan_Portfolio/blob/gh-pages/Project%202/images/tenure%20countplot.png?raw=true)

We can see that there are a lot of customers with a small tenure i.e. customers who have just started and are maybe on a month to month contract

We can also see that there are a lot of customers with a high tenure value. These are the customers who have been using the service for a long period of time and will most likely continue to use it.

Moreover, there are spikes around 12,24,48 months mark. It is likely that these customers had a one/two year contract who churned after their contract was over.

Comparing tenure of customers based on contract and churn where columns represent contract type and rows represent churn

![](https://github.com/rayhan-ali/Rayhan_Portfolio/blob/gh-pages/Project%202/images/tenure%20displot.png?raw=true)

We can see that customers with one/two year contract type are less likely to churn than the month-to-month contract type

An interesting thing here is that there are a lot of customers with a high tenure yet they are still on month-to-month contract type. We know that a month-to-month contract type customer is more likely to churn. Therefore, if we can switch these high tenure month-to-month contract type customer to one/two year contract then maybe we could reduce their churn rate. This could be done by offering some discount on yearly plans.

Exploring Monthly charges vs Total Charges with a Scatterplot:

![](https://github.com/rayhan-ali/Rayhan_Portfolio/blob/gh-pages/Project%202/images/total%20charges%20vs%20monthly%20charges%20scatterplot.png?raw=true)

This implies that a lot of people tend to churn if their monthly charges are higher. The company could then reduce monthly charges as from the company's perspective the goal is to maximize total charges.



## Part 3: Predictive Modelling:

1. Explanatory and dependant variables are separated. 
2. Dummies are created for the categorical variables.
3. The column 'customerID' is dropped since it has no pattern.
4. A train test split is performed using a test size of 10%

### Decision Tree Classifier

A decision tree model is initiated, fitted and trained. 

Then model predictions are calculated and test metrics are evaluated using a confusion matrix.

![](https://github.com/rayhan-ali/Rayhan_Portfolio/blob/gh-pages/Project%202/images/conf%20matrix%20dt.png?raw=true)

The confusion matrix shows that a decision tree classifier is performing much better on no churn than yes churn.
Almost 50% of the customers were wrongly classified to not churn.
Since our main objective is to correctly identify the people that did churn, the model is not perfomingly satisfactory!

Barplot showing the Importance of various features according to Decision Tree Classifier:

![](https://github.com/rayhan-ali/Rayhan_Portfolio/blob/gh-pages/Project%202/images/feat%20imp%20dt.png?raw=true)

It was not clear from earlier data exploration that internet service fiber optics would be such an important feature!

### Random Forest Classifier

![](https://github.com/rayhan-ali/Rayhan_Portfolio/blob/gh-pages/Project%202/images/conf%20matrix%20rf.png?raw=true)


It looks like a random forest model with default values is performing worse than a decision tree classifier.
To improve this model we can do a grid search to tune the hyperparameters.

### Adaboost Classifier

![](https://github.com/rayhan-ali/Rayhan_Portfolio/blob/gh-pages/Project%202/images/conf%20matrix%20ada.png?raw=true)

AdaBoost is performing slightly better than a decision tree model


### Logistic Regression

![](https://github.com/rayhan-ali/Rayhan_Portfolio/blob/gh-pages/Project%202/images/conf%20matrix%20log.png?raw=true)


### Support Vector Classifier

![](https://github.com/rayhan-ali/Rayhan_Portfolio/blob/gh-pages/Project%202/images/conf%20matrix%20svm.png?raw=true)


### Conclusion: Since all the methods are performing kind of similar, this might be the best possible outcome for this task. In addition to this, we could do some feature engineering or a grid search to improve the model.


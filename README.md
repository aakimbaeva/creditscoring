# Credit scoring project
This was a collaborative project with my classmate for a Data Mining course. The purpose of this project was to conduct a comparative study on the accuracy 
and to explore the best performance of credit scoring models to evaluate the applicant’s credit score from the applicant’s input features. In order to find 
the most accurate model, the historical dataset with 26,962 business loans of an anonymous bank for 2016-2017 were used. In addition, 44 variables of company 
information, financial data, socio-demographic and behavioral features were provided. The target variable has a binary value, where Class 1 stands for Default
(defaulted credit obligations over a period of 24 months) and 0 for No default (has not defaulted credit obligations over a period of 24 months). Data mining 
algorithms selected for this comparison of credit scoring models are the following: Decision Tree, Logistic Regression, K-Nearest Neighbors and Random Forest.
The results showed that Random Forest has higher accuracy rates and therefore outperformes the other proposed classification methods when it comes to 
distinguishing between good and bad payers. The following sections explain in more detail how we approached the problem and selected a prediction model for 
estimating the probability score of a credit default. 

## Data Understanding
The training dataset consists of 26,963 rows and 46 columns. The first row consists of indicators’ names and other 26,962 are loan cases. Columns consist of 
an identification number of loans (LoanID), a target variable and 44 features. To get a clear idea of the training dataset and the characteristics of each object, 
the following steps were performed. 

First, the data types were defined using the corresponding built-in method. In general, features have either a string or numeric type. All the variables of string 
type are considered as categorical data, while numeric variables as numerical data. However, it is important to note that some variables with numerical data 
should also be measured as categorical, therefore they were transformed to strings. 

Secondly, by using DataFrame.describe() method the summary statistics of variables were studied to understand how variables are distributed and what their 
range is. Then the number and percentage of missing values for each variable were calculated. After going through all of the above steps, we have obtained an 
insight of how each variable should be treated at the Data Preparation phase.

## Data Preparation
First of all, columns with more than 90% of missing values were dropped from training, validation and test sets, because we considered them as hard to fill. 
For columns with more than 50% and less than 90% of missing values we replaced existing values with 0 and missing values (NaN) with 1. Next, for numerical 
columns with less than 50% of missing data we replaced NaN in training and test sets with the average value of training set, while for categorical columns we 
used the most frequent value. 

For categorical variables we decided to use dummy encoding by pd.get_dummies(), but before that we analyzed those variables using .nunique() method to see 
whether they can be transferred or not. So, we created dummy variables only for features with less than 10 unique values, because it will take less time to 
create columns of dummy variables. High-cardinality categorical variables with more than 200 unique values were dropped to make process simpler, but for 
other 2 variables with around 50 unique values, we used percentage frequency in all data for replacement, because we believe it will give us some important 
information in prediction process. 

Variables that are measured at different scales do not contribute equally to the model fitting and model learned function and might create a bias. Thus, to 
deal with this potential problem feature-wise standardization also known as statistical normalization was used prior to model fitting by creating 
def Preprocessing_continuous() function, whereas outliers were treated via truncation.  

## Modeling and Evaluation 
We believe that it’s a good practice to experiment with a number of different methods when modeling or mining data rather than relying on a single model for 
final deployment. Therefore, the four models such as Decision Tree, Logistic Regression, K-Nearest Neighbors and Random Forest were deployed on the validation
set. For all these models we implemented a 5-fold cross validation on the training set. 

First, the training dataset was split into 5 randomly generated subsets, which were composed of 80% of the data for training and 20% for testing using 
sklearn.model_selection.KFold(). In order to select the best model based on the AUC score, the hyperparameters of each model were first configured on each 
subset of the training dataset using sklearn.model_selection.GridSearchCV() with cv=5, and then were tested on the remaining 20% of that subset, so that the 
AUC was calculated accordingly. As the result 5 AUC scores were calculated on different data splits for each model.

To estimate the default probability scores on DSC2020_Test dataset, the data was preprocessed with the same procedure explained in section 3 and data inconsistency 
was fixed. Then, the number of trees (n_estimators) of Random Forest model was tuned using a 5-fold cross-validation grid search. So, the model was run on the test 
dataset and probability scores of default and no default variables were derived. 

## Conclusion
This project shows an importance of data mining in working with big data to make right decisions by the management. First of all, it is very crucial to dedicate 
time and effort to the data understanding and data preparation phases, for the reason that outcomes will be more accurate and unbiased. With well-prepared and 
standardized data there is a big chance to find the best performed model to solve specific problems. In our case the best model for calculating creditworthiness 
of loan recipients is the Random Forest model. 





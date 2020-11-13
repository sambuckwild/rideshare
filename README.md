# Churn Prediction for Ride-Share Company, "Lyber"

![rideshare](https://www.sharedmobility.news/wp-content/uploads/2017/10/shared11.png)

## Table of Contents

- [Background](#Background)
- [The Task](#The-Task)
- [The Data](#The-Data)
- [Initial Hypotheses and Predictions](#Initial-Hypotheses-and-Predictions)
- [EDA](#EDA) 
- [Analysis](#Analysis)
- [Conclusion](#Conclusion)

## Background

Ride share companies are very popular in the US and worldwide, as they provide on-demand ride services for passengers in places wihtout local taxi outposts. They also are a big player in the gig-economy, where millions of people can make extra income as a ride-share driver.

Among with new goverment regulations and news scandals, ride-share apps are still wildly popular. But, many people stop using ride-share apps. We want to get to the bottom of this.

## The Task
We set out to answer this question for the ride-share company, which we will call **Lyber**:
**What factors are the best predictors for retention?** 
<p>
To answer this question, we considered a user as a **churn** if they had not taken a trip since June 1, 2014. We used this information as our target to predict in the models we built to explore the question. We also care about building a model that minimizes error and lets us interpret the factors that contributed to our predictions.

## The Data

We were given a csv file that represented user information from data pulled on July 1, 2014

- `city`: city this user signed up in 
- `phone`: primary device for this user, either iPhone or Android
- `signup_date`: date of account registration; in the form `YYYYMMDD`
- `last_trip_date`: the last time this user completed a trip; in the form `YYYYMMDD
- `avg_dist`: the average distance (in miles) per trip taken in the first 30 days after signup
- `avg_rating_by_driver`: the rider’s average rating over all of their trips 
- `avg_rating_of_driver`: the rider’s average rating of their drivers over all of their trips 
- `surge_pct`: the percent of trips taken with surge multiplier > 1 
- `avg_surge`: The average surge multiplier over all of this user’s trips 
- `trips_in_first_30_days`: the number of trips this user took in the first 30 days after signing up 
- `luxury_car_user`: TRUE if the user took a luxury car in their first 30 days; FALSE otherwise 
- `weekday_pct`: the percent of the user’s trips occurring during a weekday

Overall, there were 50000 rows and 12 columns in the original `churn.csv ` dataset. The columns for `phone`,  `avg_rating_of_driver`, and `avg_rating_by_driver` had 396, 8122, and 201 NAN values, respectively. 


## Initial Hypotheses and Predictions 
Upon looking at the column names, and given our own experienes with ride-sharing, we thought a few features may be more important to the reason why someone would churn or stay with this ride-share.
<p>
  
- ```weekday_pct``` : We theoried that a higher weekday percentage would mean the peson is more likely using it  for work than for leisure, and this would mean a person with a higher ```weekday_pct``` would be **less** likely to churn.  
- ```avg_surge``` and ```surge_pct```: If these values are high, we think that these users may be more likely to stick with the program since they're willing to suck up the additional surge cost because they truly need this service. 
- ```phone```: A person may be more likely to churn due to the ride-share app on the specific model of phone they have (iPhone or Android). Sam made a good point about this one: Perhaps the app on a one phone model is worse than the app on another, and the user is more likely to churn because using the app is a pain in the butt. We'll leave that to the app developers once we present them with our findings.  
<p>
  
**Other Hypothesis**
The NAN values represent a type of person: For instance, the people who have NAN values for ```avg_rating_by_driver``` represent a type of person who doesn't leave ratings. That might have something to do with whether they churn. Unfortunately, that is beyond the scope of this analysis. 

## Cleaning and EDA 

### Cleaning
The cleaning pipeline is found in [clean.py](https://github.com/JCurley10/supervised-learning-case-study/blob/master/clean.py)

#### Adjusting Data Types:
- `phone`: Replaced "Android" with 1 and "iPhone" with 2
- `signup_date`:  The original data was a string; we converted from a string to a datetime 
- `last_trip_date`: **Our Target**
  - First the data were converted from string to a datetime, then a boolean where "True" means they churned (the last trip date was before June 1), or "False" means they are retained (the last trip was after June 1, 2014)
  - Then the booleans were converted to integers, where 0 = the customer was retained and 1 = the customer churned.

#### Handling NAN Values:
`phone`,  `avg_rating_of_driver` `avg_rating_by_driver` had NAN values, so we decided to drop the rows that had at least one NAN value before we split our data. We decided not to impute these NAN's in the interest of time. In the end there were 33132 rows in the training dataset, `churn_train.csv` 

### EDA


## Analysis

### Models: Knn, Logistic Regression, Random Forest  
#### Knn

#### Logistic Regression
We ran a logistic regression using K-Fold cross validation with 10 folds. The beta values for each feature were: 

- avg_rating_by_driver: 0.0345
- avg_rating_of_driver: 0.1257
- avg_surge: -0.0011
- phone: 0.5629
- surge_pct: 1.0876
- trips_in_first_30_days: -0.0067
- luxury_car_user: -0.1051
- weekday_pct: -0.8486  

The following statistics were determined from performing the cross validation:   

| **Average Accuracy** | **Average Precision** | **Average Recall** | **Average MSE** | 
| :--------: | :-------: | :-------: | :------: |
| 0.673 | 0.690 | 0.807 | 0.323| 
| 67.3% of predicions were correct based on what they really should be | 69% probability that someone actually churned given they were predicted to have churned | 80.7% probability we predict someone to churn given they actually churned | Difference between estimated and actual values| <br>  

We also created a confusion matrix to visualize the results.  
![](images/log_cm.svg)

#### Random Forest

### Feature Importance Plots


## Conclusion
**Based on insights from the model, what plans do we propose to reduce churn?**
For our **random forest model**, there were a few features that were most important in predicting if a customer churned or not. The Top 2 features based on their importance were average distance traveled and percentage of weekday trips. So, we are proposing to incentivize riders who are traveling longer distances (i.e. longer than 10 miles since the mean was 5 miles per training dataset) as well as riders that utilize the rideshare service during business hours. The incentive could be 10% off the next rider’s ride fee if they take a rider longer than 10 miles or use the service during the work week (Monday - Friday 9am - 5pm, excluding holidays)

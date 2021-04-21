# Import the three supervised learning models from sklearn
import sklearn.metrics as sm
from sklearn.linear_model import  LinearRegression
from sklearn.preprocessing import RobustScaler,MinMaxScaler,QuantileTransformer
from sklearn.model_selection import train_test_split
from time import time
from pyspark.sql import SparkSession
import numpy as np
import matplotlib.pyplot as plt
import re
from Modelling.Regression.CollegeScoreCard.src.exploratoryFeatureAnalysis import corelationCoeffmatrix
from Modelling.Regression.CollegeScoreCard.src.clean_data import groupCIPdesc
import pandas as pd

def salaryGroup(df):

 df['EARN_MDN_HI_1YR'] = pd.to_numeric(df['EARN_MDN_HI_1YR'],errors='coerce')
 df['EARNINGS'] = df.EARN_MDN_HI_1YR
 df['CREDLEV'] = pd.to_numeric(df['CREDLEV'],errors='coerce')
 df = df.replace(np.nan, 0, regex=True)
 df = df.drop('EARN_MDN_HI_1YR', axis=1)
 return df


def normalizeAndSplitFeatures():
    ''' Normalize the features and split them for train and test
        inputs: Void
        return:
               - X_train: features training set
               - y_train: flag training set
               - X_test: features testing set
               - y_test: flag testing set
        '''
    spark = SparkSession.builder.appName('CollegeScoreCard').getOrCreate()
    # Read features from csv files
    college_features = spark.read.csv('features/*.csv', header=True, inferSchema=True)
    # Convert sprark Data frame to Pandas Data Frame

    college_features_df = college_features.toPandas()
    college_features_df = college_features_df.fillna(0)
    college_features_df = salaryGroup(college_features_df)
    features=groupCIPdesc(college_features_df)


    features=features[['UNITID','ADMIN_TYPE','CREDLEV','STEM','EARNINGS']]

    # Remove unnecessary columns from features
    features_columns = features.describe().columns.tolist()

    processed_X_features = features.drop(['EARNINGS'], axis=1)
    processed_X_features = pd.get_dummies(processed_X_features)
    # print(processed_features.head())

    x = processed_X_features
    y = features['EARNINGS']


    # Split the 'features' into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

    # Return training and testing sets
    print("Finished normalizeAndSplitFeatures")
    return X_train, y_train,X_test, y_test

def train_predict(model, X_train, y_train, X_test, y_test):
    '''
    inputs:
       - learner: Learning algorithm for training and prediction
       - sample_size: the size of samples
       - X_train: features training set
       - y_train: flag training set
       - X_test: features testing set
       - y_test: flag testing set
    '''
    print("In train_predict")

    resultset = {}

    # Fit the learner to the training data using slicing with 'sample_size'
    start_time = time()  # Get start time
    # print(start_time)
    model = model.fit(X_train, y_train)
    end_time = time()  # Get end time
    # print(end_time)
    # Calculate the training time
    resultset['train_time'] = end_time - start_time
    print("\nTraining time ", resultset['train_time'])
    # Get the predictions on the test set,
    #       then get predictions on the first 300 training samples
    start_time = time()  # Get start time
    print(f'alpha = {model.intercept_}')
    print(f'betas = {model.coef_}')
     # Compute metrics on training set
    print("Training set score: {:.2f}".format(model.score(X_train, y_train)))
    y_train_pred = model.predict(X_train)
    y_train_pred = np.round(y_train_pred).astype(int)

    y_test_pred = model.predict(X_test)
    y_test_pred = np.round(y_test_pred).astype(int)
    end_time = time()  # Get end time

    # Calculate the total prediction time
    resultset['pred_time'] = end_time - start_time
    print("\nPrediction time ", resultset['pred_time'])
    y_test=np.array(y_test)

    # Compute metrics on test set

    print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
    print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2))
    print("Explain variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2))
    print("Test Set Score =", model.score(X_test, y_test))

    samplesize=30000

    plt.scatter(X_train['CREDLEV'][:samplesize], y_train[:samplesize], color='blue', marker='o')
    plt.plot(X_train['CREDLEV'][:samplesize], y_train_pred[:samplesize], color='red')
    plt.title('Credential Level Vs Earnings (Training set)')
    plt.xlabel('Credential Level')
    plt.ylabel('Earnings upon Graduation')
    plt.savefig("../FeatureAnalysisPlots/Earnings_vs_CredentialLvl_PredTrain.png", bbox_inches='tight')

    # Visualizing the Test set results
    plt.scatter(X_test['CREDLEV'][:samplesize], y_test[:samplesize], color='blue', marker='o')
    plt.plot(X_test['CREDLEV'][:samplesize], y_test_pred[:samplesize], color='red')
    plt.title('Credential Level Vs Earnings (Test set)')
    plt.xlabel('Credential Level')
    plt.ylabel('Earnings upon Graduation')
    plt.savefig("../FeatureAnalysisPlots/Earnings_vs_CredentialLvl_PredTest.png", bbox_inches='tight')

    # Success
    # print
    # "{} trained on {} samples.".format(model.__class__.__name__, sample_size)
    print('===================================================')
    print( "Regression Model - ",model.__class__.__name__)
    print('===================================================')

    #return the resultset with prediction metrics
    return resultset




def regresion_model_predictions():

    # get training and testing sets from features
    X_train, Y_train, X_test, Y_test = normalizeAndSplitFeatures()

    # Initialize the three models
    model2 = LinearRegression()

    resultset = {}
    resultset = train_predict(model2, X_train, Y_train, X_test, Y_test)
    return model2

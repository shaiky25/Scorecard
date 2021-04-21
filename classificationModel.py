# Import the three supervised learning models from sklearn
import sklearn.metrics as sm
from sklearn.linear_model import  LinearRegression
from sklearn.preprocessing import RobustScaler,MinMaxScaler,QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from time import time
from pyspark.sql import SparkSession
import numpy as np
import re
from Modelling.Regression.CollegeScoreCard.src.exploratoryFeatureAnalysis import corelationCoeffmatrix, plotAfterNormalization
from Modelling.Regression.CollegeScoreCard.src.clean_data import groupCIPdesc
import pandas as pd



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

    features=groupCIPdesc(college_features_df)
    features=features[['UNITID','ADMIN_TYPE','POSTGRAD','STEM','SALARY_GT_40']]

    # corelationCoeffmatrix(features.corr(), features, 'SALARY_GT_40')
    # Remove unnecessary columns from features
    features_columns = features.describe().columns.tolist()
    # print(features_columns)

    # scaler = RobustScaler()
    # scaler = MinMaxScaler() #good error
    scaler = QuantileTransformer()

    processed_features = features.copy()
    processed_features[features_columns] = scaler.fit_transform(processed_features[features_columns])

    processed_features = pd.get_dummies(processed_features)
    # processed_features.loc[processed_features.UNITID >= 0.5, "UNITID"] = 1
    # processed_features.loc[processed_features.UNITID < 0.5, "UNITID"] = 0
    processed_features[features_columns] = processed_features[features_columns].applymap(np.int64)
    features_df = processed_features.drop(['SALARY_GT_40'], axis=1)
    x = features_df
    y = processed_features['SALARY_GT_40']


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
    model = model.fit(X_train, y_train)
    end_time = time()  # Get end time
    # Calculate the training time
    resultset['train_time'] = end_time - start_time

    # Get the predictions on the test set,
    #       then get predictions on the training samples
    start_time = time()  # Get start time
    print(f'alpha = {model.intercept_}')
    print(f'betas = {model.coef_}')

    y_train_pred = model.predict(X_train)
    y_train_pred = np.where(y_train_pred<0.5,0,1)
    end_time = time()  # Get end time
    # Compute accuracy on training set
    print("==========")
    print("Accuracy Score for Training Set =", sm.accuracy_score(y_train, y_train_pred))
    start_time = time()
    y_test_pred = np.array(model.predict(X_test))
    y_test_pred = np.where(y_test_pred<0.5,0,1)
    end_time = time()  # Get end time
   # Calculate the total prediction time
    resultset['pred_time'] = end_time - start_time

    # Compute accuracy on testing set
    print("==========")
    print("Accuracy Score for Testing Set =", sm.accuracy_score(y_test, y_test_pred))

    # Success
    # print
    # "{} trained on {} samples.".format(model.__class__.__name__, sample_size)
    print('===================================================')
    print( "Classification Model - ",model.__class__.__name__)
    print('===================================================')

    #return the resultset with predection metrics
    return resultset


def classification_model_predictions():

    # get training and testing sets from features
    X_train, Y_train, X_test, Y_test = normalizeAndSplitFeatures()

    # Initialize the three models
    model1 = LinearRegression()

    resultset = {}
    resultset = train_predict(model1, X_train, Y_train, X_test, Y_test)
    return model1

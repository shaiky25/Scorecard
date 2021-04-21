import inline as inline
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import warnings
import xgboost as xgb
import lightgbm as lgb
from pyspark.sql import SparkSession
from scipy.stats import skew
from scipy import stats
from scipy.stats import norm

from Modelling.Regression.CollegeScoreCard.src.clean_data import _read_data

warnings.filterwarnings('ignore')
sns.set(style='white', context='notebook', palette='deep')
# %config InlineBackend.figure_format = 'retina' #set 'png' here when working on notebook
# %matplotlib inline
from pyspark.sql.functions import trim,when
FOLDER_PATH = 'data'
# Plot Histogram

def featureAnalysis(see_plots:bool):
    # spark = SparkSession.builder.appName('CollegeScoreCard').getOrCreate()
    # # Read features from csv files
    fieldOfStudy_df=_read_data('FIELD_OF_STUDY')
    # Convert sprark Data frame to Pandas Data Frame
    fieldOfStudy_df= fieldOfStudy_df.withColumn("CONTROL", when(fieldOfStudy_df["CONTROL"] == 'Public', 1)
                      .otherwise(0))
    df = fieldOfStudy_df.toPandas()
    df= df[(df['EARN_MDN_HI_1YR']!='NULL') & (df['EARN_MDN_HI_1YR']!='PrivacySuppressed')]
    df['EARN_MDN_HI_1YR'] = pd.to_numeric(df['EARN_MDN_HI_1YR'],errors='coerce')
    df['CREDLEV'] = pd.to_numeric(df['CREDLEV'],errors='coerce')
    df_copy=df.copy()
    print("\nBasic Statistics for Earnings data",df['EARN_MDN_HI_1YR'].describe())

    sns.distplot(df['EARN_MDN_HI_1YR'] , fit=norm);

    # Get the fitted parameters used by the function
    (mu, sigma) = norm.fit(df['EARN_MDN_HI_1YR'])
    print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
                loc='best')
    plt.ylabel('Frequency')
    plt.title('EARN_MDN_HI_1YR distribution')

    figure = plt.figure()
    result = stats.probplot(df['EARN_MDN_HI_1YR'], plot=plt)
    plt.savefig("../FeatureAnalysisPlots/EARNINGS_1_YR_Distribution.png", bbox_inches='tight')
    print("Skewness: %f" % df['EARN_MDN_HI_1YR'].skew())
    print("Kurtosis: %f" % df['EARN_MDN_HI_1YR'].kurt())

    totalFeatures(df_copy)
    correlationmatrix = df_copy[['UNITID','OPEID6','INSTNM','CONTROL','CREDLEV',\
                                'CIPCODE','CIPDESC','EARN_MDN_HI_1YR']].corr()

    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(correlationmatrix, vmax=.8, square=True);
    plt.savefig("../FeatureAnalysisPlots/CorrelationMatrixForDiffAttributes.png", bbox_inches='tight')
    corelationCoeffmatrix(correlationmatrix, df_copy, 'EARN_MDN_HI_1YR')
    plotJointGraph_CREDLEV(df_copy,'EARN_MDN_HI_1YR')
    if see_plots :
         plt.show()

def totalFeatures(df):
    cat = len(df.select_dtypes(include=['object']).columns)
    num = len(df.select_dtypes(include=['int64','float64']).columns)
    print('Total Features: ', cat, 'categorical', '+',
          num, 'numerical', '=', cat+num, 'features')

def corelationCoeffmatrix(correlationmatrix, df, target):
    number = 10 #number of variables for heatmap
    cols = correlationmatrix.nlargest(number, target)[target].index
    confMatrix = np.corrcoef(df[cols].values.T)
    sns.set(font_scale=1.25)
    f, ax = plt.subplots(figsize=(8, 6))
    f = sns.heatmap(confMatrix, cbar=False, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
    plt.savefig("../FeatureAnalysisPlots/CorelationCoeffmatrixForDiffAttributes.png",bbox_inches='tight')

def plotJointGraph_CREDLEV(df,target):
    var = 'CREDLEV'
    data = pd.concat([df[target], df[var]], axis=1)
    fig1, ax = plt.subplots(figsize=(8, 6))
    fig1 = sns.boxplot(x=df[var], y=df[target])
    fig1.axis(ymin=0, ymax=800000);
    plt.savefig("../FeatureAnalysisPlots/CREDLEV_vs_EARNINGS_1_YR_boxPlot.png",bbox_inches='tight')

    fig2, ax = plt.subplots(figsize=(8, 6))
    fig2=sns.jointplot(x=df[var], y=df[target], kind='reg')
    plt.savefig("../FeatureAnalysisPlots/CREDLEV_vs_EARNINGS_1_YR_jointPlot.png",bbox_inches='tight')

# Not used as of now
def plotAfterNormalization(df):
    # We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
    df["EARNINGS"] = np.log1p(df["EARNINGS"])

    #Check the new distribution
    sns.distplot(df['EARNINGS'] , fit=norm);

    # Get the fitted parameters used by the function
    (mu, sigma) = norm.fit(df['EARNINGS'])
    print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
                loc='best')
    plt.ylabel('Frequency')
    plt.title('EARNINGS distribution')

    fig = plt.figure()
    res = stats.probplot(df['EARNINGS'], plot=plt)
    plt.savefig("../FeatureAnalysisPlots/EARNINGS_ProbabilityPlot.png",bbox_inches='tight')

    y_train = df.EARNINGS.values

    print("Skewness: %f" % df['EARNINGS'].skew())
    print("Kurtosis: %f" % df['EARNINGS'].kurt())
    return y_train

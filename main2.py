import locale

from pyspark.sql import SparkSession
from sklearn.preprocessing import QuantileTransformer

from src.clean_data import build_dataframe
from src.feature_construction import construct_student_features
from src.classificationModel import classification_model_predictions, \
import clean_data
from src.exploratoryFeatureAnalysis import featureAnalysis
import regressionModel
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate


app = dash.Dash(__name__)
server = app.server


def getFeatures():
    spark = SparkSession.builder.appName('CollegeScoreCard').getOrCreate()
    # Read features from csv files
    field_of_study = spark.read.csv('features/*.csv', header=True, inferSchema=True)
    # Convert sprark Data frame to Pandas Data Frame

    field_of_study_df = field_of_study.toPandas()
    field_of_study_df = field_of_study_df.fillna(0)
    field_of_study_df = regressionModel.salaryGroup(field_of_study_df)
    field_of_study_df = clean_data.groupCIPdesc(field_of_study_df)

    #features = groupCIPdesc(field_of_study_df)
    features = field_of_study_df
    features = features[['UNITID', 'INSTNM', 'CONTROL', 'CREDLEV', 'CIPDESC', 'EARNINGS', 'ADMIN_TYPE', 'STEM','POSTGRAD']]

    print(features.head(5))
    return features

feature_df = getFeatures()
credentials = ['Undergraduate Certificates or Diplomas','Associate’s Degrees','Bachelor’s Degrees', \
               'Post-Baccalaureate Certificates','Graduate credentials','Master’s Degrees','Doctoral Degrees', \
               'First Professional Degrees','Graduate / Professional Certificates']
model1 = regressionModel.regresion_model_predictions()
# model2 = classification_model_predictions()
app.layout = html.Div([
   html.H2('College Score Card'),
   html.Label('College'),
   dcc.Dropdown(
       id='college',
       placeholder='Select College',
       options=[{'label': i, 'value': i} for i in feature_df.INSTNM.unique()],

   ),
   html.Br(),
   html.Div(id='control'),
   html.Br(),
   html.Label('Credential Level'),
   dcc.Dropdown(
       id='credential',
       placeholder='Select Credential Level',
       options=[{'label': credentials[i], 'value': i} for i in feature_df['CREDLEV'].unique()],

   ),
   html.Br(),
   html.Label('Academic Field'),
   dcc.Dropdown(
       id='cip',
       placeholder='Select Academic Field',
       options=[{'label': i, 'value': i} for i in feature_df['CIPDESC'].unique()],

   ),
   # html.Br(),
   # dcc.RadioItems(id='model_type',
   # options=[
   #     {'label': 'Linear Regression', 'value': '1'},
   #     {'label': 'Classification', 'value': '2'}
   # ],
   # value='1'
   # ),
   html.Br(),
   html.Br(),
   html.Button('Predict Salary', id='submit-val', n_clicks=0, style={'font-weight': 'bold'}),
   html.Br(),
   html.Br(),
   html.Div(id='earnings',style={'font-weight': 'bold'})
],style={'width': '50%'})
@app.callback(
    dash.dependencies.Output('earnings', 'children'),
    [dash.dependencies.Input('submit-val', 'n_clicks')],
    state=[State(component_id='college', component_property='value'),
            State(component_id='credential', component_property='value'),
            State(component_id='cip', component_property='value'),
            #State(component_id='model_type', component_property='value')
           ])
def update_output(n_clicks, college,  credential, cip):
   if college is None:
      raise PreventUpdate

   unit_id = feature_df.loc[feature_df['INSTNM'] == college, 'UNITID'].head(1).values
   control_id = feature_df.loc[feature_df['INSTNM'] == college, 'ADMIN_TYPE'].head(1).values
   cip_id = feature_df.loc[feature_df['CIPDESC'] == cip, 'STEM'].head(1).values
   feature = pd.DataFrame()
   #feature = [unit_id,control_id,credential_lvl,cip_id]
   feature['UNITID'] = [unit_id]
   feature['ADMIN_TYPE'] = [control_id]
   feature['CREDLEV'] = [credential]
   feature['STEM'] = [cip_id]

   print(feature)
   print(feature.size)
   # if model_type == '1':

   predicted_salary = model1.predict(feature)
   predicted_salary = np.round(predicted_salary).astype(int)
   min_sal = predicted_salary - 10000
   min_sal = "{:,}".format(int(min_sal))
   max_sal = predicted_salary + 10000
   max_sal = "{:,}".format(int(max_sal))
   print(min_sal)
   print(max_sal)
   return ('Predicted Salary Range: ${} - ${}'.format(min_sal,max_sal))
   # else:
   #     scaler = QuantileTransformer()
   #     features_columns = ['UNITID','ADMIN_TYPE','POSTGRAD','STEM']
   #
   #     processed_features = feature.copy()
   #     processed_features[features_columns] = scaler.fit_transform(processed_features[features_columns])
   #
   #     processed_features = pd.get_dummies(processed_features)
   #     #predicted_salary = model2.predict(processed_features)
   #
   #     print(predicted_salary)
   #     salary = '<$40,000'
   #     if predicted_salary == '1':
   #         salary = '>$40,000'
   #
   #     return ('Predicted Salary : ' + salary)

@app.callback(
    dash.dependencies.Output('control', 'children'),
    [dash.dependencies.Input('college', component_property='value')])
def update_control(college):
   control = feature_df.loc[feature_df['INSTNM'] == college, 'CONTROL'].head(1).values
   return ('Control : '+control)


def main():
   # # 1. Feature Analysis
   # featureAnalysis(see_plots = False)
   # # 2. Clean data
   # field_of_study_df = build_dataframe('FIELD_OF_STUDY')
   # print("After Cleaning Field OF Study file ",field_of_study_df.count())
   #
   # # 2. Feature construction
   #
   # college_feature_df = construct_student_features(field_of_study_df)
   # print("No of features in DF",college_feature_df.count())
   #
   # # #
   #
   # college_feature_df.write.csv('features', mode='overwrite', header=True, nullValue=0)
   # print("Feature Construction Completed")


   app.run_server(debug=True)
   # regresion_model_predictions()

if __name__ == "__main__":
    main()

# # Run this app with `python app.py` and
# # visit http://127.0.0.1:8050/ in your web browser.
#
# import dash
# import dash_core_components as dcc
# import dash_html_components as html
# from dash.dependencies import Input, Output
# import plotly.express as px
# import pandas as pd
# import numpy as np
# from flask import Flask
# app = Flask(__name__)
#
# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
#
# # app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
#
# # df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminderDataFiveYear.csv')
#
# df = pd.read_csv('features/part-00000-564d082e-3fa3-4e75-9dde-908a834f6d4e-c000.csv')
#
# colors = {
#     'background': '#585141',
#     'text': '#7FDBFF'
# }
#
# credentials = ['Undergrad or Diplomas','Associate’s','Bachelor’s', \
#                'Post-Baccalaureate','Graduate','Master’s','Doctoral', \
#                'First Professional','Professional']
#
# df['Earning Salary'] = df['EARN_MDN_HI_1YR']
# df['Type of Administration'] = df['CONTROL']
#
# def credential(df):
#      df['Credential Level'] = df['CREDLEV']
#      df['CREDLEV'] = pd.to_numeric(df['CREDLEV'],errors='coerce')
#      df = df.replace(np.nan, 0, regex=True)
#      df.loc[(df.CREDLEV==1), "Credential Level"] = credentials[0]
#      df.loc[(df.CREDLEV==2), "Credential Level"] = credentials[1]
#      df.loc[(df.CREDLEV==3), "Credential Level"] = credentials[2]
#      df.loc[(df.CREDLEV==4), "Credential Level"] = credentials[3]
#      df.loc[(df.CREDLEV==5), "Credential Level"] = credentials[4]
#      df.loc[(df.CREDLEV==6), "Credential Level"] = credentials[5]
#      df.loc[(df.CREDLEV==7), "Credential Level"] = credentials[6]
#      df.loc[(df.CREDLEV==8), "Credential Level"] = credentials[7]
#      return df
#
# df['year']=df['CREDLEV']
#
# df = credential(df)
#
#
# app.layout = html.Div([
#     dcc.Graph(id='graph-with-slider'),
#     dcc.Slider(
#         id='year-slider',
#         min=df['year'].min(),
#         max=df['year'].max(),
#         value=df['year'].min(),
#         marks={str(year): str(year) for year in df['year'].unique()},
#         step=None
#     )
# ])
#
#
# @app.callback(
#     Output('graph-with-slider', 'figure'),
#     Input('year-slider', 'value'))
# def update_figure(selected_year):
#     filtered_df = df[df.year == selected_year]
#
#     fig = px.scatter(filtered_df, x="Earning Salary", y="Type of Administration",
#                  size="Earning Salary", color="Credential Level",
#                      log_x=True, size_max=55)
#
#     fig.update_layout(transition_duration=500)
#
#     return fig
#
#
# if __name__ == '__main__':
#     app.run(debug=False)

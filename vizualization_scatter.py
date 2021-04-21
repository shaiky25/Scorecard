# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
import numpy as np
import glob
# files = glob.glob("src/features/*.csv")
#
# df = pd.DataFrame()
# for f in files:
#     csv = pd.read_csv(f)
#     df = df.append(csv)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# df = pd.read_csv('https://gist.githubusercontent.com/chriddyp/5d1ea79569ed194d432e56108a04d188/raw/a9f9e8076b837d541398e999dcbac2b2826a81f8/gdp-life-exp-2007.csv')
df = pd.read_csv('src/features/part-00000-564d082e-3fa3-4e75-9dde-908a834f6d4e-c000.csv')
colors = {
    'background': '#585141',
    'text': '#7FDBFF'
}

credentials = ['Undergrad or Diplomas','Associate’s','Bachelor’s', \
               'Post-Baccalaureate','Graduate','Master’s','Doctoral', \
               'First Professional','Professional']
def credential(df):
 df['Credential Level'] = df['CREDLEV']
 df['CREDLEV'] = pd.to_numeric(df['CREDLEV'],errors='coerce')
 df = df.replace(np.nan, 0, regex=True)
 df.loc[(df.CREDLEV==1), "Credential Level"] = credentials[0]
 df.loc[(df.CREDLEV==2), "Credential Level"] = credentials[1]
 df.loc[(df.CREDLEV==3), "Credential Level"] = credentials[2]
 df.loc[(df.CREDLEV==4), "Credential Level"] = credentials[3]
 df.loc[(df.CREDLEV==5), "Credential Level"] = credentials[4]
 df.loc[(df.CREDLEV==6), "Credential Level"] = credentials[5]
 df.loc[(df.CREDLEV==7), "Credential Level"] = credentials[6]
 df.loc[(df.CREDLEV==8), "Credential Level"] = credentials[7]
 return df


df['Earning Salary'] = df['EARN_MDN_HI_1YR']
df['Type of Administration'] = df['CONTROL']
df = credential(df)
# no_of_features=df['UNITID'].count().__str__()
no_of_features=''
fig1 = px.scatter(df, x="Earning Salary", y="Type of Administration",
                 size="Earning Salary", color="Credential Level", #hover_name="Earning Salary",
                 log_x=False, size_max=60)

fig1.update_layout(
    plot_bgcolor=colors['background'],
    paper_bgcolor=colors['background'],
    font_color=colors['text']
)


app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.H1(
        children='Earnings Data Vs Credential Level',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }
    ),

    html.Div(children='Total Number of features used for Prediction:'+no_of_features, style={
        'textAlign': 'center',
        'color': colors['text']
    }),

    dcc.Graph(
        id='admin type vs earnings',
        figure=fig1
    )

])
#
# if __name__ == '__main__':
#     app.run_server(debug=True)

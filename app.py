# import numpy as np
import pandas as pd

from joblib import load
import plotly.express as px

import dash_bootstrap_components as dbc
from dash import Dash, html, Input, Output, dcc


app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

linear_model = load("linear_model.joblib")

df = pd.read_csv("new_data.csv", usecols = ['X0', 'X1', 'X2' ,'y'])

app.layout = html.Div(children=[
    html.H6("Choisir une abscisse :"),

    html.Div([
        dcc.Dropdown(
                df.columns.drop('y'),
                'X0',
                id='xaxis-column'
            )
    ], style={'width': '48%', 'display': 'inline-block'}),

    html.Br(),

    dcc.Graph(id='scatter-plot'),

    # 5. Aligner les 3 sliders sur la même ligne
    html.Div([
        html.Div([
            html.H6("X0"),

            dcc.Slider(
                id='x0-slider',
                min=int(df['X0'].min()),
                max=int(df['X0'].max()),
                value=int(df['X0'].min()),
                marks={str(i): str(i) for i in range(int(df['X0'].min()), int(df['X0'].max())+1)},
                step=None
            ),
        ], style={'width': '32%', 'display': 'inline-block'}),
        html.Div([
            html.H6("X1"),
            
            dcc.Slider(
                id='x1-slider',
                min=int(df['X1'].min()),
                max=int(df['X1'].max()),
                value=int(df['X1'].min()),
                marks={str(i): str(i) for i in range(int(df['X1'].min()), int(df['X1'].max())+1)},
                step=None
            ),
        ], style={'width': '32%', 'display': 'inline-block'}),
        html.Div([
            html.H6("X2"),

            dcc.Slider(
                id='x2-slider',
                min=int(df['X2'].min()),
                max=int(df['X2'].max()),
                value=int(df['X2'].min()),
                marks={str(i): str(i) for i in range(int(df['X2'].min()), int(df['X2'].max())+1)},
                step=None
            ),
        ], style={'width': '32%', 'display': 'inline-block'}),
    ]),

    html.Br(),

    html.Div([
        html.Div(id='y_pred'),
    ]),

    dcc.Graph(id='histogram'),

])



# Questions

# 1. Définir un callback permettant de tracer un scatter plot (avec px.scatter(...))
# avec en abscisses une colonne Xi du dataframe df au choix de l'utilisateur
# (définie par un dropdown placé dans le layout ci-dessus)
# et en ordonnées la colonne y du dataframe df

# 4. Modifier le graphe du 1er callback pour qu'il affiche le point (Xi, y_pred),
# où y_pred est calculé dans la question 2.

@app.callback(
    Output(component_id='scatter-plot', component_property='figure'),
    Input(component_id='xaxis-column', component_property='value'),
    Input(component_id='x0-slider', component_property='value'),
    Input(component_id='x1-slider', component_property='value'),
    Input(component_id='x2-slider', component_property='value')
)
def update_graph(x_axis, X0, X1, X2):
    fig = px.scatter(df, x=x_axis, y='y')
    y_chap = linear_model.predict([[X0, X1, X2]])
    if x_axis == 'X0':
        fig.add_scatter(x=[X0], y=[y_chap[0]], mode='markers', marker=dict(size=10, color='red'), name='y_pred')
    elif x_axis == 'X1':
        fig.add_scatter(x=[X1], y=[y_chap[0]], mode='markers', marker=dict(size=10, color='red'), name='y_pred')
    elif x_axis == 'X2':
        fig.add_scatter(x=[X2], y=[y_chap[0]], mode='markers', marker=dict(size=10, color='red'), name='y_pred')
    return fig


# 2. Définir un 2e callback permettant, à partir des valeurs de X0, X1, X2
# choisies avec 3 sliders placés dans le layout ci-dessus
# de calculer la prévision de y correspondante (en utilisant linear_model)
# et de l'afficher dans le dashboard
@app.callback(
    Output(component_id='y_pred', component_property='children'),
    Input(component_id='x0-slider', component_property='value'),
    Input(component_id='x1-slider', component_property='value'),
    Input(component_id='x2-slider', component_property='value')
)
def update_y_pred(X0, X1, X2):
    y_chap = linear_model.predict([[X0, X1, X2]])
    return f"ŷ = {y_chap[0]}"

# Questions subsidiaires :
# 3. Tracer un histogramme de Xi (selon le choix du dropdown utilisé en 1.)
@app.callback(
    Output(component_id='histogram', component_property='figure'),
    Input(component_id='xaxis-column', component_property='value')
)
def update_graph(x_axis):
    return px.histogram(df, x=x_axis)


if __name__ == '__main__':
    app.run_server(debug=True)
